# from .detector3d_template import Detector3DTemplate
import time
import torch
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models import builder
from mmdet3d.ops import Voxelization
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.autograd import Variable
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import copy
import spconv.pytorch.functional as spF
import spconv.pytorch as spconv
from ...utils.spconv_utils import replace_feature
import torch.nn as nn
import numpy as np
# from mmdet3d.core.bbox import LiDARInstance3DBoxes
@DETECTORS.register_module(force=True)
class VoxelNeXtCoopertive(Base3DDetector):
    """this is the cooperaivte version of VoxelNeXt, 
    which is used to fuse the results of multiple agents in DAIR-V2X dataset."""
    def __init__(self, pts_voxel_layer,pts_voxel_encoder,
                backbone_3d, fusion_channels, dense_head, post_processing,\
                single=False, proj_first=False, raw=False, dairv2x=True,\
                train_cfg = None, test_cfg = None, **kwargs ):
                  #num_class, dataset):
        super(VoxelNeXtCoopertive,self).__init__()
        # self.module_list = self.build_networks()
        self.single = single
        self.proj_first = proj_first
        self.raw = raw
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if backbone_3d:
            self.backbone_3d = builder.build_backbone(backbone_3d)
        if not self.single and not self.raw and self.backbone_3d:
            self.inf_backbone_3d = builder.build_backbone(backbone_3d)
        self.fuse_net = self.build_fusion_net(fusion_channels)
        if dense_head:
            train_cfg = train_cfg.pts if train_cfg else None
            dense_head.update(train_cfg=train_cfg)
            test_cfg = test_cfg.pts if test_cfg else None
            dense_head.update(test_cfg=test_cfg)
            self.dense_head = builder.build_head(dense_head)
        if post_processing:
            self.post_processing_cfg = post_processing
        self.dairv2x = dairv2x
    def build_fusion_net(self,channels=[512,384,256]):
        fusion_net = spconv.SparseSequential(
            spconv.SubMConv2d(channels[0], channels[1], 5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(True),
            spconv.SubMConv2d(channels[1],channels[2], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(True),
        )
        return fusion_net
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)

        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(pts)
        # about 23k voxels
        # voxels: N x 10 x4
        # num_points: N
        # distribution of num_points:[21519,1348,198,32,8,1,1,1,0,13]
        # coors: Nx4, last dim is batch
        batch_dict = dict()
        batch_dict['batch_size'] = len(pts)
        batch_dict['voxels'] = voxels
        batch_dict['voxel_coords'] =  coors
        batch_dict['voxel_num_points'] = num_points
        voxel_features = self.pts_voxel_encoder(batch_dict)
        # batch_size = coors[-1, 0] + 1
        # x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        return voxel_features # 23k x 4 features+ 23k x4 coords
    def extract_img_feat(self, img, img_metas):
        return None
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        voxel_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, voxel_feats)
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      infrastructure_points=None,
                      img=None,
                      infrastructure_img=None,
                      **kwargs
                      ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        batch_dict = dict()
        device = points[0].device
        "fuse gt_bboxes_3d to BXMX9"
        max_box_num = max([len(bboxes) for bboxes in gt_bboxes_3d])
        box_dim = gt_bboxes_3d[0].tensor.shape[-1]
        bboxes_3d_tensor = torch.stack([
            torch.cat([b.tensor, torch.zeros(max_box_num - len(b), box_dim).to(b.tensor.device)])
            for b in gt_bboxes_3d
        ]).to(device)
        # # get first 7 dims
        "first filter out -1 labels in DAIR-V2X dataset by FFNet"
        if self.dairv2x:
            if self.dense_head.num_class>1:
                for i,l in enumerate(gt_labels_3d):
                    gt_labels_3d[i][l == -1] = self.dense_head.num_class-1
            else:
                gt_bboxes_3d = [bboxes_3d_tensor[i][l != -1] for i,l in enumerate(gt_labels_3d)]
                gt_labels_3d = [l[l != -1] for l in gt_labels_3d]
        gt_labels_3d_tensor = torch.stack([
            torch.cat([l.float()+1, torch.zeros(max_box_num - len(l)).to(l.device)])
            for l in gt_labels_3d
        ]).to(device)
        bboxes_3d_tensor = torch.cat([bboxes_3d_tensor, gt_labels_3d_tensor.unsqueeze(-1)], dim=-1)
        batch_dict['gt_boxes'] = bboxes_3d_tensor
        if 'cooperative_agents' in img_metas[0].keys() and infrastructure_points is None:
            infrastructure_points = []
            for i in range(len(img_metas)): # we only support 2 agents
                agent = list(img_metas[i]['cooperative_agents'].keys())[0]
                cooperative_points = img_metas[i]['cooperative_agents'][agent]['points']
                if cooperative_points.device != device:
                    cooperative_points = cooperative_points.to(device)
                infrastructure_points.append(cooperative_points)
        if not self.single and self.proj_first:
            "project inf points to vehicle coordinate system in raw data"
            for i in range(len(infrastructure_points)):
                pcd_range = torch.tensor(self.pts_voxel_layer.point_cloud_range).to(device)
                rotatation_matrix = torch.tensor(img_metas[i]['inf2veh']['rotation']).to(device).float()
                translation = torch.tensor(img_metas[i]['inf2veh']['translation']).to(device).float()
                transfrom_matrix = torch.zeros((4,4)).to(device)
                transfrom_matrix[0:3,0:3] = rotatation_matrix
                transfrom_matrix[0:3,3] = translation.T
                transfrom_matrix[3,3] = 1
                points_inf_feat = infrastructure_points[i][:,3:]
                points_inf = infrastructure_points[i][:,:3]
                points_inf = torch.cat([points_inf,torch.ones(points_inf.shape[0],1).to(device)],dim=1)
                points_inf = transfrom_matrix@points_inf.T
                points_inf = points_inf.T
                points_inf = points_inf[:,0:3]
                mask = (points_inf[:,0] > pcd_range[0]) &\
                    (points_inf[:,0] < pcd_range[3]) &\
                    (points_inf[:,1] > pcd_range[1]) &\
                    (points_inf[:,1] < pcd_range[4]) &\
                    (points_inf[:,2] > pcd_range[2]) &\
                    (points_inf[:,2] < pcd_range[5])
                points_inf = points_inf[mask]
                points_inf_feat = points_inf_feat[mask]
                if len(points_inf) == 0:
                    points_inf = torch.zeros((1,3)).to(device)
                    points_inf_feat = torch.zeros((1,points_inf_feat.shape[1])).to(device)
                infrastructure_points[i] = torch.cat([points_inf,points_inf_feat],dim=1)
                if self.raw:
                    points[i] = torch.cat([points[i],infrastructure_points[i]],dim=0)
        for i,points_inf in enumerate(infrastructure_points):
            if len(points_inf) == 0:
                points_inf = torch.tensor([[46,0,-1.5,1.3e-3]]).to(device)
                infrastructure_points[i] = points_inf
        img_feats, voxel_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        voxel_feats.update(gt_boxes=bboxes_3d_tensor)
        pts_feats = self.backbone_3d(voxel_feats)
        if not self.single and not self.raw:
            img_feats_inf, voxel_feats_inf = self.extract_feat(
            infrastructure_points, img=infrastructure_img, img_metas=img_metas)
            voxel_feats_inf.update(gt_boxes=bboxes_3d_tensor)
            pts_feats_inf = self.inf_backbone_3d(voxel_feats_inf)
            pts_feats = self.feature_fusion(pts_feats, pts_feats_inf, img_metas)
            # if self.proj_first:
            #     pts_feats = self.feature_fusion(pts_feats, pts_feats_inf, img_metas)
            # else:
            #     pts_feats = self.feature_fusion_warp(pts_feats, pts_feats_inf, img_metas)
        batch_dict.update(pts_feats)
        output_dict = self.dense_head(batch_dict) # img_feats for future use
        losses, loss_dict = self.dense_head.get_loss()
        if 'loss_box_of_pts_sprs' in batch_dict.keys():
            loss_dict['loss_box_of_pts_sprs'] = batch_dict['loss_box_of_pts_sprs']
        return loss_dict
    
    def aug_test(self):
        pass

    def simple_test(self,points=None,img_metas=None,img=None,infrastructure_points=None,infrastructure_img=None,**kwargs):
        "we cannot project points in test pipelines, so we need to project them here"
        device = points[0].device
        if not self.single and infrastructure_img is not None: # dair-v2x cannot project points in test pipelines, but other datasets can
            "project inf points to vehicle coordinate system in raw data"
            device = points[0].device
            for i in range(len(infrastructure_points)):
                if not isinstance(infrastructure_points[0],torch.Tensor):
                    if isinstance(infrastructure_points[0],list):
                        infrastructure_points = infrastructure_points[0]
                pcd_range = torch.tensor(self.pts_voxel_layer.point_cloud_range).to(device)
                rotatation_matrix = torch.tensor(img_metas[i]['inf2veh']['rotation']).to(device).float()
                translation = torch.tensor(img_metas[i]['inf2veh']['translation']).to(device).float()
                transfrom_matrix = torch.zeros((4,4)).to(device)
                transfrom_matrix[0:3,0:3] = rotatation_matrix
                transfrom_matrix[0:3,3] = translation.T
                transfrom_matrix[3,3] = 1
                points_inf_feat = infrastructure_points[i][:,3:]
                points_inf = infrastructure_points[i][:,:3]
                points_inf = torch.cat([points_inf,torch.ones(points_inf.shape[0],1).to(device)],dim=1)
                points_inf = transfrom_matrix@points_inf.T
                points_inf = points_inf.T
                points_inf = points_inf[:,0:3]
                mask = (points_inf[:,0] > pcd_range[0]) &\
                    (points_inf[:,0] < pcd_range[3]) &\
                    (points_inf[:,1] > pcd_range[1]) &\
                    (points_inf[:,1] < pcd_range[4]) &\
                    (points_inf[:,2] > pcd_range[2]) &\
                    (points_inf[:,2] < pcd_range[5])
                points_inf = points_inf[mask]
                points_inf_feat = points_inf_feat[mask]
                if len(points_inf) == 0:
                    points_inf = torch.tensor([[46,0,-1.5]]).to(device)
                    points_inf_feat = 1.3e-3*torch.ones((1,points_inf_feat.shape[1])).to(device)
                infrastructure_points[i] = torch.cat([points_inf,points_inf_feat],dim=1)
                if self.raw:
                    points[i] = torch.cat([points[i],infrastructure_points[i]],dim=0)
                # infrastructure points: about 45k x 4, last dim is intensity
        if 'cooperative_agents' in img_metas[0].keys() and infrastructure_points is None:
            infrastructure_points = []
            for i in range(len(img_metas)): # we only support 2 agents
                agent = list(img_metas[i]['cooperative_agents'].keys())[0]
                cooperative_points = img_metas[i]['cooperative_agents'][agent]['points']
                if cooperative_points.device != device:
                    cooperative_points = cooperative_points.to(device)
                if len(cooperative_points) == 0:
                    cooperative_points = torch.tensor([[0,0,-1.5,1.3e-3]]).to(device)
                infrastructure_points.append(cooperative_points)
        for i,points_inf in enumerate(infrastructure_points):
            if len(points_inf) == 0:
                points_inf = torch.zeros((1,4)).to(device)
                infrastructure_points[i] = points_inf
        img_feats, voxel_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        pts_feats = self.backbone_3d(voxel_feats)
        if not self.single and not self.raw and len(infrastructure_points[0])>10:  # when there are too few points, the network will crash       
            "if infrastructure_points[0] is not torch.Tensor"
            if not isinstance(infrastructure_points[0],torch.Tensor):
                if isinstance(infrastructure_points[0],list):
                    infrastructure_points = infrastructure_points[0]
                else:
                    pts_feats_inf = None
            # infrastructure_points = [Variable(p,requires_grad=True) for p in infrastructure_points]
            """
            for p in self.parameters():
                p.requires_grad_(False)
            """
            img_feats_inf, voxel_feats_inf = self.extract_feat(
            infrastructure_points, img=infrastructure_img, img_metas=img_metas)
            "不确定要不要反向传播计算原始特征梯度"
            #voxel_feats_inf['voxel_features'] = Variable(voxel_feats_inf['voxel_features'],requires_grad=True)
            "voxel_feats_inf['voxel_features']=voxel_feats_inf['voxel_features'].requires_grad_(True)"
            pts_feats_inf = self.inf_backbone_3d(voxel_feats_inf)
            # self.inf_backbone_3d.train()
            # with torch.no_grad():
            #     pts_feats_inf = self.inf_backbone_3d(voxel_feats_inf)
            """
            torch.autograd.detect_anomaly()
            s = torch.sum(pts_feats_inf['encoded_spconv_tensor'].features)
            s.backward()
            """
            #pts_feats_inf['encoded_spconv_tensor'].features.backward(torch.ones_like(pts_feats_inf['encoded_spconv_tensor'].features))
            
            # if self.proj_first:
            #     pts_feats = self.feature_fusion(pts_feats, pts_feats_inf, img_metas)
            # else:
            #     pts_feats = self.feature_fusion_warp(pts_feats, pts_feats_inf, img_metas)
            pts_feats = self.feature_fusion(pts_feats, pts_feats_inf, img_metas)
        # img_feats_inf, pts_feats_inf = self.extract_feat(
        #     infrastructure_points, img=infrastructure_img, img_metas=img_metas)
        # if pts_feats_inf is not None:
        #     pts_feats = self.feature_fusion(pts_feats, pts_feats_inf, img_metas)
        batch_dict = pts_feats
        output_dict = self.dense_head(batch_dict)
        # pred_dicts, recall_dicts = self.post_processing(batch_dict)
        bbox_list = output_dict['final_box_dicts'] 
        for i in range(len(bbox_list)):
            bbox_list[i]['pred_labels'] = bbox_list[i]['pred_labels']-1
        result_list = []
        for i,bbox in enumerate(bbox_list):
            result_list.append(dict())
            result_box = dict()
            result_box['boxes_3d'] = self.get_boxes(img_metas,bbox['pred_boxes'].cpu())
            result_box['scores_3d'] = bbox['pred_scores'].cpu()
            result_box['labels_3d'] = bbox['pred_labels'].cpu()
            result_list[i]['pts_bbox'] = result_box
            result_list[i]['img_metas'] = img_metas
            result_list[i]['boxes_3d'] = result_box['boxes_3d']
            result_list[i]['scores_3d'] = result_box['scores_3d']
            result_list[i]['labels_3d'] = result_box['labels_3d']
        return result_list  

    def feature_fusion_add(self,pts_feats, pts_feats_inf, img_metas):
        batch_size = pts_feats['batch_size']
        # pts_feats['encoded_spconv_tensor'] = pts_feats['encoded_spconv_tensor']\
        # + pts_feats_inf['encoded_spconv_tensor']
        pts_feats['encoded_spconv_tensor']=spF.sparse_add(pts_feats['encoded_spconv_tensor'],pts_feats_inf['encoded_spconv_tensor'])
        # spF.sparse_add_hash_based(pts_feats['encoded_spconv_tensor'],pts_feats_inf['encoded_spconv_tensor'])
        # grid_size = torch.tensor(pts_feats_inf['encoded_spconv_tensor']\
        #                          .spatial_shape[::-1])
        # fused_feat = torch.cat([pts_feats['encoded_spconv_tensor'].features,\
        #         pts_feats_inf['encoded_spconv_tensor'].features],dim=0)
        # fused_coords = torch.cat([pts_feats['encoded_spconv_tensor'].indices,\
        #         pts_feats_inf['encoded_spconv_tensor'].indices],dim=0)
        # fused_sparse_feat = spconv.SparseConvTensor(fused_feat,fused_coords,[grid_size[1],grid_size[0]],batch_size)
        # pts_feats['encoded_spconv_tensor'] = fused_sparse_feat
        
        return pts_feats
    def feature_fusion(self,pts_feats, pts_feats_inf, img_metas):
        batch_size = pts_feats['batch_size']
        # pts_feats['encoded_spconv_tensor'] = pts_feats['encoded_spconv_tensor']\
        # + pts_feats_inf['encoded_spconv_tensor']
        "cat zero and add"
        pts_feats['encoded_spconv_tensor']= replace_feature(pts_feats['encoded_spconv_tensor'],\
                        torch.cat([pts_feats['encoded_spconv_tensor'].features,\
                                   torch.zeros_like(pts_feats['encoded_spconv_tensor'].features)],dim=1))
        
        pts_feats_inf['encoded_spconv_tensor']= replace_feature(pts_feats_inf['encoded_spconv_tensor'],\
                        torch.cat([torch.zeros_like(pts_feats_inf['encoded_spconv_tensor'].features),\
                                  pts_feats_inf['encoded_spconv_tensor'].features ],dim=1))                                                         
        pts_feats['encoded_spconv_tensor']=spF.sparse_add(pts_feats['encoded_spconv_tensor'],pts_feats_inf['encoded_spconv_tensor'])
        pts_feats['encoded_spconv_tensor']= self.fuse_net(pts_feats['encoded_spconv_tensor'])
        # spF.sparse_add_hash_based(pts_feats['encoded_spconv_tensor'],pts_feats_inf['encoded_spconv_tensor'])
        # grid_size = torch.tensor(pts_feats_inf['encoded_spconv_tensor']\
        #                          .spatial_shape[::-1])
        # fused_feat = torch.cat([pts_feats['encoded_spconv_tensor'].features,\
        #         pts_feats_inf['encoded_spconv_tensor'].features],dim=0)
        # fused_coords = torch.cat([pts_feats['encoded_spconv_tensor'].indices,\
        #         pts_feats_inf['encoded_spconv_tensor'].indices],dim=0)
        # fused_sparse_feat = spconv.SparseConvTensor(fused_feat,fused_coords,[grid_size[1],grid_size[0]],batch_size)
        # pts_feats['encoded_spconv_tensor'] = fused_sparse_feat
        
        return pts_feats

    def feature_fusion_warp(self,pts_feats, pts_feats_inf, img_metas):
        voxel_coords_inf = copy.deepcopy(pts_feats_inf['encoded_spconv_tensor'].indices)#copy.deepcopy(pts_feats_inf['voxel_coords'])
        voxel_coords_inf = voxel_coords_inf.float()
        device = voxel_coords_inf.device
        batch_size = pts_feats['batch_size']
        feats_inf = pts_feats_inf['encoded_spconv_tensor'].features
        "first reverse from BYX to b=BXY"
        voxel_coords_inf = voxel_coords_inf[:,[0,2,1]]
        pcd_range = torch.tensor(self.pts_voxel_layer.point_cloud_range).to(device)
        grid_size = torch.tensor(pts_feats_inf['encoded_spconv_tensor']\
                                 .spatial_shape[::-1]).to(device)
        voxel_size = (pcd_range[3:5]-pcd_range[0:2])/grid_size.float()
        "from int indice to float coords, the center of the voxels"
        voxel_coords_inf[:,1:] = (voxel_coords_inf[:,1:]+0.5) * voxel_size + pcd_range[0:2]
        # N X 4 batch_idx, z,y,x
        
        fused_feat = [pts_feats['encoded_spconv_tensor'].features]
        fused_coords = [pts_feats['encoded_spconv_tensor'].indices]
        for i in range(batch_size):
            
            # warp_feat_trans = F.grid_sample(inf_feature, grid_r_t, mode='bilinear', align_corners=False)
            # wrap_feats_ii.append(warp_feat_trans)
            "get the rotation and translation matrix"
            rotatation_matrix = torch.tensor(img_metas[i]['inf2veh']['rotation']).to(device).float()
            translation = torch.tensor(img_metas[i]['inf2veh']['translation']).to(device).float()
            "only in xy plane"
            rotatation_matrix = rotatation_matrix[0:2,0:2]
            translation = translation[0:2]
            transform_matrix = torch.zeros((3,3)).to(device)
            transform_matrix[0:2,0:2] = rotatation_matrix
            transform_matrix[0:2,2] = translation.T
            validate = False
            #######
            if validate:
                "original code of FFNet in NIPS2023"
                calib_inf2veh_rotation = img_metas[i]['inf2veh']['rotation']
                calib_inf2veh_translation = img_metas[i]['inf2veh']['translation']
                inf_pointcloud_range = pcd_range

                theta_rot = torch.tensor([[calib_inf2veh_rotation[0][0], -calib_inf2veh_rotation[0][1], 0.0],
                                            [-calib_inf2veh_rotation[1][0], calib_inf2veh_rotation[1][1], 0.0],
                                            [0,0,1]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
                theta_rot= torch.FloatTensor(self.generate_matrix(theta_rot,-1,0)).type(dtype=torch.float).cuda(next(self.parameters()).device)
                # Moving right and down is negative.
                x_trans = -2 * calib_inf2veh_translation[0][0] / (inf_pointcloud_range[3] - inf_pointcloud_range[0])
                y_trans = -2 * calib_inf2veh_translation[1][0] / (inf_pointcloud_range[4] - inf_pointcloud_range[1])
                theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans],[0.0, 0.0 , 1]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
                theta_r_t=torch.mm(theta_rot,theta_trans, out=None)
                grid_r_t = F.affine_grid(theta_r_t[0:2].unsqueeze(0), size=[1,128,grid_size[0],grid_size[1]], align_corners=False)
                "grid_r_t: 原BEV特征图归一化到坐标[-1,-1]到[1,1]之间,乘以grid_size即可得到新坐标系下坐标"
                x=copy.deepcopy(pcd_range[0:2])
                x+=0.5*voxel_size
                x=x-translation.T
                x=x.matmul(rotatation_matrix)
                x-=torch.tensor([46.08,0]).to(device)
                x=x/46.08
                print(x-grid_r_t[0,0,0])
                y=copy.deepcopy(pcd_range[3:5])
                y-=0.5*voxel_size
                y=y-translation.T
                y=y.matmul(rotatation_matrix)
                y-=torch.tensor([46.08,0]).to(device)
                y=y/46.08
                print(y-grid_r_t[0,-1,-1])
            #######
            "rotate and translate"
            voxel_coords_inf_b = copy.deepcopy(voxel_coords_inf[voxel_coords_inf[:,0]==i])
            feats_inf_b = feats_inf[voxel_coords_inf[:,0]==i]
            voxel_coords_inf_b = voxel_coords_inf_b[:,1:] # Nx2
            voxel_coords_inf_b = torch.cat([voxel_coords_inf_b,torch.ones(voxel_coords_inf_b.shape[0],1).to(device)],dim=1)
            voxel_coords_inf_b = transform_matrix.matmul(voxel_coords_inf_b.T).T # ((3X3)X(3XN))' = Nx3
            voxel_coords_inf_b = voxel_coords_inf_b[:,0:2]
            voxel_coords_inf_b = torch.cat([i*torch.ones(voxel_coords_inf_b.shape[0],1).to(device),voxel_coords_inf_b],dim=1)
            # voxel_coords_inf_b[:,1:] = voxel_coords_inf_b[:,1:] - translation.T
            # voxel_coords_inf_b[:,1:] = voxel_coords_inf_b[:,1:].matmul(rotatation_matrix)
            "mask the voxels out of ego vehicle's range"            
            mask = (voxel_coords_inf_b[:,1] >= 0) & (voxel_coords_inf_b[:,1] < pcd_range[3]) & \
                (voxel_coords_inf_b[:,2] >= 0) & (voxel_coords_inf_b[:,2] < pcd_range[4])
            voxel_coords_inf_b = voxel_coords_inf_b[mask]
            feats_inf_b = feats_inf_b[mask]
            "from float coords to int indice"
            voxel_coords_inf_b[:,1:] = (voxel_coords_inf_b[:,1:] - pcd_range[0:2]) / voxel_size - 0.5

            # voxel_coords_inf_b = voxel_coords_inf_b.int()
            voxel_coords_inf_b = voxel_coords_inf_b[:,[0,2,1]].int()
            fused_feat.append(feats_inf_b)
            fused_coords.append(voxel_coords_inf_b)
        "concatenate the two voxel features"
        new_features = torch.cat(fused_feat,dim=0)
        new_indices = torch.cat(fused_coords,dim=0)
        fused_sparse_feat = spconv.SparseConvTensor(new_features,new_indices,[grid_size[1],grid_size[0]],batch_size)
        pts_feats['encoded_spconv_tensor'] = fused_sparse_feat
        # pts_feats['voxel_coords'] = torch.cat([pts_feats['voxel_coords'],voxel_coords_inf],dim=0)
        # pts_feats['voxel_num_points'] = torch.cat([pts_feats['voxel_num_points'],pts_feats_inf['voxel_num_points']],dim=0)
        return pts_feats
    def generate_matrix(self,theta,x0,y0):
        c = theta[0][0]
        s = theta[1][0]
        matrix = np.zeros((3,3))
        matrix[0,0] = c
        matrix[0,1] = -s
        matrix[1,0] = s
        matrix[1,1] = c
        matrix[0,2] = -c * x0 + s * y0 + x0
        matrix[1,2] =  -c * y0 - s * x0 + y0
        matrix[2,2] = 1
        return matrix
    def get_boxes(self,img_metas,box):
        box = img_metas[0]['box_type_3d'](box,box_dim=box.shape[-1],with_yaw=True)
        return box
    def post_processing(self, batch_dict):
        post_process_cfg = self.post_processing_cfg
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.recall_thresh_list
            )

        return final_pred_dict, recall_dict
    
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

