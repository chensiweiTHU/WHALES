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
from .import VoxelNeXtCoopertive

class PointQuantization(object):
    def __init__(self, voxel_size, quantize_coords_range, q_delta=1.6e-5):
        self.voxel_size = np.array(voxel_size)
        self.quantize_coords_range = quantize_coords_range
        self.low_bound = np.array(quantize_coords_range[:3])
        self.q_delta = q_delta
    # 2340x2304x16 = 8.5e7 log2(8.5e7) = 26.4 4+1=5B for each point
    def __call__(self, points):
        device = points.device
        low_bound = torch.tensor(self.low_bound).to(device)
        voxel_size = torch.tensor(self.voxel_size).to(device)
        points[:, :3] -= (low_bound + voxel_size / 2)
        points[:, :3] = torch.round(points[:, :3] / voxel_size)
        points[:, :3] *= voxel_size
        points[:, :3] += (low_bound + voxel_size / 2)
        "we assume that intensity is always ≥ 0 and quantisize uniformly"
        points[:,3] = torch.round(points[:,3]/self.q_delta)*self.q_delta
        return points

@DETECTORS.register_module(force=True)
class (VoxelNeXtCoopertive):
    """this is the cooperaivte version of VoxelNeXt, 
    which is used to fuse the results of multiple agents in DAIR-V2X dataset."""
    def __init__(self, pts_voxel_layer,pts_voxel_encoder,
                  backbone_3d, fusion_channels, dense_head, post_processing, single=False,proj_first=False,raw=False, pruning=None, quant_levels=[[0.04, 0.04, 0.0625],[0.16, 0.16, 0.25]], **kwargs ):
                  #num_class, dataset):
        super(,self).__init__(pts_voxel_layer,pts_voxel_encoder,
                  backbone_3d, fusion_channels, dense_head, post_processing, single=single,proj_first=proj_first,raw=raw, **kwargs)
        if pruning is not None:
            self.pruning = builder.build_backbone(pruning) # we put the pruning block in the backbone
        # self.pruning.bound_backbone(self.inf_backbone_3d)
        self.quant_levels = quant_levels
        self.point_cloud_range = pts_voxel_layer.point_cloud_range
        self.pointQuantization = []
        for quant_level in self.quant_levels:
            self.pointQuantization.append(PointQuantization(quant_level,self.point_cloud_range))
            
    def quantisize(self,points_with_levels,index):
        # quantisized_points = []
        # for i,points in enumerate(points_with_levels):
        #     quantisized_points.append(self.pointQuantization[i](points))
        return self.pointQuantization[index](points_with_levels)
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
        batch_size = len(points)
        "fuse gt_bboxes_3d to BXMX9"
        max_box_num = max([len(bboxes) for bboxes in gt_bboxes_3d])
        box_dim = gt_bboxes_3d[0].tensor.shape[-1]
        bboxes_3d_tensor = torch.stack([
            torch.cat([b.tensor, torch.zeros(max_box_num - len(b), box_dim).to(b.tensor.device)])
            for b in gt_bboxes_3d
        ]).to(device)
        # # get first 7 dims
        "first filter out -1 labels in DAIR-V2X dataset by FFNet"
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
            
            #######use pruning to choose important points#########
            "we use pruning to choose important points"
            voxel_feats_inf.update(gt_boxes=bboxes_3d_tensor)
            important_coords, unimportant_coords,important_voxels, unimportant_voxels, voxel_feats_inf = self.pruning(voxel_feats_inf)
            # "from coords to points"
            # pcd_range = torch.tensor(self.pts_voxel_layer.point_cloud_range).to(device)
            # voxel_size = torch.tensor(self.pts_voxel_layer.voxel_size).to(device)
            compressed_points_inf = []

            "voxels contains points in the voxel,first reshape into -1,4"
            important_points = important_voxels.reshape(-1,important_voxels.shape[-1]) # Nxmax_points_per_voxelx4 -> N*max_points_per_voxelx4
            unimportant_points = unimportant_voxels.reshape(-1,unimportant_voxels.shape[-1]) # Nxmax_points_per_voxelx4 -> N*max_points_per_voxelx4
            for b_id in range(batch_size):   

                important_points_bid = important_points[important_coords[:,0] == b_id]
                unimportant_points_bid = unimportant_points[unimportant_coords[:,0] == b_id]

                important_points_bid = self.quantisize(important_points_bid,0)
                unimportant_points_bid = self.quantisize(unimportant_points_bid,1)

                points_bid = torch.cat([important_points_bid,unimportant_points_bid],dim=0)
                compressed_points_inf.append(points_bid)
                # important_coords_bid = important_coords[important_coords[:,0] == b_id][:,1:] # ZYX
                # "from ZYX to XYZ"
                # important_coords_bid = important_coords_bid[:,[2,1,0]]
                # important_points_bid = important_coords_bid*voxel_size + pcd_range[0:3]
                # important_points_inf.append(important_points_bid)

                # unimportant_coords_bid = unimportant_coords[unimportant_coords[:,0] == b_id][:,1:] # ZYX
                # "from ZYX to XYZ"
                # unimportant_coords_bid = unimportant_coords_bid[:,[2,1,0]]
                # unimportant_points_bid = unimportant_coords_bid*voxel_size + pcd_range[0:3]
                # unimportant_points_bid = self.quantisize(unimportant_points_bid,1)
                # unimportant_points_inf.append(unimportant_points_bid)

            # unimportant_points_inf = self.quantisize(unimportant_points_inf,1)
            # for b_id in range(len(important_points_inf)):
            #     important_points_inf[b_id] = torch.cat([important_points_inf[b_id],unimportant_points_inf[b_id]],dim=0)
            # points_inf = important_points_inf
            loss_box_of_pts_sprs = voxel_feats_inf['loss_box_of_pts_sprs']
            img_feats_inf, voxel_feats_inf = self.extract_feat(
            compressed_points_inf, img=infrastructure_img, img_metas=img_metas)
            voxel_feats_inf.update(gt_boxes=bboxes_3d_tensor)
            voxel_feats_inf.update(loss_box_of_pts_sprs_pruning=loss_box_of_pts_sprs)
            #######use pruning to choose important points#########
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
            loss_dict['loss_box_of_pts_sprs_pruning'] = loss_box_of_pts_sprs
        return loss_dict
    
    def simple_test(self,points=None,img_metas=None,img=None,infrastructure_points=None,infrastructure_img=None,**kwargs):
        batch_size = len(points)
        "we cannot project points in test pipelines, so we need to project them here"
        if not self.single: 
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
        for i,points_inf in enumerate(infrastructure_points):
            if len(points_inf) == 0:
                points_inf = torch.zeros((1,4)).to(device)
                infrastructure_points[i] = points_inf
        img_feats, voxel_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        pts_feats = self.backbone_3d(voxel_feats)
        if not self.single and not self.raw:        
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
            #######use pruning to choose important points#########
            "we use pruning to choose important points"
            important_coords, unimportant_coords,important_voxels, unimportant_voxels, voxel_feats_inf = self.pruning(voxel_feats_inf)
            # "from coords to points"
            compressed_points_inf = []
            "voxels contains points in the voxel,first reshape into -1,4"
            important_points = important_voxels.reshape(-1,important_voxels.shape[-1]) # Nxmax_points_per_voxelx4 -> N*max_points_per_voxelx4
            unimportant_points = unimportant_voxels.reshape(-1,unimportant_voxels.shape[-1]) # Nxmax_points_per_voxelx4 -> N*max_points_per_voxelx4
            for b_id in range(batch_size):   

                important_points_bid = important_points[important_coords[:,0] == b_id]
                unimportant_points_bid = unimportant_points[unimportant_coords[:,0] == b_id]

                important_points_bid = self.quantisize(important_points_bid,0)
                unimportant_points_bid = self.quantisize(unimportant_points_bid,1)

                points_bid = torch.cat([important_points_bid,unimportant_points_bid],dim=0)
                compressed_points_inf.append(points_bid)
            img_feats_inf, voxel_feats_inf = self.extract_feat(
            compressed_points_inf, img=infrastructure_img, img_metas=img_metas)
            #######use pruning to choose important points#########
            pts_feats_inf = self.inf_backbone_3d(voxel_feats_inf)
            pts_feats = self.feature_fusion(pts_feats, pts_feats_inf, img_metas)
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
