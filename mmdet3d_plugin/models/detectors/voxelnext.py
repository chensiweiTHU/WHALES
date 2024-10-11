# from .detector3d_template import Detector3DTemplate
import time
import torch
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models import builder
from mmdet3d.ops import Voxelization
from mmcv.runner import force_fp32
from torch.nn import functional as F
from pcdet.ops.iou3d_nms import iou3d_nms_utils
# from mmdet3d.core.bbox import LiDARInstance3DBoxes
@DETECTORS.register_module(force=True)
class VoxelNeXt(Base3DDetector):
    def __init__(self, pts_voxel_layer,pts_voxel_encoder,
                  backbone_3d, dense_head, post_processing, **kwargs ):
                  #num_class, dataset):
        super(VoxelNeXt,self).__init__()
        # self.module_list = self.build_networks()
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if backbone_3d:
            self.backbone_3d = builder.build_backbone(backbone_3d)
        if dense_head:
            self.dense_head = builder.build_head(dense_head)
        if post_processing:
            self.post_processing_cfg = post_processing
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
    def extract_pts_feat(self, pts, img_feats, img_metas,gt_boxes=None):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(pts)
        batch_dict = dict()
        batch_dict['batch_size'] = len(pts)
        batch_dict['voxels'] = voxels
        batch_dict['voxel_coords'] =  coors
        batch_dict['voxel_num_points'] = num_points
        voxel_features = self.pts_voxel_encoder(batch_dict)
        if gt_boxes!=None:
            voxel_features['gt_boxes'] = gt_boxes
        # batch_size = coors[-1, 0] + 1
        # x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone_3d(voxel_features)

        return x
    def extract_img_feat(self, img, img_metas):
        return None
    def extract_feat(self, points, img, img_metas,gt_boxes=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas,gt_boxes=gt_boxes)
        return (img_feats, pts_feats)
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
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
        "fuse gt_bboxes_3d to BXMX9"
        device = points[0].device
        batch_dict = dict()
        max_box_num = max([len(bboxes) for bboxes in gt_bboxes_3d])
        box_dim = gt_bboxes_3d[0].tensor.shape[-1]
        bboxes_3d_tensor = torch.stack([
            torch.cat([b.tensor, torch.zeros(max_box_num - len(b), box_dim).to(b.tensor.device)])
            for b in gt_bboxes_3d
        ]).to(device)
        # # get first 7 dims
        # bboxes_3d_tensor = bboxes_3d_tensor[:, :, :7]
        # attach gt_labels_3d to tensor
        gt_labels_3d_tensor = torch.stack([
            torch.cat([l.float()+1, torch.zeros(max_box_num - len(l)).to(l.device)])
            for l in gt_labels_3d
        ]).to(device)
        bboxes_3d_tensor = torch.cat([bboxes_3d_tensor, gt_labels_3d_tensor.unsqueeze(-1)], dim=-1)

        batch_dict['gt_boxes'] = bboxes_3d_tensor
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas,gt_boxes=bboxes_3d_tensor)
        # batch_dict = pts_feats.update(batch_dict)
        batch_dict.update(pts_feats)
        output_dict = self.dense_head(batch_dict) # img_feats for future use

        losses, loss_dict = self.dense_head.get_loss()
        if 'loss_box_of_pts_sprs' in batch_dict.keys():
            loss_dict['loss_box_of_pts_sprs'] = batch_dict['loss_box_of_pts_sprs']
        return loss_dict
    
    def aug_test(self):
        pass

    def simple_test(self,points=None,img_metas=None,img=None,**kwargs):
        img_feats, pts_feats = self.extract_feat(
        points, img=img, img_metas=img_metas)
        batch_dict = pts_feats
        # "fuse gt_bboxes_3d to BXMX9"
        # max_box_num = max([len(bboxes) for bboxes in gt_bboxes_3d])
        # box_dim = gt_bboxes_3d[0].tensor.shape[-1]
        # bboxes_3d_tensor = torch.stack([
        #     torch.cat([b.tensor, torch.zeros(max_box_num - len(b), box_dim).to(b.tensor.device)])
        #     for b in gt_bboxes_3d
        # ]).to(pts_feats['voxel_features'].device)
        # # # get first 7 dims
        # # bboxes_3d_tensor = bboxes_3d_tensor[:, :, :7]
        # # attach gt_labels_3d to tensor
        # gt_labels_3d_tensor = torch.stack([
        #     torch.cat([l.float()+1, torch.zeros(max_box_num - len(l)).to(l.device)])
        #     for l in gt_labels_3d
        # ]).to(pts_feats['voxel_features'].device)
        # bboxes_3d_tensor = torch.cat([bboxes_3d_tensor, gt_labels_3d_tensor.unsqueeze(-1)], dim=-1)

        # batch_dict['gt_boxes'] = bboxes_3d_tensor
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
        return result_list  
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
    # pred_dicts = [[dict()] for i in range(len(output_dict['pred_dicts']))] #output_dict['pred_dicts']
                #         (preds_dict[0]['reg'], preds_dict[0]['height'],
                #  preds_dict[0]['dim'], preds_dict[0]['rot'],
                #  preds_dict[0]['vel']),
        # for i in range(len(pred_dicts)):
        #     pred_dicts[i][0]['heatmap']=output_dict['pred_dicts'][i]['hm']
        #     pred_dicts[i][0]['reg']=output_dict['pred_dicts'][i]['center']
        #     pred_dicts[i][0]['height']=output_dict['pred_dicts'][i]['center_z']
        #     pred_dicts[i][0]['dim']=output_dict['pred_dicts'][i]['dim']
        #     pred_dicts[i][0]['rot']=output_dict['pred_dicts'][i]['rot']
        #     pred_dicts[i][0]['vel']=output_dict['pred_dicts'][i]['vel']
        # losses = dict()
        # losses = self.dense_head.loss(gt_bboxes_3d, gt_labels_3d, pred_dicts)
    # def forward(self, batch_dict):

    #     for cur_module in self.module_list:
    #         batch_dict = cur_module(batch_dict)

    #     if self.training:
    #         loss, tb_dict, disp_dict = self.get_training_loss()
    #         ret_dict = {
    #             'loss': loss
    #         }
    #         return ret_dict, tb_dict, disp_dict
    #     else:
    #         pred_dicts, recall_dicts = self.post_processing(batch_dict)
    #         return pred_dicts, recall_dicts

    # def get_training_loss(self):
        
    #     disp_dict = {}
    #     loss, tb_dict = self.dense_head.get_loss()
        
    #     return loss, tb_dict, disp_dict

    # def post_processing(self, batch_dict):
    #     post_process_cfg = self.model_cfg.POST_PROCESSING
    #     batch_size = batch_dict['batch_size']
    #     final_pred_dict = batch_dict['final_box_dicts']
    #     recall_dict = {}
    #     for index in range(batch_size):
    #         pred_boxes = final_pred_dict[index]['pred_boxes']

    #         recall_dict = self.generate_recall_record(
    #             box_preds=pred_boxes,
    #             recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
    #             thresh_list=post_process_cfg.RECALL_THRESH_LIST
    #         )

    #     return final_pred_dict, recall_dict
