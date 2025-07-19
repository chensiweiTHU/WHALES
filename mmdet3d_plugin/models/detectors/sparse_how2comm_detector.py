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
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from mmdet3d_plugin.mmcv_custom import SPConvVoxelization

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
class VoxelNeXtHow2commPruning(MVXTwoStageDetector):

    def __init__(self,
                 **kwargs):
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super(VoxelNeXtHow2commPruning, self).__init__(**kwargs)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)
        self.single = kwargs.get('single', False)
        self.proj_first = kwargs.get('proj_first', False)

    def init_weights(self):
        """Initialize model weights."""
        super(VoxelNeXtHow2commPruning, self).init_weights()

    def extract_feat(self, points, img_metas, gt_boxes=None):
        """Extract features from images and points."""
        pts_feats = self.extract_pts_feat(points, img_metas,gt_boxes=gt_boxes)
        return pts_feats

    @force_fp32(apply_to=('pts'))    
    def extract_pts_feat(self, pts, img_metas, gt_boxes=None):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        # print('voxel_features',voxel_features.shape)
        backbone_input = dict()
        backbone_input.update({
            'voxel_features': voxel_features,
            'batch_size':batch_size,
            'voxel_coords': coors,
            'gt_boxes': gt_boxes})
        
        
        x = self.pts_backbone(backbone_input)
        return x

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

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      infrastructure_points=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
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
        max_box_num = max([len(bboxes) for bboxes in gt_bboxes_3d])
        box_dim = gt_bboxes_3d[0].tensor.shape[-1]
        bboxes_3d_tensor = torch.stack([
            torch.cat([b.tensor, torch.zeros(max_box_num - len(b), box_dim).to(b.tensor.device)])
            for b in gt_bboxes_3d
        ]).to(device)
        # get first 7 dims
        "first filter out -1 labels in DAIR-V2X dataset by FFNet"
        if self.pts_bbox_head.num_classes[0]>1:
            for i,l in enumerate(gt_labels_3d):
                gt_labels_3d[i][l == -1] = self.pts_bbox_head.num_classes[0]-1
        else:
            gt_bboxes_3d = [bboxes_3d_tensor[i][l != -1] for i,l in enumerate(gt_labels_3d)]
            gt_labels_3d = [l[l != -1] for l in gt_labels_3d]
        gt_labels_3d_tensor = torch.stack([
            torch.cat([l.float()+1, torch.zeros(max_box_num - len(l)).to(l.device)])
            for l in gt_labels_3d
        ]).to(device)
        bboxes_3d_tensor = torch.cat([bboxes_3d_tensor, gt_labels_3d_tensor.unsqueeze(-1)], dim=-1)
        batch_dict['gt_boxes'] = bboxes_3d_tensor
        batch_dict['gt_bboxes_3d'] = gt_bboxes_3d
        if not self.single and self.proj_first:
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

        voxel_feats = self.extract_feat(
            points=points, img_metas=img_metas,gt_boxes=bboxes_3d_tensor)
        voxel_feats.update(gt_bboxes_3d=bboxes_3d_tensor)
        pred_single = self.pts_bbox_head(voxel_feats, img_metas)

        if not self.single:
            voxel_feats_inf = self.extract_feat(     
                points=infrastructure_points, img_metas=img_metas,gt_boxes=bboxes_3d_tensor)
            voxel_feats_inf.update(gt_bboxes_3d=bboxes_3d_tensor)
            pred_single_inf = self.pts_bbox_head(voxel_feats_inf, img_metas)
            fused_feats = self.pts_fusion_layer(voxel_feats,voxel_feats_inf,pred_single,pred_single_inf)
        else:
            fused_feats = voxel_feats    
        losses = dict()
        pred_fused = self.pts_bbox_head(fused_feats, img_metas)
        losses_pts = self.pts_bbox_head.loss(
            gt_bboxes_3d, gt_labels_3d, pred_fused)
        losses.update(losses_pts)
        # nvtx.range_pop()
        # nvtx.range_pop()
        return losses

    @force_fp32(apply_to=('pts_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          ):
        """Forward function for point cloud branch.

        Args:
            pts_feats (dict): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None, 
                     rescale = False,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        batch_size = len(points)

        if num_augs == 1:
            return self.simple_test(points[0], img_metas[0], rescale=rescale)
        else:
            return self.aug_test(points, img_metas, **kwargs)
    
    @force_fp32(apply_to=('x'))
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results

    def simple_test(self, points, img_metas, rescale=False):
        """Test function without augmentaiton."""

        pts_feats = self.extract_feat(
            points, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        if self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
                result_dict['img_metas'] = img_metas
                result_dict['boxes_3d'] = pts_bbox['boxes_3d']
                result_dict['scores_3d'] = pts_bbox['scores_3d']
                result_dict['labels_3d'] = pts_bbox['labels_3d']
        return bbox_list

