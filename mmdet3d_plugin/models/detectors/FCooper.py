# from .OpenCood_detector import OpenCoodDetector
from torch import nn
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
from .get_bev import get_bev_FCooper
from mmdet.models import DETECTORS
from types import MethodType
import torch
@DETECTORS.register_module(force=True)
class FCooper(MVXFasterRCNN):
    def __init__(self, **kwargs):
        super(FCooper, self).__init__(**kwargs)
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
        device = points[0].device
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if 'cooperative_agents' in img_metas[0].keys():
            cooperative_points = []
            for i in range(len(img_metas)): # we only support 2 agents
                agent = list(img_metas[i]['cooperative_agents'].keys())[0]
                cooperative_point = img_metas[i]['cooperative_agents'][agent]['points']
                if cooperative_point.device != device:
                    cooperative_point = cooperative_point.to(device)
                if len(cooperative_point) == 0:
                    cooperative_point = torch.tensor([[0,0,-1.5,1.3e-3]]).to(device)
                cooperative_points.append(cooperative_point)
        cooperative_img_feats, cooperative_pts_feats = self.extract_feat(
            cooperative_points, img=img, img_metas=img_metas)
        pts_feats[0] = torch.unsqueeze(pts_feats[0], -1)
        cooperative_pts_feats[0] = torch.unsqueeze(cooperative_pts_feats[0], -1)
        fused_pts_feats = torch.cat([pts_feats[0], cooperative_pts_feats[0]], dim=-1)
        fused_pts_feats, _ = torch.max(fused_pts_feats, dim=-1)
        fused_pts_feats = [fused_pts_feats]
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(fused_pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        device = points[0].device
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if 'cooperative_agents' in img_metas[0].keys():
            cooperative_points = []
            for i in range(len(img_metas)): # we only support 2 agents
                agent = list(img_metas[i]['cooperative_agents'].keys())[0]
                cooperative_point = img_metas[i]['cooperative_agents'][agent]['points']
                if cooperative_point.device != device:
                    cooperative_point = cooperative_point.to(device)
                if len(cooperative_point) == 0:
                    cooperative_point = torch.tensor([[0,0,-1.5,1.3e-3]]).to(device)
                cooperative_points.append(cooperative_point)
        cooperative_img_feats, cooperative_pts_feats = self.extract_feat(
            cooperative_points, img=img, img_metas=img_metas)
        pts_feats[0] = torch.unsqueeze(pts_feats[0], -1)
        cooperative_pts_feats[0] = torch.unsqueeze(cooperative_pts_feats[0], -1)
        fused_pts_feats = torch.cat([pts_feats[0], cooperative_pts_feats[0]], dim=-1)
        fused_pts_feats, _ = torch.max(fused_pts_feats, dim=-1)
        fused_pts_feats = [fused_pts_feats]
        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                fused_pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
                result_dict['img_metas'] = img_metas
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
                result_dict['img_metas'] = img_metas
        return bbox_list