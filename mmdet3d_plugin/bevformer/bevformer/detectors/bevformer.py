import copy

import torch
from mmcv.runner import auto_fp16

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from ...models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """Single-frame BEVFormer detector for the WHALES port.

    Drops the temporal queue / can-bus features that upstream BEVFormer
    relies on for nuScenes; everything else (image backbone, BEV encoder,
    DETR-style decoder) is identical.

    Args:
        use_grid_mask (bool): Whether to apply GridMask augmentation.
        video_test_mode (bool): Whether to use temporal info during inference.
        can_bus_in_dataset (bool): Whether the dataset provides ``can_bus``
            in ``img_metas``.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 can_bus_in_dataset=False):

        super(BEVFormer, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.can_bus_in_dataset = can_bus_in_dataset
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract multi-level image features for a (B, N, C, H, W) input."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(dim=0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue,
                                  int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Image-only feature extraction (BEVFormer is camera-only)."""
        return self.extract_img_feat(img, img_metas, len_queue=len_queue)

    def forward_pts_train(self,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Compute losses from image features and GT 3D boxes."""
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      **kwargs):
        """Single-frame BEVFormer training step (no temporal queue)."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d,
            img_metas, gt_bboxes_ignore, prev_bev=None)
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        """Test entry. Mirrors mmdet3d's nested-list contract."""
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # Single-frame mode: never carry BEV across calls.
        scene_key = 'scene_token'
        if scene_key in img_metas[0][0] and \
                img_metas[0][0][scene_key] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        if scene_key in img_metas[0][0]:
            self.prev_frame_info['scene_token'] = img_metas[0][0][scene_key]

        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        if self.can_bus_in_dataset and 'can_bus' in img_metas[0][0]:
            tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
            if self.prev_frame_info['prev_bev'] is not None:
                img_metas[0][0]['can_bus'][:3] -= \
                    self.prev_frame_info['prev_pos']
                img_metas[0][0]['can_bus'][-1] -= \
                    self.prev_frame_info['prev_angle']
            else:
                img_metas[0][0]['can_bus'][-1] = 0
                img_metas[0][0]['can_bus'][:3] = 0
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0],
            prev_bev=self.prev_frame_info['prev_bev'], **kwargs)

        if self.can_bus_in_dataset and 'can_bus' in img_metas[0][0]:
            self.prev_frame_info['prev_pos'] = tmp_pos
            self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test pts head (returns BEV + per-image bbox results)."""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False,
                    **kwargs):
        """End-to-end test forward."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
