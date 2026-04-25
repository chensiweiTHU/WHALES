import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

from ..util import denormalize_bbox


@BBOX_CODERS.register_module()
class NMSFreeCoderBEVFormer(BaseBBoxCoder):
    """NMS-free decoder used by BEVFormer's DETR-like head.

    Renamed from upstream ``NMSFreeCoder`` to avoid colliding with other
    plugins.

    Args:
        pc_range (list[float]): Point cloud range used to encode boxes.
        post_center_range (list[float]): Limit on the predicted center
            coordinates after decoding.
        max_num (int): Maximum number of predictions kept per image.
        score_threshold (float): Optional confidence threshold for filtering.
        num_classes (int): Number of foreground classes.
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode predictions for a single image."""
        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)
            if self.score_threshold:
                mask &= thresh_mask
            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            return dict(bboxes=boxes3d, scores=scores, labels=labels)
        raise NotImplementedError(
            'Need to reorganize output as a batch, only '
            'support post_center_range is not None for now!')

    def decode(self, preds_dicts):
        """Decode a batch of predictions taken from the last decoder layer."""
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        batch_size = all_cls_scores.size()[0]
        return [
            self.decode_single(all_cls_scores[i], all_bbox_preds[i])
            for i in range(batch_size)
        ]
