import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST


@MATCH_COST.register_module()
class BBox3DL1CostBEVFormer(object):
    """L1 distance cost between predicted and GT 3D boxes.

    Renamed from upstream ``BBox3DL1Cost`` to avoid collisions with other
    plugins. Behaviour is identical.
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
