import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss


@LOSSES.register_module()
class WeightedFocalLoss(nn.Module):
    """Sigmoid focal loss with per-class scaling on the C output channels.

    Standard ``FocalLoss`` uses ``alpha`` to balance foreground vs background;
    it cannot rebalance across classes when one class is underrepresented in
    the training set. ``class_weight`` here scales the per-channel BCE before
    reduction so rare classes receive proportionally more gradient.

    Args:
        class_weight (list[float] | None): Per-class multiplier of length C.
            None disables class weighting (equivalent to plain FocalLoss).
        gamma, alpha, reduction, loss_weight: passed to ``py_sigmoid_focal_loss``.
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 class_weight=None,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        assert use_sigmoid, 'WeightedFocalLoss only supports sigmoid mode.'
        self.use_sigmoid = True
        self.gamma = gamma
        self.alpha = alpha
        self.class_weight = list(class_weight) if class_weight is not None else None
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        num_classes = pred.size(1)
        if self.class_weight is not None:
            assert len(self.class_weight) == num_classes, (
                f'class_weight length {len(self.class_weight)} != '
                f'num_classes {num_classes}')

        target_oh = F.one_hot(target, num_classes=num_classes + 1)
        target_oh = target_oh[:, :num_classes].type_as(pred)

        if self.class_weight is not None:
            cw = pred.new_tensor(self.class_weight)        # (C,)
            class_w = cw.unsqueeze(0).expand_as(pred)      # (N, C)
        else:
            class_w = pred.new_ones(pred.shape)

        if weight is not None:
            sample_w = weight.view(-1, 1) if weight.dim() == 1 else weight
            class_w = class_w * sample_w

        return self.loss_weight * py_sigmoid_focal_loss(
            pred,
            target_oh,
            class_w,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)
