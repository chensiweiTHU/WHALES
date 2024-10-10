# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead
from mmdet3d.models.dense_heads.anchor_free_mono3d_head import AnchorFreeMono3DHead
from mmdet3d.models.dense_heads.base_conv_bbox_head import BaseConvBboxHead
from mmdet3d.models.dense_heads.base_mono3d_dense_head import BaseMono3DDenseHead
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.models.dense_heads.fcos_mono3d_head import FCOSMono3DHead
from mmdet3d.models.dense_heads.free_anchor3d_head import FreeAnchor3DHead
from mmdet3d.models.dense_heads.groupfree3d_head import GroupFree3DHead
from mmdet3d.models.dense_heads.parta2_rpn_head import PartA2RPNHead
from mmdet3d.models.dense_heads.shape_aware_head import ShapeAwareHead
from mmdet3d.models.dense_heads.ssd_3d_head import SSD3DHead
from mmdet3d.models.dense_heads.vote_head import VoteHead
from .position_aware_pruning import SeparateTaskHead, FSTRHead
from .voxelnext_head import VoxelNeXtHead
from .voxelnext_head_maxpool import VoxelNeXtHeadMaxPool
__all__ = [
    'Anchor3DHead', 'FreeAnchor3DHead', 'PartA2RPNHead', 'VoteHead',
    'SSD3DHead', 'BaseConvBboxHead', 'CenterHead', 'ShapeAwareHead',
    'BaseMono3DDenseHead', 'AnchorFreeMono3DHead', 'FCOSMono3DHead',
    'GroupFree3DHead', 
    'SeparateTaskHead', 'FSTRHead', 
    'VoxelNeXtHead', 'VoxelNeXtHeadMaxPool'
]
