# Copyright (c) OpenMMLab. All rights reserved.

# from mmdet3d.models.detectors.base import Base3DDetector
# from mmdet3d.models.detectors.centerpoint import CenterPoint
# from mmdet3d.models.detectors.dynamic_voxelnet import DynamicVoxelNet
# from mmdet3d.models.detectors.fcos_mono3d import FCOSMono3D
# from mmdet3d.models.detectors.groupfree3dnet import GroupFree3DNet
# from mmdet3d.models.detectors.h3dnet import H3DNet
# from mmdet3d.models.detectors.imvotenet import ImVoteNet
# from mmdet3d.models.detectors.imvoxelnet import ImVoxelNet
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
from .mvx_two_stage import MVXTwoStageDetector
# from mmdet3d.models.detectors.parta2 import PartA2
# from mmdet3d.models.detectors.single_stage_mono3d import SingleStageMono3DDetector
# from mmdet3d.models.detectors.ssd3dnet import SSD3DNet
# from mmdet3d.models.detectors.votenet import VoteNet

from .OpenCood_detector import OpenCoodDetector
from .FCooper import FCooper
from .V2VNet import V2VNet
from .OpenCood_point_pillar import PointPillarOpenCOOD
from .voxelnext import VoxelNeXt
__all__ = [
    'MVXTwoStageDetector',  'OpenCoodDetector', 'DynamicMVXFasterRCNN', 'MVXFasterRCNN',
    'FCooper', 'V2VNet', 'PointPillarOpenCOOD', 'VoxelNeXt', 
]
# __all__ = [
#     'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
#     'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
#     'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
#     'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet','OpenCoodDetector'
# ]
