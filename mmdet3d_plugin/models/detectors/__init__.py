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
from .voxelnet import VoxelNet
from .OpenCood_detector import OpenCoodDetector
from .FCooper import FCooper
# from .V2VNet_opencood import V2VNet
from .V2VNet import V2VNet
from .OpenCood_point_pillar import PointPillarOpenCOOD
from .voxelnext import VoxelNeXt
from .voxelnext_Cooperative import VoxelNeXtCoopertive
from .voxelnext_Cooperative_pruning import VoxelNeXtCoopertivePruning
from .voxelnext_Cooperative_Temporal import VoxelNeXtCoopertiveTemporal
from .voxelnext_Cooperative_pruning_confidence import VoxelNeXtCoopertivePruningConfidence
from .v2x_voxelnet import V2XVoxelNet
from .feature_flownet import FeatureFlowNet
from .bevf_faster_rcnn import BEVF_FasterRCNN
__all__ = [
    'MVXTwoStageDetector', 'VoxelNet', 'OpenCoodDetector', 'DynamicMVXFasterRCNN', 'MVXFasterRCNN',
    'FCooper_opencood', 'FCooper', 'V2VNet', 'PointPillarOpenCOOD', 'VoxelNeXt', 'V2XVoxelNet', 'FeatureFlowNet', 
    'VoxelNeXtCoopertive', 'VoxelNeXtCoopertiveTemporal', 'VoxelNeXtCoopertivePruning', 'VoxelNeXtCoopertivePruningConfidence',
    'BEVF_FasterRCNN'
]
# __all__ = [
#     'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
#     'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
#     'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
#     'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet','OpenCoodDetector'
# ]
