# from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x
# from .spconv_backbone_focal import VoxelBackBone8xFocal
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from .spconv_backbone_voxelnext2d import VoxelResBackBone8xVoxelNeXt2D
from .spconv_backbone_voxelnext_sps import VoxelResBackBone8xVoxelNeXtSPS
from .spconv_backbone_voxelnext2d_sps import VoxelResBackBone8xVoxelNeXt2DSPS
from .spconv_unet import UNetV2
__all__ = [
        'VoxelBackBone8x',
        'UNetV2',
        'VoxelResBackBone8x',
        'VoxelResBackBone8xVoxelNeXt',
        'VoxelResBackBone8xVoxelNeXt2D',
        'VoxelResBackBone8xVoxelNeXtSPS',
        'VoxelResBackBone8xVoxelNeXt2DSPS',
        'PillarBackBone8x',
        'PillarRes18BackBone8x'
]
# __all__ = {
#     'VoxelBackBone8x': VoxelBackBone8x,
#     'UNetV2': UNetV2,
#     # 'PointNet2Backbone': PointNet2Backbone,
#     # 'PointNet2MSG': PointNet2MSG,
#     'VoxelResBackBone8x': VoxelResBackBone8x,
#     # 'VoxelBackBone8xFocal': VoxelBackBone8xFocal,
#     'VoxelResBackBone8xVoxelNeXt': VoxelResBackBone8xVoxelNeXt,
#     'VoxelResBackBone8xVoxelNeXt2D': VoxelResBackBone8xVoxelNeXt2D,
#     'VoxelResBackBone8xVoxelNeXtSPS': VoxelResBackBone8xVoxelNeXtSPS,
#     'VoxelResBackBone8xVoxelNeXt2DSPS': VoxelResBackBone8xVoxelNeXt2DSPS,
#     'PillarBackBone8x': PillarBackBone8x,
#     'PillarRes18BackBone8x': PillarRes18BackBone8x
# }
