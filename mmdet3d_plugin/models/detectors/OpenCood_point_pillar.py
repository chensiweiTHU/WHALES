from .OpenCood_detector import OpenCoodDetector
from mmdet.models import DETECTORS
from types import MethodType
from opencood.models.sub_modules.downsample_conv import DownsampleConv
def get_bev(self, data_dict):

    voxel_features = data_dict['processed_lidar']['voxel_features']
    voxel_coords = data_dict['processed_lidar']['voxel_coords']
    voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

    batch_dict = {'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points}

    batch_dict = self.pillar_vfe(batch_dict)
    batch_dict = self.scatter(batch_dict)
    batch_dict = self.backbone(batch_dict)
    # 256x112x112
    spatial_features_2d = batch_dict['spatial_features_2d']
    if self.shrink_flag:
        spatial_features_2d = self.shrink_conv(spatial_features_2d)
    return [spatial_features_2d]

@DETECTORS.register_module(force=True)
class PointPillarOpenCOOD(OpenCoodDetector):
    def __init__(self, **kwargs):
        super(PointPillarOpenCOOD, self).__init__(**kwargs)
        self.opencood_model.get_bev = MethodType(get_bev, self.opencood_model)
        if 'shrink_header' in self.hypes['model']['args'].keys():
            self.opencood_model.shrink_conv = DownsampleConv(self.hypes['model']['args']['shrink_header'])
            self.opencood_model.shrink_flag = True
        else:
            self.opencood_model.shrink_flag = False