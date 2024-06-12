from mmdet.datasets.builder import PIPELINES
import opencood.hypes_yaml.yaml_utils as yaml_utils
import numpy as np
from mmdet3d.core.points import  get_points_type
@PIPELINES.register_module(force=True)
class transferOpencoodData(object):
    def __init__(self, hypes_yaml,coord_type='LIDAR'):
        self.hypes = yaml_utils.load_yaml(hypes_yaml)
        self.coord_type = coord_type
    def __call__(self, results):
        object_bbx_center = results['ego']['object_bbx_center'] # Nx7
        object_bbx_mask = results['ego']['object_bbx_mask'] # Nx7
        "we only have one class in the dataset, so we can set all labels to 1.0"
        gt_boxes_3d = object_bbx_center[object_bbx_mask==1, 0:7] # Nx7
        gt_labels_3d = np.ones((object_bbx_center.shape[0], 1)) # Nx1
        results ['gt_boxes_3d'] = gt_boxes_3d
        results ['gt_labels_3d'] = gt_labels_3d
        points_class = get_points_type(self.coord_type)
        points = points_class(
            results['ego']['origin_lidar'], \
            points_dim=results['ego']['origin_lidar'].shape[-1], attribute_dims=None)
        results['points'] = points
        if results['ego']['transformation_matrix'].dtype != np.float32:
            results['ego']['transformation_matrix'] = \
                results['ego']['transformation_matrix'].astype(np.float32)
        return results
    
