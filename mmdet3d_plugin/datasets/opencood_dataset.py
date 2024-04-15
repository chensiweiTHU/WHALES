import opencood
from mmdet.datasets import DATASETS
import opencood.hypes_yaml.yaml_utils as yaml_utils
from mmdet3d.datasets.custom_3d import Custom3DDataset
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from .pipelines import Compose
from mmdet3d.core.bbox import get_box_type
import torch
@DATASETS.register_module(force=True)
class OpencoodDataset(Custom3DDataset):
    def __init__(self, hypes_yaml,
                 pipeline=None,
                 classes=['Car'],
                 modality = dict(
                 use_lidar=True,
                 use_camera=False,
                 use_radar=False,
                 use_map=False,
                 use_external=False),
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False):
        "we donnot need super().__init__"
        self.hypes = yaml_utils.load_yaml(hypes_yaml)
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.classes = classes
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        print('Dataset Building')
        self.opencood_train_dataset = build_dataset(self.hypes, visualize=True, train=True)
        self.opencood_val_dataset = build_dataset(self.hypes, visualize=True, train=False)
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
    
    def get_opencood_dataset(self, train=True):
        return self.opencood_train_dataset if train else self.opencood_val_dataset
    
    def get_opeencood_data(self, index, train=True):
        return self.get_opencood_dataset(train)[index]

    def prepare_train_data(self, index):
        input_dict = self.get_opeencood_data(index, train=True)
        # print(input_dict)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example
    def prepare_test_data(self, index):
        input_dict = self.get_opeencood_data(index, train=False)
        # print(input_dict)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    def __len__(self):
        if self.test_mode:
            return len(self.opencood_val_dataset)
        else:
            return len(self.opencood_train_dataset)
    
    def evaluate(self, results,**kwargs):
        for i,result in enumerate(results):
            pred_box_tensor = result['pts_bbox']['boxes_3d']
            pred_score = result['pts_bbox']['scores_3d']
            pred_label = result['pts_bbox']['labels_3d']
            # we only have cars in OPV2V
            pred_box_tensor = pred_box_tensor[pred_label==0]
            pred_score = pred_score[pred_label==0]
            data_dict = result['data_dict']
            if data_dict['ego']['transformation_matrix'].dim() == 3:
                data_dict['ego']['transformation_matrix'] = \
                    data_dict['ego']['transformation_matrix'].squeeze(0)
            gt_box_tensor = self.get_opencood_dataset(train=False).\
                post_processor.generate_gt_bbx(data_dict)
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                0.5: {'tp': [], 'fp': [], 'gt': 0},
                0.7: {'tp': [], 'fp': [], 'gt': 0}}
            scale_bias = 2
            bottom_height = -2.0
            dims = pred_box_tensor.dims
            bottom_centers = pred_box_tensor.center
            yaws = pred_box_tensor.yaw.unsqueeze(1)
            bottom_centers[:,2] = bottom_height
            dims = dims * scale_bias
            corrected_box_tensor = torch.cat([bottom_centers, dims, yaws], dim=1)
            corrected_box = type(pred_box_tensor)(corrected_box_tensor)
            eval_utils.caluclate_tp_fp(corrected_box.corners,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3)
            eval_utils.caluclate_tp_fp(corrected_box.corners,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5)
            eval_utils.caluclate_tp_fp(corrected_box.corners,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)
        eval_utils.eval_final_results(result_stat,
                                  self.hypes['model_dir'])
        return result_stat