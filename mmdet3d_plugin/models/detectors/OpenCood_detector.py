"""
by Siwei Chen in 20240307
Build OpenCood(by DerrickXuNu) detector models, including preprocessing and post processing, 
see https://github.com/DerrickXuNu/OpenCOOD for more details
"""
import opencood
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.base import Base3DDetector
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
import torch
@DETECTORS.register_module()
class OpenCoodDetector(Base3DDetector):
    def __init__(self, **kwargs):
        super(OpenCoodDetector, self).__init__()
        keys,values = zip(*kwargs.items())
        if keys[0]=="hypes_yaml":
            # assert "train" in keys
            # train = values[keys.index("train")]
            self.__initfromhypes(values[0],train=True)
            # init with train=True set train=False when testing
        else:
            # To do :build directly from mmdet3d config
            raise NotImplementedError("To do :build directly from mmdet3d config")
    def __initfromhypes(self, hypes_yaml, train):
        hypes = yaml_utils.load_yaml(hypes_yaml, None)
        self.opencood_model = train_utils.create_model(hypes) 
        self.pre_processor = build_preprocessor(hypes['preprocess'], train)
        self.post_processor = build_postprocessor(hypes['postprocess'], train)
    def map_mmdet3d2opencood(self, **kwargs):
        data_dict = dict()


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # new code
        batch_size = len(points)
        points_np = [point.cpu().numpy() for point in points]
        points_voxel = [
            self.pre_processor.preprocess(points_np[i])
            for i in range(len(points_np))
        ]
        data_dict = dict()
        "merger the processed lidar to one batch"
        voxel_features = [points_voxel[i]['voxel_features']for i in range(batch_size)]
        voxel_coords = [points_voxel[i]['voxel_coords']for i in range(batch_size)]
        voxel_num_points = [points_voxel[i]['voxel_num_points']for i in range(batch_size)]
        data_dict['processed_lidar'] = {'voxel_features': voxel_features,
                                        'voxel_coords': voxel_coords,
                                        'voxel_num_points': voxel_num_points}
        data_dict['record_len'] = [len(img_metas[i]['cooperative_agents'].keys()) for i in range(batch_size)]
        
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()
        self.opencood_model(data_dict)



        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass
    def simple_test(self, img, img_metas, **kwargs):
        pass