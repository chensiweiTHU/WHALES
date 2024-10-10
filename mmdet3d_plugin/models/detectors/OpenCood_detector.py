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
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
import torch
from mmdet3d.models import builder
from torch.nn import functional as F
from mmcv.runner import force_fp32
import numpy as np
from mmdet3d.ops import Voxelization
from .get_bev import get_bev_V2XVIT
from types import MethodType
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
@DETECTORS.register_module()
class OpenCoodDetector(Base3DDetector):
    def __init__(self, **kwargs):
        super(OpenCoodDetector, self).__init__()
        keys,values = zip(*kwargs.items())
        if "pts_voxel_layer" in keys:
            self.pts_voxel_layer = Voxelization(**values[keys.index("pts_voxel_layer")])
        if keys[0]=="hypes_yaml":
            # assert "train" in keys
            # train = values[keys.index("train")]
            self.__initfromhypes(values[0],train=True)
            # init with train=True set train=False when testing
        else:
            # To do :build directly from mmdet3d config
            raise NotImplementedError("To do :build directly from mmdet3d config")
        if "train_cfg" in keys:
            train_cfg = values[keys.index("train_cfg")]
        if "test_cfg" in keys:
            test_cfg = values[keys.index("test_cfg")]
        if "pts_bbox_head" in keys:
            pts_bbox_head = values[keys.index("pts_bbox_head")]
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
    def __initfromhypes(self, hypes_yaml, train):
        hypes = yaml_utils.load_yaml(hypes_yaml, None)
        self.hypes = hypes
        self.opencood_model = train_utils.create_model(hypes) 
        self.pre_processor = build_preprocessor(hypes['preprocess'], train)
        self.post_processor = build_postprocessor(hypes['postprocess'], train)
        self.loss = train_utils.create_loss(hypes)
        self.opencood_model.get_bev = MethodType(get_bev_V2XVIT, self.opencood_model)
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
        # todo: 之前少了协同车的预处理
        batch_size = len(points)
        preprocessed_voxels = []
        cooperative = 'cooperative_agents' in img_metas[0].keys()
        for i in range(batch_size):
            points_np = [points[i].cpu().numpy()]
            if cooperative:
                for key in img_metas[i]['cooperative_agents'].keys():
                    if len(img_metas[i]['cooperative_agents'][key]['points'])>0:
                        points_np.append(img_metas[i]['cooperative_agents'][key]['points'].cpu().numpy())
                    else:
                        # relative_position = np.array(img_metas[i]['cooperative_agents'][key]['ego2global_translation']) \
                        #     - np.array(img_metas[i]['ego_agent']['ego2global_translation'])
                        "create one point for empty point cloud"
                        one_point = np.array([[50, 50, 0, 0.1]])
                        points_np.append(one_point)
            points_voxel = [
                self.pre_processor.preprocess(points_np[p])
                for p in range(len(points_np))
            ]
            points_voxel = IntermediateFusionDataset.merge_features_to_dict(points_voxel)
            preprocessed_voxels.append(points_voxel)
        # fuse twice
        preprocessed_voxels = IntermediateFusionDataset.merge_features_to_dict(preprocessed_voxels)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(preprocessed_voxels)
        
        for key in processed_lidar_torch_dict.keys():
            processed_lidar_torch_dict[key] = processed_lidar_torch_dict[key].to(points[0].device)
        # voxels, num_points, coors = self.voxelize(points)

        data_dict = dict()
        # "merger the processed lidar to one batch"
        # voxel_features = [points_voxel[i]['voxel_features']for i in range(batch_size)]
        # voxel_coords = [points_voxel[i]['voxel_coords']for i in range(batch_size)]
        # voxel_num_points = [points_voxel[i]['voxel_num_points']for i in range(batch_size)]
        # data_dict['processed_lidar'] = {'voxel_features': voxel_features,
        #                                 'voxel_coords': voxel_coords,
        #                                 'voxel_num_points': voxel_num_points}
        data_dict['processed_lidar'] = processed_lidar_torch_dict
        if cooperative:
            data_dict['record_len'] = [len(img_metas[i]['cooperative_agents'].keys())+1 for i in range(batch_size)]
        else:
            data_dict['record_len'] = [1 for i in range(batch_size)]
        # (B, max_cav, 3)
        if cooperative:
            max_cav = max(data_dict['record_len'])
            velocity = torch.zeros(batch_size, max_cav)
            time_delay = torch.zeros(batch_size, max_cav)
            infra = torch.zeros(batch_size, max_cav)
            cls_map = {'vehicle':0, 'rsu':1}
            for i in range(batch_size):
                keys = list(img_metas[i]['cooperative_agents'].keys())
                velocity[i, 0] = np.linalg.norm( img_metas[i]['ego_agent']['ego_velocity'])
                time_delay[i, 0] = 0
                infra[i, 0] = cls_map[img_metas[i]['ego_agent']['veh_or_rsu']]
                for j in range(data_dict['record_len'][i]-1):
                    key = keys[j]
                    velocity[i, j+1] = np.linalg.norm( img_metas[i]['cooperative_agents'][key]['ego_velocity'])
                    time_delay[i, j+1] =img_metas[i]['ego_agent']['timestamp'] - img_metas[i]['cooperative_agents'][key]['timestamp']
                    infra[i, j+1] = cls_map[img_metas[i]['cooperative_agents'][key]['veh_or_rsu']]
            # B, max_cav, 3(dt dv infra), 1, 1
            prior_encoding = \
                torch.stack([velocity, time_delay, infra], dim=-1).float()
            prior_encoding = prior_encoding.to(points[0].device)
            data_dict['prior_encoding'] = prior_encoding
            data_dict['spatial_correction_matrix'] = torch.eye(4,device=points[0].device).unsqueeze(0).repeat(batch_size,max_cav, 1, 1)
        
        data_dict['record_len'] = torch.tensor(data_dict['record_len'],device=points[0].device)
        # data_dict = train_utils.to_device(data_dict, device)
        # out_puts = self.opencood_model(data_dict)
        # # todo process gt_bboxes3d and gt_labels_3d into opencood
        # for i in range(batch_size):
        #     gt_bboxes_3d[i].tensor = gt_bboxes_3d[i].tensor
        #     gt_labels_3d[i] = gt_labels_3d[i].to(points[0].device)
        # # target_dict = {
        # #     "target": gt_bboxes_3d[:,:7], # the open cood model cannot predict speed
        # # }
        # # object_bbx_center = [gt_bboxes_3d[i].tensor[:,:7] for i in range(batch_size)]
        # anchor_box = self.post_processor.generate_anchor_box()
        # mask = np.zeros(self.hypes['postprocess']['max_num'])
        # mask[:len(gt_bboxes_3d)] = 1
        # label_dict_list=[]
        # for i in range(batch_size):
        #     gt_box_center = torch.zeros((self.hypes['postprocess']['max_num'],7))
        #     gt_box_center[:len(gt_bboxes_3d[i])] = gt_bboxes_3d[i].tensor[:,:7]
        #     gt_box_center=gt_box_center.numpy()
        #     label_dict = \
        #         self.post_processor.generate_label(
        #             gt_box_center=gt_box_center, # hwl
        #             anchors=anchor_box,
        #             mask=mask)
        #     label_dict_list.append(label_dict)
        # label_torch_dict = \
        #     self.post_processor.collate_batch(label_dict_list)
        # for key in label_torch_dict.keys():
        #     label_torch_dict[key] = label_torch_dict[key].to(points[0].device)
        # loss = self.loss(out_puts, label_torch_dict)

        # losses = dict()
        # losses['loss'] = loss
        # img_feats, pts_feats = self.extract_feat(
        #     points, img=img, img_metas=img_metas)
        pts_feats = self.opencood_model.get_bev(data_dict)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        return losses
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass
    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        batch_size = len(points)
        preprocessed_voxels = []
        cooperative = 'cooperative_agents' in img_metas[0].keys()
        for i in range(batch_size):
            points_np = [points[i].cpu().numpy()]
            if cooperative:
                for key in img_metas[i]['cooperative_agents'].keys():
                    # print(type(img_metas[i]['cooperative_agents'][key]['points']))
                    points_np.append(img_metas[i]['cooperative_agents'][key]['points'].tensor.cpu().numpy())
                    # points_np.append(img_metas[i]['cooperative_agents'][key]['points'].numpy())
            points_voxel = [
                self.pre_processor.preprocess(points_np[p])
                for p in range(len(points_np))
            ]
            points_voxel = IntermediateFusionDataset.merge_features_to_dict(points_voxel)
            preprocessed_voxels.append(points_voxel)
        # fuse twice
        preprocessed_voxels = IntermediateFusionDataset.merge_features_to_dict(preprocessed_voxels)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(preprocessed_voxels)
        
        for key in processed_lidar_torch_dict.keys():
            processed_lidar_torch_dict[key] = processed_lidar_torch_dict[key].to(points[0].device)
        # voxels, num_points, coors = self.voxelize(points)

        data_dict = dict()
        # "merger the processed lidar to one batch"
        # voxel_features = [points_voxel[i]['voxel_features']for i in range(batch_size)]
        # voxel_coords = [points_voxel[i]['voxel_coords']for i in range(batch_size)]
        # voxel_num_points = [points_voxel[i]['voxel_num_points']for i in range(batch_size)]
        # data_dict['processed_lidar'] = {'voxel_features': voxel_features,
        #                                 'voxel_coords': voxel_coords,
        #                                 'voxel_num_points': voxel_num_points}
        data_dict['processed_lidar'] = processed_lidar_torch_dict
        if cooperative:
            data_dict['record_len'] = [len(img_metas[i]['cooperative_agents'].keys())+1 for i in range(batch_size)]
        else:
            data_dict['record_len'] = [1 for i in range(batch_size)]
        # (B, max_cav, 3)
        if cooperative:
            max_cav = max(data_dict['record_len'])
            velocity = torch.zeros(batch_size, max_cav)
            time_delay = torch.zeros(batch_size, max_cav)
            infra = torch.zeros(batch_size, max_cav)
            cls_map = {'vehicle':0, 'rsu':1}
            for i in range(batch_size):
                keys = list(img_metas[i]['cooperative_agents'].keys())
                velocity[i, 0] = np.linalg.norm( img_metas[i]['ego_agent']['ego_velocity'])
                time_delay[i, 0] = 0
                infra[i, 0] = cls_map[img_metas[i]['ego_agent']['veh_or_rsu']]
                for j in range(data_dict['record_len'][i]-1):
                    key = keys[j]
                    velocity[i, j+1] = np.linalg.norm( img_metas[i]['cooperative_agents'][key]['ego_velocity'])
                    time_delay[i, j+1] =img_metas[i]['ego_agent']['timestamp'] - img_metas[i]['cooperative_agents'][key]['timestamp']
                    infra[i, j+1] = cls_map[img_metas[i]['cooperative_agents'][key]['veh_or_rsu']]
            # B, max_cav, 3(dt dv infra), 1, 1
            prior_encoding = \
                torch.stack([velocity, time_delay, infra], dim=-1).float()
            prior_encoding = prior_encoding.to(points[0].device)
            data_dict['prior_encoding'] = prior_encoding
            data_dict['spatial_correction_matrix'] = torch.eye(4,device=points[0].device).unsqueeze(0).repeat(batch_size,max_cav, 1, 1)
        data_dict['record_len'] = torch.tensor(data_dict['record_len'],device=points[0].device)

        pts_feats = self.opencood_model.get_bev(data_dict)
        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for img_meta in img_metas:
                if 'cooperative_agents' in img_meta.keys():
                    for key in img_meta['cooperative_agents'].keys():
                        if 'points' in img_meta['cooperative_agents'][key].keys():
                            "delete the points, because imgmetas is saved in temp files"
                            del img_meta['cooperative_agents'][key]['points']
                if 'ego_agent' in img_meta.keys():
                    if 'points' in img_meta['ego_agent'].keys():
                        "delete the points, because imgmetas is saved in temp files"
                        del img_meta['ego_agent']['points']
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
                result_dict['img_metas'] = img_metas
        return bbox_list
    
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            img_metas (list[dict]): Meta information of images.
            gt_bboxes (list[torch.Tensor]): Ground truth boxes of each image
                sample.
            gt_labels (list[torch.Tensor]): Ground truth labels of boxes.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            proposals (list[torch.Tensor], optional): Proposals of each sample.
                Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                              self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # bbox head forward and loss
        if self.with_img_bbox:
            # bbox head forward and loss
            img_roi_losses = self.img_roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, **kwargs)
            losses.update(img_roi_losses)

        return losses