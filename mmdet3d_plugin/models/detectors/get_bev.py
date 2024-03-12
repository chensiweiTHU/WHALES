# -*- coding: utf-8 -*-
# Original Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# by Siwei Chen in 2024.3
# from opencood.models.point_pillar_v2vnet import PointPillarV2VNet
# from mmdet.models import DETECTORS
# @DETECTORS.register_module(force=True)
# class PointPillarV2VNetMMDet(PointPillarV2VNet):
from opencood.models.fuse_modules.fuse_utils import regroup
import torch
def get_bev(self, data_dict):
    voxel_features = data_dict['processed_lidar']['voxel_features']
    voxel_coords = data_dict['processed_lidar']['voxel_coords']
    voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
    record_len = data_dict['record_len']
    spatial_correction_matrix = data_dict['spatial_correction_matrix']

    # B, max_cav, 3(dt dv infra), 1, 1
    prior_encoding =\
        data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

    batch_dict = {'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points,
                    'record_len': record_len}
    # n, 4 -> n, c
    batch_dict = self.pillar_vfe(batch_dict)
    # n, c -> N, C, H, W
    batch_dict = self.scatter(batch_dict)
    batch_dict = self.backbone(batch_dict)

    spatial_features_2d = batch_dict['spatial_features_2d']
    # downsample feature to reduce memory
    if self.shrink_flag:
        spatial_features_2d = self.shrink_conv(spatial_features_2d)
    # compressor
    if self.compression:
        spatial_features_2d = self.naive_compressor(spatial_features_2d)
    # N, C, H, W -> B,  L, C, H, W
    regroup_feature, mask = regroup(spatial_features_2d,
                                    record_len,
                                    self.max_cav)
    # prior encoding added
    prior_encoding = prior_encoding.repeat(1, 1, 1,
                                            regroup_feature.shape[3],
                                            regroup_feature.shape[4])
    regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

    # b l c h w -> b l h w c
    regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
    # transformer fusion
    fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
    # b h w c -> b c h w
    fused_feature = fused_feature.permute(0, 3, 1, 2)
    return [fused_feature]