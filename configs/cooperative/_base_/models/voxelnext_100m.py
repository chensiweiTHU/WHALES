plugin = True
plugin_dir = "mmdet3d_plugin/"
class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
voxel_size=[0.15, 0.15, 0.2]
num_point_features=4
point_cloud_range=[-108, -108, -5, 108, 108, 3]#[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
grid_size = [1440, 1440, 40]
# grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
# self.grid_size = np.round(grid_size).astype(np.int64)
model = dict(
    type='VoxelNeXt',
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000)),
    pts_voxel_encoder=dict(
            type='MeanVFE',
            num_point_features=num_point_features
            ),
    backbone_3d=dict(
        type='VoxelResBackBone8xVoxelNeXt',
        input_channels = num_point_features,
        grid_size = grid_size,
        spconv_kernel_sizes=[3, 3, 3, 3], 
        channels=[16, 32, 64, 128, 128], 
        out_channel=128,
        ),
    dense_head=dict(
        type='VoxelNeXtHead',
        model_cfg=dict(
        class_agnostic=False,
        input_features=128,
        class_names_each_head=[['Vehicle'],['Pedestrian'],['Cyclist']],#[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
        shared_conv_channel=128,
        kernel_size_head=3,
        use_bias_before_norm=False,
        num_hm_conv=2,
        separate_head_cfg=dict(
            head_order=['center', 'center_z', 'dim', 'rot', 'vel'],
            head_dict={
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
                'vel': {'out_channels': 2, 'num_conv': 2},
            }
        ),
        target_assigner_config=dict(
            feature_map_stride=8,
            num_max_objs=500,
            gaussian_overlap=0.1,
            min_radius=2
        ),
        loss_config=dict(
            loss_weights={'cls_weight': 1.0, 'loc_weight': 0.25, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]}
        ),
        post_processing=dict(
            score_thresh=0.1,
            post_center_limit_range=[-61.2, -61.2, -2.0, 61.2, 61.2, 2.0],
            max_obj_per_sample=500,
            nms_config=dict(
                nms_type='nms_gpu',
                nms_thresh=0.1,
                nms_pre_maxsize=1000,
                nms_post_maxsize=150
            )
        )
    ),
    input_channels = 128,
    num_class=3,
    class_names = class_names,
    grid_size = grid_size,
    point_cloud_range = point_cloud_range,
    voxel_size = voxel_size,
    bbox_coder=dict(
        type='CenterPointBBoxCoder',
        pc_range=point_cloud_range,
        post_center_range=[-122.4, -122.4, -2, 122.4, 122.4, -1],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        code_size=9),
    loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
    loss_bbox=dict(type='L1Loss', reduction='none', loss_weight=0.25),
    train_cfg=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
    test_cfg=dict(
            post_center_limit_range=[-122.4, -122.4, -10.0, 122.4, 122.4, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2),
    ),
    #     POST_PROCESSING:
#         RECALL_THRESH_LIST: [0.3, 0.5, 0.7]

#         EVAL_METRIC: kitti
    post_processing=dict(
        recall_thresh_list=[0.3, 0.5, 0.7],
        eval_metric='kitti'
    )
)
