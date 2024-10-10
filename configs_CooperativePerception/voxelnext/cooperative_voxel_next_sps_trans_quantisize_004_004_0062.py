dataset_type = 'V2XDataset'
plugin = True
plugin_dir = "mmdet3d_plugin/"
data_root = './data/DAIR-V2X-C/cooperative-vehicle-infrastructure/'
data_info_train_path = './data_process/dairv2x/flow_data_jsons/flow_data_info_train.json'
data_info_val_path = './data_process/dairv2x/flow_data_jsons/flow_data_info_val_0.json'
# work_dir = './ffnet_work_dir/work_dir_baseline'
find_unused_parameters=True
class_names = ['Car', 'truck', 'construction_vehicle','bus', 'trailer', 'barrier','motorcycle', 'bicycle','pedestrian', 'traffic_cone']
point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
voxel_size = [0.04, 0.04, 0.0625]
l = int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0])
h = int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1])
output_shape = [h, l]
z_center_pedestrian = -0.6
z_center_cyclist = -0.6
z_center_car = -2.66
out_size_factor = 8
# voxel_size=[0.2, 0.2, 0.5]
num_point_features=4
# point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
grid_size = [2304, 2304, 64]

model = dict(
    type='VoxelNeXtCoopertivePruningConfidence',
    pts_voxel_layer=dict(
        max_num_points=1,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(480000, 640000)),
    pts_voxel_encoder=dict(
            type='MeanVFE',
            num_point_features=num_point_features
            ),
    pruning=dict(
        type='VoxelResSPSQuantiseizer',
        input_channels = num_point_features,
        grid_size = grid_size,
        spconv_kernel_sizes=[3,3], 
        channels=[16,32], 
        point_cloud_range=[-3, -46.08, 0, 1, 46.08, 92.16],
        downsample_pruning_ratio = [0.8,],
    ),
    backbone_3d=dict(
        type='VoxelResBackBone8xVoxelNeXtSPS',
        input_channels = num_point_features,
        grid_size = grid_size,
        spconv_kernel_sizes=[3, 3, 3, 3], 
        channels=[32, 64, 128, 256, 256], 
        out_channel=256,
        point_cloud_range=[-3, -46.08, 0, 1, 46.08, 92.16]
        ),
    fusion_channels=[512,384,256],

    dense_head=dict(
        type='FSTRHead',
        in_channels=256,
        hidden_dim=256,
        downsample_scale=8,
        num_query=500,
        num_init_query=200,
        init_dn_query = False,
        init_learnable_query = False,
        init_query_topk = 1,
        init_query_radius = 1,
        gauusian_dn_sampling=False,
        noise_mean = 0.5,
        noise_std = 0.125,
        max_sparse_token_per_sample = 10000,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=10, class_names=[
                'car', 'truck', 'construction_vehicle',
                'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone'
            ]),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-15.12, -61.2, -4, 107.28, 61.2, 2],
            pc_range=point_cloud_range,
            max_num=500,
            voxel_size=voxel_size,
            num_classes=10), 
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=3),
        transformer=dict(
            type='FSTRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),

                    feedforward_channels=1024, #unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
        train_cfg=dict(
        pts=dict(
            dataset='dairv2x-c',
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=grid_size,  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='dairv2x-c',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
            nms_thr=0.1,
            use_rotate_nms=True,
            max_num=300
        )),
    quant_levels=[[0.04, 0.04, 0.0625],[0.64, 0.64, 1]],
    # pruning = dict(
    #     type = 'VoxelResSPSQuantiseizer',
    #     channels=[32],
        
    #     input_channels = num_point_features,
    #     grid_size = grid_size,
    #     spconv_kernel_sizes=[3], 
    #     ),
    post_processing=dict(
        recall_thresh_list=[0.3, 0.5, 0.7],
        eval_metric='kitti'
    ),
    proj_first=False,
    single=False
)
file_client_args = dict(backend='disk')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_info_train_path,
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    sensor_view='vehicle'),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    sensor_view='infrastructure'),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(type='ProjectCooperativePCD2ego'),
                # dict(
                #     type='ObjectSample',
                #     db_sampler=dict(
                #         data_root=data_root,
                #         info_path=data_root + 'kitti_dbinfos_train.pkl',
                #         rate=1.0,
                #         prepare=dict(
                #             filter_by_difficulty=[-1],
                #             filter_by_min_points=dict(
                #                 Car=5)),
                #         classes=class_names,
                #         sample_groups=dict(Car=15))),
                # dict(
                #     type='ObjectNoise',
                #     num_try=100,
                #     translation_std=[0.25, 0.25, 0.25],
                #     global_rot_range=[0.0, 0.0],
                #     rot_range=[-0.15707963267, 0.15707963267]),
                # dict(type='GlobalTransCP',trans_factor=[-46.08,0,0]),
                dict(
                    type='GlobalRotScaleTransCP',
                    #rot_range=[-0.78539816, 0.78539816], #go to nan when inf no point
                    # rot_range=[-0.38269908, 0.38269908],
                    # scale_ratio_range=[0.95, 1.05]),
                    rot_range=[-0, 0],
                    scale_ratio_range=[1, 1]),
                dict(type='RandomFlip3DCP', flip_ratio_bev_horizontal=0.5),
                # dict(type='GlobalTransCP',trans_factor=[+46.08,0,0]),
                dict(
                    type='PointsRangeFilterCP',
                    point_cloud_range=point_cloud_range),
                dict(type='PointShuffleCP'),
                # dict(
                #     type='PointQuantization',
                #     voxel_size = voxel_size,
                #     quantize_coords_range = point_cloud_range,
                #     ),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=point_cloud_range),
                # dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=class_names),
                dict(
                    type='Collect3D',
                    keys=['points', 'infrastructure_points', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                               'depth2img', 'cam2img', 'pad_shape',
                               'scale_factor', 'flip', 'pcd_horizontal_flip',
                               'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                               'img_norm_cfg', 'pcd_trans', 'sample_idx',
                               'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'inf2veh','rot_degree',
                               'gt_bboxes_3d', 'gt_labels_3d')
                )
            ],
            modality=dict(use_lidar=True, use_camera=False),
            classes=class_names,
            test_mode=False,
            pcd_limit_range=point_cloud_range,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_info_val_path,
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sensor_view='vehicle'),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sensor_view='infrastructure'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(h, l),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    # dict(
                    #     type='GlobalRotScaleTrans',
                    #     rot_range=[0, 0],
                    #     scale_ratio_range=[1.0, 1.0],
                    #     translation_std=[0, 0, 0]),
                    # dict(type='RandomFlip3D'),
                    # dict(
                    #     type='PointsRangeFilter',
                    #     point_cloud_range=point_cloud_range),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=class_names,
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'infrastructure_points'],
                    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                               'depth2img', 'cam2img', 'pad_shape',
                               'scale_factor', 'flip', 'pcd_horizontal_flip',
                               'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                               'img_norm_cfg', 'pcd_trans', 'sample_idx',
                               'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'inf2veh','rot_degree',
                               'gt_bboxes_3d', 'gt_labels_3d'))
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=class_names,
        test_mode=True,
        pcd_limit_range=point_cloud_range,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_info_val_path,
        split='testing',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sensor_view='vehicle'),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                sensor_view='infrastructure'),
            # dict(type='ProjectCooperativePCD2ego'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(h, l),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    # dict(
                    #     type='GlobalRotScaleTrans',
                    #     rot_range=[0, 0],
                    #     scale_ratio_range=[1.0, 1.0],
                    #     translation_std=[0, 0, 0]),
                    # dict(type='RandomFlip3D'),
                    # dict(
                    #     type='PointsRangeFilter',
                    #     point_cloud_range=point_cloud_range),
                    # dict(
                    # type='PointQuantization',
                    # voxel_size = voxel_size,
                    # quantize_coords_range = point_cloud_range,
                    # ),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=class_names,
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'infrastructure_points'],
                    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                               'depth2img', 'cam2img', 'pad_shape',
                               'scale_factor', 'flip', 'pcd_horizontal_flip',
                               'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                               'img_norm_cfg', 'pcd_trans', 'sample_idx',
                               'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'inf2veh'))
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=class_names,
        test_mode=True,
        pcd_limit_range=point_cloud_range,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=100,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=class_names,
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
lr = 0.001
optimizer = dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'DEBUG'
load_from = None
resume_from = None
workflow = [('train', 1)]
