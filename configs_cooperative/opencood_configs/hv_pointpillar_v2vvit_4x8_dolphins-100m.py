point_cloud_range = [-100, -100, -5, 100, 100, 3]
_base_ = [
    # '../_base_/models/hv_pointpillars_fpn_dolphins.py',
    '../_base_/datasets/dolphins-3d.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
class_names = [
    'Vehicle', 'Pedestrian', 'Cyclist'
]
plugin = True
plugin_dir = "mmdet3d_plugin/"
dataset_type = 'WhalesDataset'
data_root = 'data/whales/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'dolphins_dbinfos_train.pkl',
#     rate=1.0,
#         prepare=dict(

#         filter_by_min_points=dict(
#         Vehicle=5,
#         Pedestrian=5,
#         Cyclist=5,
#         ),),
#     classes=class_names,
#     sample_groups=dict(
#         Vehicle=2,
#         Pedestrian=2,
#         Cyclist=2,
#     ),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=[0, 1, 2, 3],
#         file_client_args=file_client_args))
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='AgentScheduling',
        mode="unicast", 
        submode="closest", 
        basic_data_limit=6e6
        ),
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='ProjectCooperativePCD2ego'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilterCP', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3DCP', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],\
        meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
                'depth2img', 'cam2img', 'pad_shape',
                'scale_factor', 'flip', 'pcd_horizontal_flip',
                'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'pcd_trans', 'sample_idx',
                'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                'transformation_3d_flow',
                # new keys
                # 'transmitted_data_size'
                'cooperative_agents',
                'ego_agent'
                ])
    # dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='AgentScheduling',
        mode="unicast", 
        submode="closest", 
        basic_data_limit=6e6
        ),
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    # dict(type='LoadAnnotations3D'),
    dict(type='ProjectCooperativePCD2ego'),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'], meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
                'depth2img', 'cam2img', 'pad_shape',
                'scale_factor', 'flip', 'pcd_horizontal_flip',
                'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'pcd_trans', 'sample_idx',
                'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                'transformation_3d_flow',
                # new keys
                'transmitted_data_size',
                'cooperative_agents',
                'ego_agent'
                ])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline
# eval_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         file_client_args=file_client_args),
#     dict(type='AgentScheduling',
#         mode="unicast", 
#         submode="closest", 
#         basic_data_limit=6e6
#         ),
#     dict(
#         type='LoadPointsFromCooperativeAgents',
#         coord_type='LIDAR',
#         load_dim=4, use_dim=4,
#         file_client_args=file_client_args
#         ),
#     dict(type='LoadAnnotations3D'),
#     dict(type='ProjectCooperativePCD2ego'),
#     # dict(
#     #     type='LoadPointsFromMultiSweeps',
#     #     sweeps_num=10,
#     #     file_client_args=file_client_args),
#     dict(
#         type='DefaultFormatBundle3D',
#         class_names=class_names,
#         with_label=False),
#     dict(type='Collect3D', keys=['points'], meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
#                     'depth2img', 'cam2img', 'pad_shape',
#                     'scale_factor', 'flip', 'pcd_horizontal_flip',
#                     'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
#                     'img_norm_cfg', 'pcd_trans', 'sample_idx',
#                     'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
#                     'transformation_3d_flow',
#                     # new keys
#                     'transmitted_data_size',
#                     'cooperative_agents',
#                     'ego_agent'
#                     ])
# ]
# model settings
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'whales_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='OpenCoodDetector',
    hypes_yaml='configs_CooperativePerception/opencood_configs/point_pillar_v2xvit-100m.yaml',
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-100, -100, -5, 100, 100, 3],
        voxel_size=[0.4, 0.4, 4],
        max_voxels=(30000, 40000),
        ),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-49.6, -49.6, -1.8, 49.6, 49.6, -1.4],
                    [-49.6, -49.6, -1.7, 49.6, 49.6, -1.3],
                    [-49.6, -49.6, -2, 49.6, 49.6, -1.6]],
            sizes=[[5, 2.2, 1.8], [0.6, 0.6, 1.8], [2.1, 0.8, 1.6]],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500))
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
runner = dict(type='EpochBasedRunner', max_epochs=24,)
evaluation = dict(interval=24, pipeline=eval_pipeline)
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.01)