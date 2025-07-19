_base_ = [
    '../_base_/models/hv_pointpillars_fpn_dolphins.py',
    '../_base_/datasets/dolphins_cam_pp.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "mmdet3d_plugin/"
# model settings
class_names = [
    'Vehicle', 'Pedestrian', 'Cyclist'
]
dataset_type = 'DolphinsDataset'
data_root = 'data/whales/'
# Input modality for Dolphins2 dataset, this is consistent with the submission
# format which requires the information in input_modality.

point_cloud_range = [-50, -50, -5, 50, 50, 3]
file_client_args = dict(backend='disk')
voxel_size = [0.25, 0.25, 8]
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[400, 400]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
        pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-49.6, -49.6, -1.8, 49.6, 49.6, -1.4],
                # [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                # [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                
                [-49.6, -49.6, -1.7, 49.6, 49.6, -1.3],
                [-49.6, -49.6, -2, 49.6, 49.6, -1.6],

                # [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                # [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965],
            ],
            sizes=[
                [5.0, 2.2, 1.8],  # car
                # [2.4560939, 6.73778078, 2.73004906],  # truck
                # [2.87427237, 12.01320693, 3.81509561],  # trailer
                
                [0.60, 0.60, 1.80],  # pedestrian
                [2.1, 0.80, 1.60],  # bicycle
                # [0.39694519, 0.40359262, 1.06232151],  # traffic_cone
                # [2.49008838, 0.48578221, 0.98297065],  # barrier
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
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
            nms_thr=0.1,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))
# model settings

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    # dict(type='AgentScheduling',
    #     mode="unicast", 
    #     submode="closest", 
    #     basic_data_limit=2e6
    #     ),
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    # dict(type='LoadAnnotations3D'),

    dict(type='RawlevelPointCloudFusion'),
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
                            'transmitted_data_size'
                            ])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
        #     dict(type='AgentScheduling',
        # mode="unicast", 
        # submode="closest", 
        # basic_data_limit=2e6
        # ),
    dict(
        type='LoadPointsFromCooperativeAgents',
        coord_type='LIDAR',
        load_dim=4, use_dim=4,
        file_client_args=file_client_args
        ),
    dict(type='LoadAnnotations3D'),
    dict(type='RawlevelPointCloudFusion'),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     file_client_args=file_client_args),
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
                    'transmitted_data_size'
                    ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0, #调试时用0
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dolphins_infos_train_pyr.pkl',
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
        ann_file=data_root + 'dolphins_infos_val_pyr.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dolphins_infos_val_pyr.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
runner = dict(type='EpochBasedRunner', max_epochs=24,)
evaluation = dict(interval=1, pipeline=eval_pipeline)