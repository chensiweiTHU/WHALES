# MODEL:
#     NAME: VoxelNeXt

#     VFE:
#         NAME: MeanVFE

#     BACKBONE_3D:
#         NAME: VoxelResBackBone8xVoxelNeXt

#     DENSE_HEAD:
#         NAME: VoxelNeXtHead
#         CLASS_AGNOSTIC: False
#         INPUT_FEATURES: 128

#         CLASS_NAMES_EACH_HEAD: [
#             ['car'], 
#             ['truck', 'construction_vehicle'],
#             ['bus', 'trailer'],
#             ['barrier'],
#             ['motorcycle', 'bicycle'],
#             ['pedestrian', 'traffic_cone'],
#         ]
        
#         SHARED_CONV_CHANNEL: 128
#         KERNEL_SIZE_HEAD: 3
        
#         USE_BIAS_BEFORE_NORM: True
#         NUM_HM_CONV: 2
#         SEPARATE_HEAD_CFG:
#             HEAD_ORDER: ['center', 'center_z', 'dim', 'rot', 'vel']
#             HEAD_DICT: {
#                 'center': {'out_channels': 2, 'num_conv': 2},
#                 'center_z': {'out_channels': 1, 'num_conv': 2},
#                 'dim': {'out_channels': 3, 'num_conv': 2},
#                 'rot': {'out_channels': 2, 'num_conv': 2},
#                 'vel': {'out_channels': 2, 'num_conv': 2},
#             }

#         TARGET_ASSIGNER_CONFIG:
#             FEATURE_MAP_STRIDE: 8
#             NUM_MAX_OBJS: 500
#             GAUSSIAN_OVERLAP: 0.1
#             MIN_RADIUS: 2

#         LOSS_CONFIG:
#             LOSS_WEIGHTS: {
#                 'cls_weight': 1.0,
#                 'loc_weight': 0.25,
#                 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]
#             }

#         POST_PROCESSING:
#             SCORE_THRESH: 0.1
#             POST_CENTER_LIMIT_RANGE: [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
#             MAX_OBJ_PER_SAMPLE: 500
#             NMS_CONFIG:
#                 NMS_TYPE: nms_gpu
#                 NMS_THRESH: 0.2
#                 NMS_PRE_MAXSIZE: 1000
#                 NMS_POST_MAXSIZE: 83

#     POST_PROCESSING:
#         RECALL_THRESH_LIST: [0.3, 0.5, 0.7]

#         EVAL_METRIC: kitti

" convert the config above into a dictionary "
model = dict(
    type='VoxelNeXt',
    vfe=dict(type='MeanVFE'),
    backbone_3d=dict(type='VoxelResBackBone8xVoxelNeXt'),
    dense_head=dict(
        type='VoxelNeXtHead',
        class_agnostic=False,
        input_features=128,
        class_names_each_head=[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
        shared_conv_channel=128,
        kernel_size_head=3,
        use_bias_before_norm=True,
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
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_obj_per_sample=500,
            nms_config=dict(
                nms_type='nms_gpu',
                nms_thresh=0.2,
                nms_pre_maxsize=1000,
                nms_post_maxsize=83
            )
        )
    ),
)