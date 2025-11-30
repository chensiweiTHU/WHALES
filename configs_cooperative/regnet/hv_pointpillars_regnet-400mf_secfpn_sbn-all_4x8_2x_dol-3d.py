# _base_ = './hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_dol-3d.py'
# model settings
_base_ = [
    '../_base_/models/hv_pointpillars_fpn_dolphins.py',
    '../_base_/datasets/dolphins-3d.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
# model settings

model = dict(
    type='MVXFasterRCNN',
    pts_backbone=dict(
        _delete_=True,
        type='NoStemRegNet',
        arch=dict(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22, bot_mul=1.0),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_400mf'),
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        strides=(1, 2, 2, 2),
        base_channels=64,
        stem_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        norm_eval=False,
        style='pytorch'),
    pts_neck=dict(
        type='SECONDFPN',
        _delete_=True,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 160, 384],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-49.6, -49.6, -1.8, 49.6, 49.6, -1.4],
                [-49.6, -49.6, -1.7, 49.6, 49.6, -1.3],
                [-49.6, -49.6, -2, 49.6, 49.6, -1.6],
            ],
            sizes=[
                [5, 2.2, 1.8],  # car
                [0.60, 0.60, 1.8],  # pedestrian
                [2.1, 0.80, 1.6],  # bicycle
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True)))
