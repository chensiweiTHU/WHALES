_base_ = './hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_dol-3d.py'
# model settings
model = dict(
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
                [2.5, 1.1, 0.9],  # car
                [0.30, 0.30, 0.9],  # pedestrian
                [1.05, 0.40, 0.80],  # bicycle
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True)))
