_base_ = [
    '../_base_/models/hv_pointpillars_fpn_dolphins.py',
    '../_base_/datasets/dolphins-3d.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
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
            reshape_out=True)))
runner = dict(type='EpochBasedRunner', max_epochs=24,)
