_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/imvotenet_image.py',
]

class_names = ('Vehicle', 'Pedestrian', 'Cyclist')

model = dict(
    img_roi_head=dict(
        bbox_head=dict(num_classes=len(class_names))))

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1600, 900), (1600, 810), (1600, 720)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/whales/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'whales_infos_train_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'whales_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'whales_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6])
runner = dict(type='EpochBasedRunner', max_epochs=8)

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'  # noqa
