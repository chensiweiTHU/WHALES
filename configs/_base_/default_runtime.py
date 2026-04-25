custom_imports = dict(
    imports=['mmdet3d_plugin.models', 'mmdet3d_plugin.datasets'],
    allow_failed_imports=False)

checkpoint_config = dict(interval=1, save_last=True)
evaluation = dict(interval=1, save_best='pts_bbox_NuScenes/mAP', rule='greater')
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
