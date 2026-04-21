"""Compute point-in-GT-box statistics for a 3D detection dataset.

Reports the mean fraction of LiDAR points that fall inside ground-truth
boxes (and inside a 10%-dilated variant) across all samples in an info PKL.
Helpful for gauging label tightness and how much cloud is "background".
"""
import argparse

import numpy as np
from mmcv import track_iter_progress

from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.builder import build_dataset


_LOAD_PIPELINES = {
    'KittiDataset': [
        dict(type='LoadPointsFromFile', coord_type='LIDAR',
             load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    ],
    'NuScenesDataset': [
        dict(type='LoadPointsFromFile', coord_type='LIDAR',
             load_dim=5, use_dim=5),
        dict(type='LoadPointsFromMultiSweeps', sweeps_num=10,
             use_dim=[0, 1, 2, 3, 4], pad_empty_sweeps=True,
             remove_close=True),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    ],
    'WhalesDataset': [
        dict(type='LoadPointsFromFile', coord_type='LIDAR',
             load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    ],
    'WaymoDataset': [
        dict(type='LoadPointsFromFile', coord_type='LIDAR',
             load_dim=6, use_dim=5),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    ],
}


def statistic_database(dataset_class_name: str,
                       data_path: str,
                       info_path: str,
                       used_classes=None,
                       dilation: float = 0.1) -> None:
    """Run the point-in-box statistic for a single (dataset, info PKL) pair.

    Args:
        dataset_class_name: Registry name of the dataset class.
        data_path: Dataset root passed to the dataset as ``data_root``.
        info_path: Info PKL used as ``ann_file``.
        used_classes: Optional list restricting which class names are counted
            when reporting per-class object totals.
        dilation: Fractional size increase applied to every box's (L, W, H)
            when computing the dilated "near-ground-truth" coverage.
    """
    if dataset_class_name not in _LOAD_PIPELINES:
        raise KeyError(f'No loading pipeline registered for '
                       f'{dataset_class_name!r}; add one to _LOAD_PIPELINES.')

    dataset_cfg = dict(
        type=dataset_class_name,
        data_root=data_path,
        ann_file=info_path,
        pipeline=_LOAD_PIPELINES[dataset_class_name])
    if dataset_class_name == 'KittiDataset':
        dataset_cfg.update(test_mode=False, split='training')
    elif dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(use_valid_flag=True)
    elif dataset_class_name == 'WaymoDataset':
        dataset_cfg.update(test_mode=False, split='training')

    print(f'Building dataset: {dataset_class_name}')
    dataset = build_dataset(dataset_cfg)
    print(f'  samples: {len(dataset)}')

    class_counts: dict = {}
    total_pct = 0.0
    total_pct_dilated = 0.0
    n_samples = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example['ann_info']
        points = example['points'].tensor.numpy()
        gt_boxes = annos['gt_bboxes_3d'].tensor.numpy()
        names = annos['gt_names']
        if len(points) == 0 or len(gt_boxes) == 0:
            continue

        indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        total_pct += np.sum(indices) / points.shape[0]

        dilated = gt_boxes.copy()
        dilated[:, 3:6] *= (1.0 + dilation)
        indices_d = box_np_ops.points_in_rbbox(points, dilated)
        total_pct_dilated += np.sum(indices_d) / points.shape[0]
        n_samples += 1

        for name in names:
            if used_classes is not None and name not in used_classes:
                continue
            class_counts[name] = class_counts.get(name, 0) + 1

    for k, v in class_counts.items():
        print(f'  {k}: {v} objects')

    if n_samples:
        print(f'mean fraction of points inside GT   : '
              f'{total_pct / n_samples:.4f}')
        print(f'mean fraction inside GT (+{int(dilation * 100)}% dilated): '
              f'{total_pct_dilated / n_samples:.4f}')
    else:
        print('no samples with both points and boxes — nothing to report')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='WhalesDataset',
                   choices=sorted(_LOAD_PIPELINES),
                   help='Registered dataset class name (default WhalesDataset).')
    p.add_argument('--data-path', required=True,
                   help='Dataset root directory (data_root).')
    p.add_argument('--info-path', required=True,
                   help='Info PKL path (ann_file).')
    p.add_argument('--used-classes', nargs='*', default=None,
                   help='Restrict class counting to these names.')
    p.add_argument('--dilation', type=float, default=0.1,
                   help='Fractional per-axis box dilation (default 0.1).')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    statistic_database(
        dataset_class_name=args.dataset,
        data_path=args.data_path,
        info_path=args.info_path,
        used_classes=args.used_classes,
        dilation=args.dilation,
    )


if __name__ == '__main__':
    main()
