# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pickle
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from os import path as osp
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets.builder import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

def statistic_database(dataset_class_name,
                        data_path, 
                        info_path=None,
                        mask_anno_path=None,
                        used_classes=None,
                        with_mask=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (（)str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        mask_anno_path (str): Path of the mask_anno.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """
    print(f'Get GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        file_client_args = dict(backend='disk')
        dataset_cfg.update(
            test_mode=False,
            split='training',
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=with_mask,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    file_client_args=file_client_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=file_client_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'DolphinsDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4),
                        dict(
                type='LoadPointsFromCooperativeAgents',
                coord_type='LIDAR',
                load_dim=4, use_dim=4,
                
                ),
    # dict(type='LoadAnnotations3D'),
                dict(type='RawlevelPointCloudFusion'),
                # dict(
                #     type='LoadPointsFromMultiSweeps',
                #     sweeps_num=10,
                #     load_dim=5,
                #     use_dim=[0, 1, 2, 3, 4],
                #     pad_empty_sweeps=True,
                #     remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'WaymoDataset':
        file_client_args = dict(backend='disk')
        dataset_cfg.update(
            test_mode=False,
            split='training',
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=5,
                    file_client_args=file_client_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=file_client_args)
            ])
    print("building dataset")
    dataset = build_dataset(dataset_cfg)
    print("dataset built")

    all_db_infos = dict()
    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info['file_name']: i})
    print("start loading data")

    group_counter = 0
    total_percentage_in_gt = 0.0
    total_dilated_percentage_in_gt = 0.0
    point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].tensor.numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
        names = annos['gt_names']
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        # print(f'percentage of file {j} in gt')
        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)
        percentage_in_gt = np.sum(point_indices) / points.shape[0]
        total_percentage_in_gt += percentage_in_gt
        # print(f'percentage_in_gt: {percentage_in_gt}, num_points_in_gt: {np.sum(point_indices)}, num_points_total: {points.shape[0]}')

        dilated_gt_boxes_3d = gt_boxes_3d.copy()
        dilated_gt_boxes_3d[:, 3:6] += dilated_gt_boxes_3d[:, 3:6] * 0.1
        dilated_point_indices = box_np_ops.points_in_rbbox(points, dilated_gt_boxes_3d)
        percentage_in_dilated_gt = np.sum(dilated_point_indices) / points.shape[0] 
        total_dilated_percentage_in_gt += percentage_in_dilated_gt
        # print(f'percentage_in_dilated_gt: {percentage_in_dilated_gt}, num_points_in_dilated_gt: {np.sum(dilated_point_indices)}, num_points_total: {points.shape[0]}')

        # visualize bev with points and gt boxes
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # from matplotlib.collections import PatchCollection
        # fig, ax = plt.subplots()
        # ax.scatter(points[:, 0], points[:, 1], s=1, c='b')
        # for i in range(num_obj):
        #     gt_box = gt_boxes_3d[i]
        #     rect = Rectangle((gt_box[0], gt_box[1]), gt_box[3], gt_box[4], angle=gt_box[6], edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # savedir = '/home/ubuntu/Projects/v2x-agents/tools/visualization'
        # plt.savefig(f'{savedir}/{j}.png')
        # plt.close(fig)
        # input("Press Enter to continue...")

        if with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in file2id.keys():
                print(f'skip image {img_path} for empty mask')
                continue

        for i in range(num_obj):
            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]
            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_total': points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')
    
    #calculate average percentage of points in gt

    print(f'average percentage of points in gt: {total_percentage_in_gt / len(dataset)}')
    print(f'average percentage of points in gt: {total_dilated_percentage_in_gt / len(dataset)}')


        
def main():
    # pointcloud = np.fromfile('/home/ubuntu/Projects/v2x-agents/data/DAIR-V2X-C/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne/000010.bin', dtype=np.float32,count=-1)
    # print(pointcloud.shape)
    # pointcloud = pointcloud.reshape([-1, 4])
    # print(pointcloud.shape[0])
    # print(pointcloud)
    statistic_database(
        dataset_class_name='KittiDataset',
        data_path='/home/ubuntu/Projects/v2x-agents/data/DAIR-V2X-C/cooperative-vehicle-infrastructure/vic3d-early-fusion-training',
        info_path='/home/ubuntu/Projects/v2x-agents/data/DAIR-V2X-C/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/kitti_infos_train.pkl',
        used_classes=['Car'],
        with_mask=False)
    input("Press Enter to continue...") 
    statistic_database(
        dataset_class_name='DolphinsDataset',
        data_path='/home/ubuntu/Projects/v2x-agents/data/dolphins-new',
        info_path='/home/ubuntu/Projects/v2x-agents/data/dolphins-new/dolphins_infos_train.pkl',
        used_classes=['Car'],
        with_mask=False)
    
if __name__ == '__main__':
    main()