import numpy as np
import scipy
import torch
import copy
from scipy.spatial import Delaunay

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from . import common_utils, box_utils

def points_in_boxes_cpu(points, boxes):
    """
    判断点云中哪些点在3D检测框内。

    Args:
        points (numpy.ndarray): 原始点云数据，形状为 (N, 3) 或 (N, 4)
        boxes (numpy.ndarray): groundtruth 检测框，形状为 (M, 7)

    Returns:
        numpy.ndarray: 布尔矩阵，形状为 (M, N)，表示每个检测框内的点
    """
    # 初始化布尔矩阵，形状为 (M, N)
    num_boxes = boxes.shape[0]
    num_points = points.shape[0]
    points_in_boxes = np.zeros((num_boxes, num_points), dtype=bool)

    for i in range(num_boxes):
        box = boxes[i]
        x, y, z, dx, dy, dz, heading = box[:7]

        # 计算检测框的8个顶点坐标
        corners = np.array([
            [dx / 2, dy / 2, dz / 2],
            [-dx / 2, dy / 2, dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [dx / 2, dy / 2, -dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, -dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
        ])

        # 旋转检测框的顶点
        rotation_matrix = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # 平移检测框的顶点
        translated_corners = rotated_corners + np.array([x, y, z])

        # 判断点是否在检测框内
        min_corner = np.min(translated_corners, axis=0)
        max_corner = np.max(translated_corners, axis=0)

        mask = np.all((points[:, :3] >= min_corner) & (points[:, :3] <= max_corner), axis=1)
        points_in_boxes[i] = mask

    return points_in_boxes

def points_in_gt_bbox(points, gt_boxes):
    """
    Args:
        points: (N, 3)
        gt_boxes: (B, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
        points: (M, 3)
        mask: (M)
    """
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    gt_boxes, _ = common_utils.check_numpy_to_torch(gt_boxes)

    gt_boxes_corners = box_utils.boxes3d_to_corners3d_kitti_camera(gt_boxes, rotate=True)
    mask = box_utils.points_in_boxes_cpu(points[:, 0:3], gt_boxes_corners)
    mask = mask.any(0)
    return points[mask], mask
