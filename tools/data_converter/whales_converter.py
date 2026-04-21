# Copyright (c) OpenMMLab. All rights reserved.
import math
import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d_plugin.datasets.whales_dataset import WhalesDataset

from tools.data_converter.whales import Whales

whales_categories = ('Vehicle', 'Pedestrian', 'Cyclist')

_LIDAR_HEIGHT_VEH = 1.8
_LIDAR_HEIGHT_RSU = 0.0

# All four cameras are co-located with the LiDAR; yaws taken from the
# CARLA spawn code (spawn_agents.py): front=0, left=-90, right=+90, back=180.
_VEH_CAMERA_POSES = {
    'camera':   (0.0, 0.0, _LIDAR_HEIGHT_VEH,   0.0),
    'camera_l': (0.0, 0.0, _LIDAR_HEIGHT_VEH, -90.0),
    'camera_r': (0.0, 0.0, _LIDAR_HEIGHT_VEH, +90.0),
    'camera_b': (0.0, 0.0, _LIDAR_HEIGHT_VEH, 180.0),
}
_RSU_CAMERA_POSES = {
    'camera':   (0.0, 0.0, _LIDAR_HEIGHT_RSU,   0.0),
    'camera_l': (0.0, 0.0, _LIDAR_HEIGHT_RSU, -90.0),
    'camera_r': (0.0, 0.0, _LIDAR_HEIGHT_RSU, +90.0),
    'camera_b': (0.0, 0.0, _LIDAR_HEIGHT_RSU, 180.0),
}

def create_whales_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of whales dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    
    all_train_infos = []
    all_val_infos = []

    whale = Whales(dataroot=root_path, verbose=True)

    all_samples = list(whale.frames.keys())
    num_train = int(len(all_samples) * 0.8)

    train_scenes = all_samples[:num_train]
    val_scenes = all_samples[num_train:]

    for scene in mmcv.track_iter_progress(whale.scenes):
        tot_vehicle_num = whale.scenen[scene]['vehicle_num']
        save_interval = int(whale.config[scene]["world"]['save_interval'])

        scene_files = os.listdir(osp.join(root_path, scene))
        scene_frames = [file for file in scene_files if file.isdigit()]
        steps = len(scene_frames)
        # Agent indices 0..N-1 are vehicles; index N is the RSU.
        vehicle_indices = list(range(0, tot_vehicle_num + 1))

        for i in range(steps):
            for index in vehicle_indices:
                train_infos, val_infos = _fill_trainval_infos(
                    whale, train_scenes, val_scenes, root_path, scene,
                    index, save_interval, i + 1, max_sweeps=max_sweeps)
                all_train_infos.extend(train_infos)
                all_val_infos.extend(val_infos)

    metadata = dict(version=version)
    print('total train samples: {}, total val samples: {}'.format(
        len(all_train_infos), len(all_val_infos)))
    data = dict(infos=all_train_infos, metadata=metadata)
    train_info_path = osp.join(root_path,
                         '{}_infos_train.pkl'.format(info_prefix))
    print(f"saving train_info to {train_info_path}")
    mmcv.dump(data, train_info_path)

    data = dict(infos=all_val_infos, metadata=metadata)
    val_info_path = osp.join(root_path,
                             '{}_infos_val.pkl'.format(info_prefix))
    print(f"saving val_info to {val_info_path}")
    mmcv.dump(data, val_info_path)


def _yaw_deg_to_quaternion(yaw_deg: float) -> np.ndarray:
    """Yaw-only quaternion (w, x, y, z) for a rotation about +Z."""
    yaw_rad = math.radians(float(yaw_deg))
    return np.array([
        math.cos(yaw_rad / 2.0),
        0.0,
        0.0,
        math.sin(yaw_rad / 2.0),
    ], dtype=np.float64)


def _fill_trainval_infos(whales:Whales,
                         train_scenes,
                         val_scenes,
                         root_path: str, 
                         sample: str, 
                         vehicle_index: int, 
                         save_interval: int, 
                         step: int,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        whales (:obj:`whales`): Dataset class in the whales dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_infos = []
    val_infos = []
    time_interval = 0.1 * save_interval

    boxes, cam_intrinsic, annotations, vehicle_locations, vehicle_rotations = \
        whales.get_sample_data(step, sample, vehicle_index, save_interval,
                               use_flat_vehicle_coordinates=True)
    vehicle_num = whales.scenen[sample]['vehicle_num']
    is_rsu = vehicle_index >= vehicle_num
    veh_or_rsu = 'rsu' if is_rsu else 'vehicle'
    agent_str = 'rsu' if is_rsu else f'vehicle{vehicle_index}'

    lidar_path = osp.join(
        root_path, sample, str(step * save_interval), agent_str,
        'point_cloud.bin')
    mmcv.check_file_exist(lidar_path)

    token = f'{sample}_{step*save_interval}_{vehicle_index}'
    timestamp = step * time_interval
    sample_data = whales.sample[whales._token2ind['sample'][token]]

    agent_yaw_deg = vehicle_rotations[vehicle_index][1]
    ego2global_rotation = _yaw_deg_to_quaternion(agent_yaw_deg)
    lidar_height = _LIDAR_HEIGHT_RSU if is_rsu else _LIDAR_HEIGHT_VEH

    info = {
        'lidar_path': lidar_path,
        'num_features': 4,
        "token": token,
        'sweeps': [],
        'cams': dict(),
        'lidar2ego_translation': np.array([0.0, 0.0, lidar_height]),
        'lidar2ego_rotation': np.array([1, 0, 0, 0]),
        'ego2global_translation': vehicle_locations[vehicle_index],
        'ego2global_rotation': ego2global_rotation,
        'timestamp': timestamp,
        'sample_info': sample_data,
        'veh_or_rsu': veh_or_rsu,
    }

    l2e_r = info['lidar2ego_rotation']
    l2e_t = info['lidar2ego_translation']
    e2g_r = info['ego2global_rotation']
    e2g_t = info['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 4 images' information per frame
    camera_types = [
        'camera',
        'camera_b',
        'camera_l',
        'camera_r'
    ]
    cam_pose_table = _RSU_CAMERA_POSES if is_rsu else _VEH_CAMERA_POSES
    for cam in camera_types:
        cam_path = osp.join(
            root_path, sample, str(step * save_interval), agent_str,
            cam + '.png')
        cam_info = obtain_sensor2top(
            whales,
            vehicle_locations,
            vehicle_rotations,
            cam_path,
            l2e_t, l2e_r_mat,
            e2g_t, e2g_r_mat,
            timestamp,
            vehicle_index,
            sensor_type=cam,
            is_rsu=is_rsu,
            cam_pose=cam_pose_table[cam],
        )
        cam_info.update(cam_intrinsic=cam_intrinsic)
        info['cams'].update({cam: cam_info})

    sweeps = []
    prev_step = step - 1
    while len(sweeps) < max_sweeps:
        if prev_step == 0:
            break
        prev_frame_key = f'{sample}_{prev_step*save_interval}'
        if prev_frame_key not in whales.frames[sample]:
            prev_step -= 1
            continue
        lidar_prev_path = osp.join(
            root_path, sample, str(prev_step * save_interval), agent_str,
            'point_cloud.bin')
        if not osp.exists(lidar_prev_path):
            prev_step -= 1
            continue
        prev_frame = whales.frames[sample][prev_frame_key]
        prev_veh_locations = prev_frame['veh_locations']
        prev_veh_rotations = prev_frame['veh_rotations']
        sweep = obtain_sensor2top(
            whales,
            prev_veh_locations,
            prev_veh_rotations,
            lidar_prev_path,
            l2e_t, l2e_r_mat,
            e2g_t, e2g_r_mat,
            timestamp,
            vehicle_index,
            sensor_type='lidar',
            is_rsu=is_rsu,
            cam_pose=None,
        )
        sweeps.append(sweep)
        prev_step -= 1
    info['sweeps'] = sweeps

    if not is_rsu and vehicle_index < len(annotations):
        keep_mask = np.ones(len(annotations), dtype=bool)
        keep_mask[vehicle_index] = False
        boxes = [b for b, k in zip(boxes, keep_mask) if k]
        annotations = [a for a, k in zip(annotations, keep_mask) if k]

    if len(boxes) == 0:
        info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
        info['gt_names'] = np.array([], dtype=object)
        info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
        info['valid_flag'] = np.zeros((0,), dtype=bool)
        if sample in train_scenes:
            train_infos.append(info)
        else:
            val_infos.append(info)
        return train_infos, val_infos

    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = -np.array(
        [b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
    velocity = np.array(
            [whales.box_velocity(sample, save_interval, step, anno, time_interval)[:2] for anno in annotations])

    for i in range(len(annotations)):
        v = whales.box_velocity(sample, save_interval, step, annotations[i], time_interval)
        annotations[i]['velocity'] = v

    valid_flag = np.array([True] * len(annotations), dtype=bool).reshape(-1)
    # Rotate velocities from global -> LiDAR frame.
    for i in range(len(annotations)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        velocity[i] = velo[:2]

    names = np.array([anno['type'] for anno in annotations])
    # NuScenesBox.wlh is (W, L, H); reorder to (L, W, H) for the mmdet3d
    # LiDAR box convention.
    gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
    assert len(gt_boxes) == len(annotations), \
        f'{len(gt_boxes)}, {len(annotations)}'
    info['gt_boxes'] = gt_boxes
    info['gt_names'] = names
    info['gt_velocity'] = velocity.reshape(-1, 2)
    info['valid_flag'] = valid_flag

    if sample in train_scenes:
        train_infos.append(info)
    else:
        val_infos.append(info)

    return train_infos, val_infos


def obtain_sensor2top(whales,
                      vehicle_locations,
                      vehicle_rotations,
                      data_path,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      timestamp,
                      index,
                      sensor_type='lidar',
                      is_rsu=False,
                      cam_pose=None):
    """Build a sweep-style info dict with sensor->top-LiDAR transforms.

    Args:
        vehicle_locations (list): per-agent (x, y, z) in CARLA world.
        vehicle_rotations (list): per-agent (pitch, yaw, roll) in degrees.
        data_path (str): sensor file path to store in the info.
        l2e_t, l2e_r_mat, e2g_t, e2g_r_mat: the key-frame's lidar->ego and
            ego->global transforms (the ego frame is the agent body).
        index (int): agent index into vehicle_locations / vehicle_rotations.
        sensor_type (str): e.g. 'camera', 'camera_l', 'lidar'.
        is_rsu (bool): True if this agent is the RSU (zero lidar height).
        cam_pose (tuple | None): (x, y, z, yaw_deg) of a camera in the agent
            body frame, or None when the sensor is the LiDAR itself.

    Returns:
        dict: Sweep info with sensor2lidar_{rotation,translation}.
    """
    if cam_pose is None:
        sensor_ego_rotation = np.array([1, 0, 0, 0], dtype=np.float64)
        sensor_ego_translation = np.array(
            [0.0, 0.0, _LIDAR_HEIGHT_RSU if is_rsu else _LIDAR_HEIGHT_VEH],
            dtype=np.float64)
    else:
        cam_x, cam_y, cam_z, cam_yaw = cam_pose
        sensor_ego_rotation = _yaw_deg_to_quaternion(cam_yaw)
        sensor_ego_translation = np.array(
            [cam_x, cam_y, cam_z], dtype=np.float64)

    agent_yaw_deg = vehicle_rotations[index][1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sensor2ego_translation': sensor_ego_translation,
        'sensor2ego_rotation': sensor_ego_rotation,
        'ego2global_translation': vehicle_locations[index],
        'ego2global_rotation': _yaw_deg_to_quaternion(agent_yaw_deg),
        'timestamp': timestamp,
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export a mono3D COCO-style JSON from a WHALES info PKL.

    Args:
        root_path (str): dataset root (used to store relative image paths).
        info_path (str): path to the whales_infos_*.pkl to read.
        version (str): dataset version string (carried through metadata).
        mono3d (bool): whether to also emit camera-frame 3D annotations.
    """
    camera_types = ['camera', 'camera_b', 'camera_l', 'camera_r']

    whales_infos = mmcv.load(info_path)['infos']
    cat2Ids = [
        dict(id=whales_categories.index(cat_name), name=cat_name)
        for cat_name in whales_categories
    ]
    coco_ann_id = 0
    img_id_counter = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)

    for info in mmcv.track_iter_progress(whales_infos):
        token = info['token']   # scene_frame_agent
        gt_boxes = info['gt_boxes']          # (N, 7) in lidar frame
        gt_names = info['gt_names']
        gt_velocity = info.get('gt_velocity', np.zeros((len(gt_boxes), 2)))

        for cam in camera_types:
            cam_info = info['cams'][cam]
            cam_path = cam_info['data_path']
            try:
                (height, width, _) = mmcv.imread(cam_path).shape
            except Exception:
                height, width = 1080, 1920

            image_id = f'{token}_{cam}'
            rel_path = osp.relpath(cam_path, root_path)
            coco_2d_dict['images'].append(
                dict(
                    file_name=rel_path,
                    id=image_id,
                    token=token,
                    cam_type=cam,
                    cam2ego_rotation=np.asarray(
                        cam_info['sensor2ego_rotation']).tolist(),
                    cam2ego_translation=np.asarray(
                        cam_info['sensor2ego_translation']).tolist(),
                    ego2global_rotation=np.asarray(
                        info['ego2global_rotation']).tolist(),
                    ego2global_translation=np.asarray(
                        info['ego2global_translation']).tolist(),
                    cam_intrinsic=np.asarray(
                        cam_info['cam_intrinsic']).tolist(),
                    width=width,
                    height=height,
                ))
            img_id_counter += 1

            coco_infos = get_2d_boxes(
                gt_boxes=gt_boxes,
                gt_names=gt_names,
                gt_velocity=gt_velocity,
                cam_info=cam_info,
                image_id=image_id,
                file_name=rel_path,
                img_hw=(height, width),
                mono3d=mono3d,
            )
            for coco_info in coco_infos:
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1

    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


_CARLA_CAM_TO_OPENCV = np.array([
    [0.0,  1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)


def _lidar_box_corners(gt_boxes: np.ndarray) -> np.ndarray:
    """Return the 8 corner xyz positions for each (N, 7) LiDAR-frame box."""
    N = gt_boxes.shape[0]
    if N == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    cx, cy, cz = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
    L, W, H = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
    yaw = gt_boxes[:, 6]
    off = np.array([[+1, +1, -1], [+1, -1, -1], [-1, -1, -1], [-1, +1, -1],
                    [+1, +1, +1], [+1, -1, +1], [-1, -1, +1], [-1, +1, +1]],
                   dtype=np.float32) * 0.5
    half = np.stack([L, W, H], axis=1)[:, None, :]
    local = off[None, :, :] * half
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.stack([
        np.stack([c, -s, np.zeros_like(c)], axis=1),
        np.stack([s,  c, np.zeros_like(c)], axis=1),
        np.stack([np.zeros_like(c), np.zeros_like(c), np.ones_like(c)], axis=1),
    ], axis=1)
    rotated = np.einsum('nij,nkj->nki', R, local)
    centre = np.stack([cx, cy, cz], axis=1)[:, None, :]
    return rotated + centre


def _lidar_to_camera_opencv(points_lidar: np.ndarray,
                            cam_info: dict) -> np.ndarray:
    """Transform Nx3 LiDAR-frame (y=left) points into the OpenCV camera frame."""
    p_carla = np.asarray(points_lidar, dtype=np.float64).copy()
    p_carla[..., 1] = -p_carla[..., 1]
    R_s2l = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float64)
    t_s2l = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float64)
    p_cam_carla = (p_carla - t_s2l) @ R_s2l
    return p_cam_carla @ _CARLA_CAM_TO_OPENCV.T


def get_2d_boxes(gt_boxes: np.ndarray,
                 gt_names: np.ndarray,
                 gt_velocity: np.ndarray,
                 cam_info: dict,
                 image_id: str,
                 file_name: str,
                 img_hw: Tuple[int, int],
                 mono3d: bool = True) -> List[dict]:
    """Project LiDAR-frame GT boxes into one camera and build COCO records.

    Args:
        gt_boxes (np.ndarray): (N, 7) in the WHALES LiDAR frame.
        gt_names (np.ndarray): (N,) class names aligned with gt_boxes.
        gt_velocity (np.ndarray): (N, 2) LiDAR-frame xy velocity or None.
        cam_info (dict): one entry from info['cams'][cam_type].
        image_id (str): unique COCO image identifier.
        file_name (str): image path stored on the annotation.
        img_hw (tuple): (height, width) of the rendered image.
        mono3d (bool): if True, add camera-frame 3D fields per annotation.
    """
    H, W = img_hw
    K = np.asarray(cam_info['cam_intrinsic'])
    recs: List[dict] = []
    if gt_boxes.shape[0] == 0:
        return recs

    # Camera yaw in the agent body frame, recovered from sensor2lidar (the
    # LiDAR has zero yaw, so this is the camera's yaw relative to LiDAR).
    R_s2l = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float64)
    cam_yaw_rad = math.atan2(R_s2l[1, 0], R_s2l[0, 0])

    corners_lidar = _lidar_box_corners(gt_boxes)
    N = corners_lidar.shape[0]
    corners_cam = _lidar_to_camera_opencv(
        corners_lidar.reshape(-1, 3), cam_info).reshape(N, 8, 3)
    centres_lidar = gt_boxes[:, :3].astype(np.float64)
    centres_cam = _lidar_to_camera_opencv(centres_lidar, cam_info)

    for i in range(N):
        name = gt_names[i] if len(gt_names) else 'Vehicle'
        if name not in whales_categories:
            continue
        corners = corners_cam[i]
        in_front = corners[:, 2] > 0
        if in_front.sum() < 2:
            continue
        corners_visible = corners[in_front]
        uv = (K @ corners_visible.T).T
        uv_px = uv[:, :2] / uv[:, 2:3]
        poly = post_process_coords(uv_px.tolist(), imsize=(W, H))
        if poly is None:
            continue
        min_x, min_y, max_x, max_y = poly

        rec = generate_record(
            ann_name=name,
            x1=float(min_x), y1=float(min_y),
            x2=float(max_x), y2=float(max_y),
            image_id=image_id,
            file_name=file_name,
        )

        if mono3d:
            cx, cy, cz = centres_cam[i]
            if cz <= 0:
                continue
            L, W_box, H_box = gt_boxes[i, 3], gt_boxes[i, 4], gt_boxes[i, 5]
            # mmdet3d camera-frame ry rotates around +y (down). The LiDAR
            # yaw must be corrected for the camera's own orientation in the
            # agent body (the y-flip from LiDAR to camera already flips the
            # lidar-yaw sign, and the camera rotation subtracts cam_yaw).
            ry = -float(gt_boxes[i, 6]) - cam_yaw_rad
            rec['bbox_cam3d'] = [float(cx), float(cy), float(cz),
                                 float(L), float(H_box), float(W_box), ry]

            if gt_velocity is not None and len(gt_velocity) > i:
                vx_l, vy_l = float(gt_velocity[i, 0]), float(gt_velocity[i, 1])
                v_lidar_carla = np.array([vx_l, -vy_l, 0.0])
                v_cam = _CARLA_CAM_TO_OPENCV @ v_lidar_carla
                rec['velo_cam3d'] = [float(v_cam[0]), float(v_cam[2])]
            else:
                rec['velo_cam3d'] = [0.0, 0.0]

            centre = np.array([[cx, cy, cz]], dtype=np.float64)
            centre2d = points_cam2img(centre, K, with_depth=True)
            rec['center2d'] = centre2d.squeeze().tolist()

            rec['attribute_name'] = name
            rec['attribute_id'] = whales_categories.index(name)

        recs.append(rec)

    return recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_name: str,
                    x1: float, y1: float, x2: float, y2: float,
                    image_id: str,
                    file_name: str) -> dict:
    """Build a single COCO-style 2D annotation record."""
    return {
        'file_name': file_name,
        'image_id': image_id,
        'area': float((y2 - y1) * (x2 - x1)),
        'category_name': ann_name,
        'category_id': whales_categories.index(ann_name),
        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        'iscrowd': 0,
    }
