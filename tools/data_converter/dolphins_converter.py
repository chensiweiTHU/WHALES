# Copyright (c) OpenMMLab. All rights reserved.
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
from mmdet3d.datasets.dolphins_dataset import DolphinsDataset

from tools.data_converter.dolphins import Dolphins

dolphins_categories = ('Vehicle', 'Pedestrian', 'Cyclist')

def create_dolphins_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of dolphins dataset.

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

    dolphins = Dolphins(dataroot=root_path, verbose=True)

    all_samples = list(dolphins.frames.keys())
    num_train = int(len(all_samples) * 0.8)

    train_scenes = all_samples[:num_train]
    val_scenes = all_samples[num_train:]

    # filter existing scenes.
    for scene in mmcv.track_iter_progress(dolphins.scenes):
        # print(scene)
        tot_vehicle_num = dolphins.scenen[scene]['vehicle_num']
        
        save_interval = int(dolphins.config[scene]["world"]['save_interval'])
        
        steps = 0
        scene_files = os.listdir(osp.join(root_path, scene))
        scene_frames = [file for file in scene_files if file.isdigit()]

        steps = len(scene_frames)
        # print(steps)
        # iterate through all vehicles
        vehicle_indices = [i for i in range(0, tot_vehicle_num+1)]

        for i in range(steps):
            for index in vehicle_indices:
                train_infos, val_infos = _fill_trainval_infos(
                    dolphins, train_scenes, val_scenes, root_path, scene, index, save_interval, i+1, max_sweeps=max_sweeps)

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


def _fill_trainval_infos(dolphins:Dolphins, 
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
        dolphins (:obj:`Dolphins`): Dataset class in the dolphins dataset.
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

    # step = 1
    time_interval = 0.1 * save_interval

    # for sample in mmcv.track_iter_progress(dolphins.frames):
    boxes, cam_intrinsic, annotations, vehicle_locations, vehicle_rotations = dolphins.get_sample_data(step, sample, vehicle_index, save_interval,\
                                                                                        use_flat_vehicle_coordinates=True)
    if vehicle_index< dolphins.scenen[sample]['vehicle_num']-1:
        lidar_path = osp.join(root_path, sample, str(step * save_interval), 'vehicle' + str(vehicle_index), 'point_cloud.bin')
    else:
        lidar_path = osp.join(root_path, sample, str(step * save_interval),'rsu', 'point_cloud.bin')
    mmcv.check_file_exist(lidar_path)
    token = f'{sample}_{step*save_interval}_{vehicle_index}'
    timestamp = step * time_interval
    sample_data = dolphins.sample[dolphins._token2ind['sample'][token]] 
    rotation = sample_data['sample_annotation'][vehicle_index]['rotation']
    veh_or_rsu = 'vehicle' if vehicle_index < dolphins.scenen[sample]['vehicle_num']-1 else 'rsu'

    info = {
        'lidar_path': lidar_path,
        'num_features': 5,
        "token": token,
        'sweeps': [],
        'cams': dict(),
        'lidar2ego_translation': np.zeros(3),
        'lidar2ego_rotation': np.array([1, 0, 0, 0]),
        'ego2global_translation': vehicle_locations[vehicle_index],
        'ego2global_rotation': rotation,
        'timestamp': timestamp,
        'sample_info': sample_data,
        "veh_or_rsu": veh_or_rsu
    }

    l2e_r = info['lidar2ego_rotation']
    l2e_t = info['lidar2ego_translation']
    e2g_r = info['ego2global_rotation']
    e2g_t = info['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 6 images' information per frame
    camera_types = [
        'camera',
        'camera_b',
        'camera_l',
        'camera_r'
    ]
    agent_str = 'vehicle' + str(vehicle_index) if veh_or_rsu == 'vehicle' else 'rsu'
    for cam in camera_types:
        cam_path = osp.join(root_path, sample, str(step * save_interval), agent_str, cam + '.png' )
        cam_info = obtain_sensor2top(dolphins, vehicle_locations, cam_path, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, timestamp, vehicle_index, cam)
        cam_info.update(cam_intrinsic=cam_intrinsic)
        info['cams'].update({cam: cam_info})

    # obtain sweeps for a single key-frame
    sweeps = []
    prev_step = step - 1
    # print("prev_step", prev_step)
    while len(sweeps) < max_sweeps:
        if prev_step == 0:
            break
        lidar_prev_path = osp.join(root_path, sample, str(prev_step * save_interval), 'vehicle' + str(vehicle_index), 'point_cloud.bin')
        prev_veh_locations = dolphins.frames[sample][f'{sample}_{prev_step*save_interval}']['veh_locations']
        sweep = obtain_sensor2top(dolphins, prev_veh_locations, lidar_prev_path, l2e_t,
                                    l2e_r_mat, e2g_t, e2g_r_mat, timestamp, vehicle_index, 'lidar')
        sweeps.append(sweep)
        prev_step -= 1
    info['sweeps'] = sweeps

    # annotations = sample['sample_annotation']
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
    velocity = np.array(
            [dolphins.box_velocity(sample, save_interval, step, anno, time_interval)[:2] for anno in annotations]) 
    
    for i in range(len(annotations)):
        v = dolphins.box_velocity(sample, save_interval, step, annotations[i], time_interval)
        annotations[i]['velocity'] = v
    
    valid_flag = np.array(
        [True for anno in annotations],
            dtype=bool).reshape(-1)
    # convert velo from global to lidar
    for i in range(len(annotations)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
            l2e_r_mat).T
        velocity[i] = velo[:2]
        # print(velocity[i])
    
    names = [anno['type'] for anno in annotations]
    names = np.array(names)
    # we need to convert box size to
    # the format of our lidar coordinate system
    # which is x_size, y_size, z_size (corresponding to l, w, h)
    gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
    assert len(gt_boxes) == len(
        annotations), f'{len(gt_boxes)}, {len(annotations)}'
    info['gt_boxes'] = gt_boxes
    info['gt_names'] = names
    info['gt_velocity'] = velocity.reshape(-1, 2)

    # info['num_lidar_pts'] = ?????
    # info['num_radar_pts'] = ?????
    info['valid_flag'] = valid_flag

    # TODO: define train scenes and rest as val scenes (8:2)
    # Current: Manually define the train/val scenes


    # print(train_scenes)
    # print(val_scenes)
    # print(info)
    if sample in train_scenes:
        train_infos.append(info)
    else:
        val_infos.append(info)
        # step += 1

    return train_infos, val_infos


def obtain_sensor2top(dolphins,
                      vehicle_locations,
                      data_path,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      timestamp,
                      index,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        dolphins (class): Dataset class in the Dolphins dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """

    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sensor2ego_translation': np.zeros(3),
        'sensor2ego_rotation': np.array([0, 0, 0, 1]),
        'ego2global_translation': vehicle_locations[index],
        'ego2global_rotation': dolphins.euler_to_quaternion(vehicle_locations[index]),
        'timestamp': timestamp,
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    index = 0
    
    # get bbox annotations for camera
    camera_types = [
        'camera',
        'camera_b',
        'camera_l',
        'camera_r'
    ]

    dolphins_infos = mmcv.load(info_path)['infos']
    dolphins = Dolphins(version=version, dataroot=root_path, verbose=True)
    # save_interval = int(dolphins.config[scene]["world"]['save_interval'])
    save_interval = 5
    # info_2d_list = []
    cat2Ids = [
        dict(id=dolphins_categories.index(cat_name), name=cat_name)
        for cat_name in dolphins_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(dolphins_infos):
        # print(info)
        step = int(int(info['token'].split('_')[1])/save_interval)
        sample = info['lidar_path'].split('/')[3]
        # print("cur step:", step)
        for cam in camera_types:
            cam_info = info['cams'][cam]
            # print(cam_info)
            cam_path = cam_info['data_path']
            # print(cam_path)
            coco_infos = get_2d_boxes(
                dolphins,
                sample,
                step,
                index,
                save_interval,
                cam_path,
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/dolphins/')
                    [-1],
                    id=cam_info['type'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(dolphins,
                 sample: str,
                 step: int,
                 index: int,
                 save_interval: int,
                 cam_path: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    # frame = dolphins.frames
    # all_samples = list(dolphins.frames.keys())

    

    boxes, camera_intrinsic, annotations, vehicle_locations, vehicle_rotations = dolphins.get_sample_data(step, sample, index, save_interval)
        # ann_recs = frame[sample][f'{sample}_{step*save_interval}']['sample_annotation']
        # print(annotations)
        # print(ann_recs)
        
        # import pdb
        # pdb.set_trace()
        # veh_locations = frame[sample][f'{sample}_{step*save_interval}']['veh_locations']
        # veh_rotations = frame[sample][f'{sample}_{step*save_interval}']['veh_rotations']
    # print(frame)
        # print(boxes)

    # assert sd_rec[
    #     'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
    #     ' for camera sample_data!'
    # if not sd_rec['is_key_frame']:
    #     raise ValueError(
    #         'The 2D re-projections are available only for keyframes.')

        # for ann_rec in ann_recs:
    repro_recs = []

    for ann_rec in annotations:
        # print(ann_rec)
        # Augment sample_annotation with token information.

        # Get the box in global coordinates.
        "modified by siwei"
        box = dolphins.get_box(ann_rec)#,dolphins.scenen[sample]['vehicle_num'])

        # print(box)

        # Move them to the ego-pose frame.
        box.translate(-np.array(vehicle_locations[index]))
        box.rotate(Quaternion(dolphins.euler_to_quaternion(vehicle_rotations[index])).inverse)

        # Move them to the calibrated sensor frame.

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        if len(corner_coords)>2:
            final_coords = post_process_coords(corner_coords)
        else:
            final_coords = None
        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    step, cam_path)

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = dolphins.box_velocity(sample, save_interval, step, ann_rec, save_interval * 0.1)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            # print(len(vehicle_locations[index]), vehicle_locations[index])
            e2g_r_mat = Quaternion(dolphins.euler_to_quaternion(vehicle_rotations[index])).rotation_matrix
            c2e_r_mat = Quaternion([0, 0, 0, 1]).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            attr_name = box.name
            attr_id = dolphins_categories.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


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


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    step: int, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = step
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = filename.split('/')[-1].split('.')[0] + '_' + str(step)
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    cat_name = ann_rec['type']
    coco_rec['category_name'] = cat_name
    # print(cat_name)
    coco_rec['category_id'] = dolphins_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec

