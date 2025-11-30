
# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes

import os
import cv2

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.saved_count = 0

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        visualize = False
        if visualize:
            img_filenames = results.get('img_filename', [])
            lidar2img_list = results.get('lidar2img', [])
            
            # Order matters! Check specific names before generic ones
            cam_name_to_view = {
                'camera_b': 'back',
                'camera_l': 'left',
                'camera_r': 'right',
                'camera': 'front',
            }
            
            # Build view data dictionary
            view_data = {}
            for idx, (img_path, lidar2img) in enumerate(zip(img_filenames, lidar2img_list)):
                for cam_name, view_name in cam_name_to_view.items():
                    if cam_name in img_path:
                        view_data[view_name] = {
                            'img_idx': idx,
                            'lidar2img': lidar2img
                        }
                        print(f"Matched '{cam_name}' in {img_path} → {view_name}")
                        break
            
            boxes_3d = results['gt_bboxes_3d']
            boxe_corners = boxes_3d.corners.cpu().numpy()  # Shape: (N, 8, 3)
            
            # DEBUG: Print first box coordinates
            print(f"\nFirst GT box corners (LiDAR coords):\n{boxe_corners[0]}")
            print(f"First GT box center: {boxes_3d.gravity_center[0]}")
            print(f"First GT box dims: {boxes_3d.dims[0]}")
            
            view_names = ['front', 'back', 'left', 'right']
            
            for view_name in view_names:
                if view_name not in view_data:
                    print(f'Skipping {view_name} - not found in data')
                    continue
                    
                print(f"\n{'='*60}")
                print(f"Processing {view_name} view")
                print(f"{'='*60}")
                
                view_idx = view_data[view_name]['img_idx']
                lidar2img = np.array(view_data[view_name]['lidar2img'])
                
                # DEBUG: Print transformation matrix
                print(f"lidar2img shape: {lidar2img.shape}")
                print(f"lidar2img matrix:\n{lidar2img}")
                
                if lidar2img.shape == (4, 4):
                    lidar2img = lidar2img[:3, :]
                
                # Take first box for detailed debugging
                first_box_corners = boxe_corners[0]  # Shape: (8, 3)
                first_box_homo = np.concatenate(
                    [first_box_corners, np.ones((8, 1))], axis=1
                )  # Shape: (8, 4)
                
                # Project first box
                first_box_2d = (lidar2img @ first_box_homo.T).T  # Shape: (8, 3)
                
                print(f"\nFirst box after projection (before division):")
                print(f"  X range: {first_box_2d[:, 0].min():.2f} to {first_box_2d[:, 0].max():.2f}")
                print(f"  Y range: {first_box_2d[:, 1].min():.2f} to {first_box_2d[:, 1].max():.2f}")
                print(f"  Z range: {first_box_2d[:, 2].min():.2f} to {first_box_2d[:, 2].max():.2f}")
                
                # Check if box is in front of camera
                if (first_box_2d[:, 2] > 0).all():
                    first_box_pixels = first_box_2d[:, :2] / first_box_2d[:, 2:3]
                    print(f"\nFirst box pixel coordinates:")
                    print(f"  X range: {first_box_pixels[:, 0].min():.2f} to {first_box_pixels[:, 0].max():.2f}")
                    print(f"  Y range: {first_box_pixels[:, 1].min():.2f} to {first_box_pixels[:, 1].max():.2f}")
                else:
                    print(f"\nFirst box is behind camera!")
                
                # Process all boxes
                boxe_corners_flat = boxe_corners.reshape(-1, 3)
                boxe_corners_homo = np.concatenate(
                    [boxe_corners_flat, np.ones((boxe_corners_flat.shape[0], 1))], 
                    axis=1
                )
                
                boxe_corners_2d = (lidar2img @ boxe_corners_homo.T).T
                boxe_corners_2d = boxe_corners_2d.reshape(-1, 8, 3)
                
                valid_boxes = []
                for i, box in enumerate(boxe_corners_2d):
                    if (box[:, 2] > 0).all():
                        box_2d = box[:, :2] / box[:, 2:3]
                        valid_boxes.append(box_2d)
                
                view_img = img[..., view_idx].copy()
                view_img = np.ascontiguousarray(view_img)
                
                # DEBUG: Print image shape
                print(f"Image shape: {view_img.shape}")
                
                if len(valid_boxes) > 0:
                    boxe_corners_2d = np.array(valid_boxes)
                    view_img = self.draw_boxes(view_img, boxe_corners_2d, color=(0, 255, 0))
                    print(f'Drew {len(valid_boxes)} GT boxes on {view_name}')
                else:
                    print(f'No valid GT boxes for {view_name}')
                
                # Test boxes
                # test_boxes_3d = LiDARInstance3DBoxes(
                #     [[2.8, 0, -1.8, 5, 2, 1.8, 0],
                #     [2.8, 2, -1.8, 5, 2, 1.8, 0]],
                # )
                # test_corners = test_boxes_3d.corners.cpu().numpy()
                # test_corners_flat = test_corners.reshape(-1, 3)
                # test_corners_homo = np.concatenate(
                #     [test_corners_flat, np.ones((test_corners_flat.shape[0], 1))], 
                #     axis=1
                # )
                
                # test_2d = (lidar2img @ test_corners_homo.T).T
                # test_2d = test_2d.reshape(-1, 8, 3)
                
                # test_valid_boxes = []
                # for box in test_2d:
                #     if (box[:, 2] > 0).all():
                #         box_2d = box[:, :2] / box[:, 2:3]
                #         test_valid_boxes.append(box_2d)
                
                # if len(test_valid_boxes) > 0:
                #     test_corners_2d = np.array(test_valid_boxes)
                #     view_img = self.draw_boxes(view_img, test_corners_2d, color=(0, 255, 255))
                device_id = int(os.environ.get('LOCAL_RANK', -1))
                if device_id < 0 and torch.cuda.is_available():
                    device_id = torch.cuda.current_device()
                device_prefix = f'gpu{device_id}_' if device_id >= 0 else ''
                output_path = f'visualizations/{device_prefix}camera_{view_name}_{self.saved_count}.jpg'                
                output_path = f'visualizations/camera_{view_name}_{self.saved_count}.jpg'
                cv2.imwrite(output_path, view_img)
                print(f'Written {output_path}')
        # if visualize:
        #     #lidar2img = results['lidar2img'][0] # 4x4 matrix
        #     vel_ego_to_cam = np.array([[0,-1,0,0], [0,0,-1,0], [1,0,0,0]])
        #     # lidar2img = results['img_info']['viewpad'] # 4x4 matrix
        #     # print(results['lidar2img'][0])
        #     lidar2img = results['lidar2img'][0]
        #     lidar2img = lidar2img[:3,:3]
        #     lidar2img[0,0] = 960
        #     lidar2img[1,1] = 960
        #     boxes_3d = results['gt_bboxes_3d']
        #     boxe_corners = boxes_3d.corners
        #     dims = boxes_3d.dims.cpu().numpy()
        #     # reshape to NX8,3
        #     boxe_corners = boxe_corners.reshape(-1, 3)
        #     # repeat dims
        #     repeated_dims = np.repeat(dims, 8, axis=0)
        #     boxe_corners[:,0] = boxe_corners[:,0]#+0.5*repeated_dims[:,0]
        #     # to numpy
        #     boxe_corners = boxe_corners.cpu().numpy()
        #     # pad 1 to convert to homogenuous coordinates
        #     boxe_corners = np.concatenate([boxe_corners, np.ones((boxe_corners.shape[0],1))], axis=1)
        #     # apply transformation
        #     boxe_corners_2d = np.matmul(vel_ego_to_cam, boxe_corners.T).T
        #     boxe_corners_2d = np.matmul(lidar2img, boxe_corners_2d.T).T
        #     # filter z<0 and divide z
        #     boxe_corners_2d = boxe_corners_2d.reshape(-1, 8, 3)
        #     newboxes = []
        #     for box in boxe_corners_2d:
        #         if (box[:,2]>0).all():
        #             # divide depth
        #             box = box/box[:,2].reshape((-1,1))
        #             newboxes.append(box)
        #     boxe_corners_2d = np.asarray(newboxes)
        #     # get first 2 dims
        #     boxe_corners_2d = boxe_corners_2d[:,:,0:2]
        #     front_img = img[..., 0]
        #     front_img = np.ascontiguousarray(front_img)
        #     front_img = self.draw_boxes(front_img, boxe_corners_2d)
        #     "use a standrad box in front to test"
        #     "5,0,-1.8,5,2,1.8"
        #     gt_bboxes_3d = LiDARInstance3DBoxes(
        #     [[2.8,0,-1.8,5,2,1.8,0],
        #     #  [5,0,-1.8,5,2,1.8,0.5*math.pi],
        #      [2.8,2,-1.8,5,2,1.8,0],
        #     #  [5,2,-1.8,5,2,1.8,0.5*math.pi],
        #      ],)
        #     boxe_corners = gt_bboxes_3d.corners
        #     dims = gt_bboxes_3d.dims.cpu().numpy()
        #     # reshape to NX8,3
        #     boxe_corners = boxe_corners.reshape(-1, 3)
        #     # repeat dims
        #     repeated_dims = np.repeat(dims, 8, axis=0)
        #     boxe_corners[:,0] = boxe_corners[:,0]#+0.5*repeated_dims[:,0]
        #     # to numpy
        #     boxe_corners = boxe_corners.cpu().numpy()
        #     # pad 1 to convert to homogenuous coordinates
        #     boxe_corners = np.concatenate([boxe_corners, np.ones((boxe_corners.shape[0],1))], axis=1)
        #     # apply transformation
        #     boxe_corners_2d = np.matmul(vel_ego_to_cam, boxe_corners.T).T
        #     boxe_corners_2d = np.matmul(lidar2img, boxe_corners_2d.T).T
        #     # filter z<0 and divide z
        #     boxe_corners_2d = boxe_corners_2d.reshape(-1, 8, 3)
        #     newboxes = []
        #     for box in boxe_corners_2d:
        #         if (box[:,2]>0).all():
        #             # divide depth
        #             box = box/box[:,2].reshape((-1,1))
        #             newboxes.append(box)
        #     boxe_corners_2d = np.asarray(newboxes)
        #     # get first 2 dims
        #     boxe_corners_2d = boxe_corners_2d[:,:,0:2]
        #     front_img = np.ascontiguousarray(front_img)
        #     front_img = self.draw_boxes(front_img, boxe_corners_2d,color=(0,255,255))
        #     # for box in boxe_corners_2d:
        #     #     front_img = cv2.circle(front_img, (int(box[0]), int(box[1])), 5, (0, 255, 0), 2)
        #     cv2.imwrite(f'camera_front_{self.saved_count}.jpg', front_img)
        #     print('written')
        self.saved_count += 1
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        # single_img_shape = img.shape[:3]  # (1080, 1920, 3)
        # results['img_shape'] = [single_img_shape] * img.shape[-1]
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str
    
    def draw_boxes(self, image, boxes, color=(0, 255, 0), thickness=2):
        if boxes is None:
            return image
        for box in boxes:
            box = box.astype(np.int32)
            for i in range(4):
                cv2.line(image, tuple(box[i]), tuple(box[(i + 1) % 4]), color, thickness)
                cv2.line(image, tuple(box[i + 4]), tuple(box[(i + 1) % 4 + 4]), color, thickness)
                cv2.line(image, tuple(box[i]), tuple(box[i + 4]), color, thickness)
        return image


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int): The max possible cat_id in input segmentation mask.
            Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points. \
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class DebugVisualizeProcessedData(object):
    """Visualize data after all pipeline transforms"""
    
    def __init__(self, save_interval=10, output_dir='debug_processed'):
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.counter = 0
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def __call__(self, results):
        """Visualize what the model will actually receive"""
        if self.counter % self.save_interval != 0:
            self.counter += 1
            return results
        
        # print(f"\n{'='*60}")
        # print(f"DEBUG: Processed data #{self.counter}")
        # print(f"{'='*60}")
        
        # Get images (they're still numpy arrays at this point)
        imgs = results['img']  # List of images
        gt_bboxes_3d = results['gt_bboxes_3d']
        lidar2img = results.get('lidar2img', [])

        if len(gt_bboxes_3d) == 0:
            print("No GT boxes in this sample, skipping visualization")
            self.counter += 1
            return results
        
        # Denormalize images for visualization
        if 'img_norm_cfg' in results:
            mean = np.array(results['img_norm_cfg']['mean'])
            std = np.array(results['img_norm_cfg']['std'])
        else:
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
        
        # Project and draw boxes
        boxe_corners = gt_bboxes_3d.corners.cpu().numpy()
        view_names = ['front', 'back', 'left', 'right']
        
        for view_idx, (img, view_name) in enumerate(zip(imgs[:4], view_names)):
            # Denormalize
            img_vis = img.copy()
            img_vis = img_vis * std + mean
            img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            
            # Get lidar2img for this view
            if view_idx < len(lidar2img):
                l2i = np.array(lidar2img[view_idx])
                # print(f"\nView {view_name} lidar2img sum: {l2i.sum():.2f}")
                
                if l2i.shape == (4, 4):
                    l2i = l2i[:3, :]
                
                # Project boxes
                boxe_corners_flat = boxe_corners.reshape(-1, 3)
                boxe_corners_homo = np.concatenate(
                    [boxe_corners_flat, np.ones((boxe_corners_flat.shape[0], 1))], 
                    axis=1
                )
                
                boxe_corners_2d = (l2i @ boxe_corners_homo.T).T
                boxe_corners_2d = boxe_corners_2d.reshape(-1, 8, 3)
                
                # Draw valid boxes
                valid_boxes = []
                for box in boxe_corners_2d:
                    if (box[:, 2] > 0).all():
                        box_2d = box[:, :2] / box[:, 2:3]
                        valid_boxes.append(box_2d)
                
                if len(valid_boxes) > 0:
                    for box in valid_boxes:
                        box = box.astype(np.int32)
                        for i in range(4):
                            cv2.line(img_vis, tuple(box[i]), tuple(box[(i + 1) % 4]), 
                                   (0, 255, 0), 2)
                            cv2.line(img_vis, tuple(box[i + 4]), tuple(box[(i + 1) % 4 + 4]), 
                                   (0, 255, 0), 2)
                            cv2.line(img_vis, tuple(box[i]), tuple(box[i + 4]), 
                                   (0, 255, 0), 2)
                    # print(f"  Drew {len(valid_boxes)} boxes on {view_name}")
                # else:
                    # print(f"  No valid boxes for {view_name}")
            
            # Save
            # output_path = f'{self.output_dir}/processed_{view_name}_{self.counter}_cuda.jpg'
            device_id = int(os.environ.get('LOCAL_RANK', -1))
            if device_id < 0 and torch.cuda.is_available():
                device_id = torch.cuda.current_device()
            device_prefix = f'gpu{device_id}_' if device_id >= 0 else ''
            output_path = f'{self.output_dir}/{device_prefix}processed_{view_name}_{self.counter}.jpg'
            cv2.imwrite(output_path, img_vis)
            # print(f"  Saved: {output_path}")
        
        self.counter += 1
        return results
    

@PIPELINES.register_module()
class FilterOccludedObjects(object):
    """
    Keep a GT 3D box iff it is visible (not fully occluded) in at least one camera.
    Uses a LiDAR-based per-view Z-buffer. Falls back to label-only if no points.
    """

    def __init__(self,
                 ds=8,                 # downsample factor for Z-buffer/image grid
                 zbuf_dilate=1,        # morphological dilation (cells) to fill holes
                 depth_margin=0.20,    # meters; treat <= margin as same layer
                 min_area_px=16,       # ignore tiny projected bboxes
                 pts_min_per_cell=1,   # min LiDAR pts per cell to trust depth
                 eps_z=1e-3):
        self.ds = int(ds)
        self.zbuf_dilate = int(zbuf_dilate)
        self.depth_margin = float(depth_margin)
        self.min_area_px = int(min_area_px)
        self.pts_min_per_cell = int(pts_min_per_cell)
        self.eps_z = float(eps_z)

    # ---------- utilities ----------

    def _proj_pts(self, xyz, l2i, H, W):
        """Project Nx3 LiDAR points -> per-pixel depth, build downsampled Z-buffer."""
        P = np.concatenate([xyz, np.ones((len(xyz),1), dtype=xyz.dtype)], axis=1)  # [N,4]
        Q = (l2i @ P.T).T                                                           # [N,3]
        in_front = Q[:,2] > self.eps_z
        if not in_front.any():
            return None

        uv = Q[in_front,:2] / Q[in_front,2:3]
        z  = Q[in_front,2]
        u = uv[:,0]; v = uv[:,1]

        # keep points that land in image
        keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not keep.any():
            return None
        u = u[keep]; v = v[keep]; z = z[keep]

        # downsample to grid
        ds = self.ds
        gw = max(1, W // ds); gh = max(1, H // ds)
        gx = np.clip((u / ds).astype(np.int32), 0, gw-1)
        gy = np.clip((v / ds).astype(np.int32), 0, gh-1)

        # per-cell min depth + counts
        zbuf = np.full((gh, gw), np.inf, dtype=np.float32)
        cnt  = np.zeros((gh, gw), dtype=np.int32)
        for x, y, zz in zip(gx, gy, z):
            if zz < zbuf[y, x]:
                zbuf[y, x] = zz
            cnt[y, x] += 1

        # trust only cells with enough points
        zbuf[cnt < self.pts_min_per_cell] = np.inf

        # optional dilation to fill small holes
        if self.zbuf_dilate > 0:
            from scipy.ndimage import grey_erosion
            # use erosion on depth to propagate nearer depths (min)
            zbuf = grey_erosion(zbuf, size=(1 + 2*self.zbuf_dilate,
                                            1 + 2*self.zbuf_dilate), mode='nearest')
        return zbuf

    def _project_boxes_rects_depth(self, corners, centers, l2i, H, W):
        """Project corners/centers; return valid_mask, rects [x0,y0,x1,y1], depth per box."""
        N = len(corners)
        c_h = np.concatenate([corners.reshape(-1,3), np.ones((N*8,1), dtype=np.float32)], axis=1)
        P   = (l2i @ c_h.T).T.reshape(N, 8, 3)     # [N,8,3]
        z   = P[...,2]
        in_front = z > self.eps_z                  # [N,8]
        ok = in_front.any(axis=1)                  # [N]

        rects = np.zeros((N,4), dtype=np.int32)
        depth = np.full((N,), np.inf, dtype=np.float32)

        if not ok.any():
            return ok, rects, depth

        # Only process boxes that have any corner in front (avoid NaN warnings)
        idxs = np.where(ok)[0]
        uv = np.empty((len(idxs),8,2), dtype=np.float32)
        uv[:] = np.nan
        z_sel = z[idxs]
        valid = in_front[idxs]

        uv[valid] = (P[idxs][valid, :2] / z_sel[valid, None])

        # bbox per selected box
        x0 = np.nanmin(uv[...,0], axis=1)
        y0 = np.nanmin(uv[...,1], axis=1)
        x1 = np.nanmax(uv[...,0], axis=1)
        y1 = np.nanmax(uv[...,1], axis=1)

        # clip to image
        x0 = np.clip(np.floor(x0), 0, W-1).astype(np.int32)
        y0 = np.clip(np.floor(y0), 0, H-1).astype(np.int32)
        x1 = np.clip(np.ceil (x1), 0, W-1).astype(np.int32)
        y1 = np.clip(np.ceil (y1), 0, H-1).astype(np.int32)

        rects[idxs] = np.stack([x0,y0,x1,y1], axis=1)

        # representative depth: 25th percentile among visible corners; fallback to center
        vis_z = np.where(valid, z_sel, np.nan)
        near_z = np.nanpercentile(vis_z, 25, axis=1)
        c_h = np.concatenate([centers[idxs], np.ones((len(idxs),1), dtype=np.float32)], axis=1)
        c_proj = (l2i @ c_h.T).T
        c_depth = c_proj[:,2]
        depth[idxs] = np.where(np.isfinite(near_z), near_z, c_depth)
        return ok, rects, depth

    def _visible_in_view_with_zbuf(self, rects, depth, H, W, zbuf):
        """Check visibility vs LiDAR Z-buffer on a ds grid."""
        ds = self.ds
        if zbuf is None:
            # no trusted depth; conservatively keep (visible) boxes
            return np.ones(len(depth), dtype=bool)

        gh, gw = zbuf.shape
        vis = np.zeros(len(depth), dtype=bool)

        for i, (x0,y0,x1,y1) in enumerate(rects):
            if x1 <= x0 or y1 <= y0:
                continue
            if (x1-x0)*(y1-y0) < self.min_area_px:
                continue

            gx0 = x0 // ds; gy0 = y0 // ds
            gx1 = min((x1 + ds - 1)//ds, gw-1)
            gy1 = min((y1 + ds - 1)//ds, gh-1)
            if gx1 < gx0 or gy1 < gy0:
                continue

            sub = zbuf[gy0:gy1+1, gx0:gx1+1]
            # free if LiDAR depth is farther than (box_depth - margin) OR no depth (inf)
            free = (sub == np.inf) | (sub > (depth[i] - self.depth_margin))
            if free.any():
                vis[i] = True
            # else fully covered by nearer LiDAR returns -> occluded
        return vis

    # ---------- main ----------

    def __call__(self, results):
        gt_bboxes_3d = results.get('gt_bboxes_3d', None)
        gt_labels_3d = results.get('gt_labels_3d', None)
        l2is = results.get('lidar2img', None)
        imgs = results.get('img', None)

        if gt_bboxes_3d is None or len(gt_bboxes_3d) == 0 or l2is is None or len(l2is)==0:
            return results

        N = len(gt_bboxes_3d)
        corners = gt_bboxes_3d.corners.cpu().numpy().astype(np.float32)   # [N,8,3]
        centers = gt_bboxes_3d.gravity_center.cpu().numpy().astype(np.float32)

        # per-view sizes
        if isinstance(imgs, list):
            sizes = [im.shape[:2] for im in imgs]
        else:
            sizes = [imgs.shape[:2]] * len(l2is)

        # LiDAR points (optional but recommended)
        pts = results.get('points', None)
        xyz = None
        if pts is not None:
            # BasePoints or ndarray
            xyz = pts.tensor[:, :3].cpu().numpy().astype(np.float32) if hasattr(pts, "tensor") else pts[:, :3]

        visible_any = np.zeros(N, dtype=bool)

        for v, (l2i_full, (H, W)) in enumerate(zip(l2is, sizes)):
            l2i = np.asarray(l2i_full)
            if l2i.shape == (4,4):
                l2i = l2i[:3,:]

            # 1) Z-buffer from LiDAR
            zbuf = None
            if xyz is not None and len(xyz) > 0:
                zbuf = self._proj_pts(xyz, l2i, H, W)

            # 2) Project boxes (skip all-NaN safely)
            ok, rects, depth = self._project_boxes_rects_depth(corners, centers, l2i, H, W)
            if not ok.any():
                continue

            # 3) Visibility vs Z-buffer
            vis = self._visible_in_view_with_zbuf(rects, depth, H, W, zbuf)

            visible_any |= vis

        mask = visible_any
        results['gt_bboxes_3d'] = gt_bboxes_3d[mask]
        results['gt_labels_3d'] = gt_labels_3d[mask]
        return results
