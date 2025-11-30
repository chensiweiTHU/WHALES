import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize, Normalize, Pad

@PIPELINES.register_module()
class ResizeMultiViewImage(Resize):

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 
                 **kwargs):
        super(ResizeMultiViewImage, self).__init__(
            img_scale=img_scale, ratio_range=ratio_range, **kwargs)

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img_list = []
            for key_im in results['img']:
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        key_im,
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                    new_h, new_w = img.shape[:2]
                    h, w = key_im.shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        key_im,
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                img_list.append(img)
            
            results['img'] = img_list
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio
            
            # FIX: Update lidar2img matrices
            # if 'lidar2img' in results:
            #     scale_matrix = np.eye(4, dtype=np.float32)
            #     scale_matrix[0, 0] = w_scale  # Scale fx and cx
            #     scale_matrix[1, 1] = h_scale  # Scale fy and cy
            #     results['lidar2img'] = [scale_matrix @ l2i for l2i in results['lidar2img']]
            if 'lidar2img' in results:
                updated_lidar2img = []
                for l2i in results['lidar2img']:
                    l2i = np.array(l2i, dtype=np.float32)
                    
                    # Create a copy to avoid modifying original
                    l2i_scaled = l2i.copy()
                    
                    # Scale the intrinsic part (first 2 rows affect image coordinates)
                    l2i_scaled[0, :] *= w_scale  # Scale x: fx, cx, and x-component of translation
                    l2i_scaled[1, :] *= h_scale  # Scale y: fy, cy, and y-component of translation
                    
                    updated_lidar2img.append(l2i_scaled)

    def __call__(self, results):

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        # print(len(results['img']), [im.shape for im in results['img']], 'call', isinstance(results['img'], list))
        return results




@PIPELINES.register_module()
class PadMultiViewImage(Pad):
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        super(PadMultiViewImage, self).__init__(
            size=size, size_divisor=size_divisor, pad_val=pad_val)

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_left = 0
        pad_top = 0
        
        if self.size is not None:
            padded_img = []
            for img in results['img']:
                # Calculate padding
                h, w = img.shape[:2]
                pad_top = (self.size[0] - h) // 2 if self.size[0] > h else 0
                pad_left = (self.size[1] - w) // 2 if self.size[1] > w else 0
                padded = mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                padded_img.append(padded)
        elif self.size_divisor is not None:
            padded_img = []
            for img in results['img']:
                h, w = img.shape[:2]
                padded = mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val)
                # Usually pads bottom and right, so pad_top=0, pad_left=0
                padded_img.append(padded)
        
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        
        # FIX: Update lidar2img if padding shifts principal point
        if 'lidar2img' in results and (pad_left != 0 or pad_top != 0):
            offset_matrix = np.eye(4, dtype=np.float32)
            offset_matrix[0, 2] = pad_left   # Shift cx
            offset_matrix[1, 2] = pad_top    # Shift cy
            results['lidar2img'] = [offset_matrix @ l2i for l2i in results['lidar2img']]


@PIPELINES.register_module()
class NormalizeMultiViewImage(Normalize):
    def __init__(self, mean, std, to_rgb=True):
        super(NormalizeMultiViewImage, self).__init__(
            mean=mean, std=std, to_rgb=to_rgb)
    def __call__(self, results):

        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class CropMultiViewImage(object):
    """Crop the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, size=None):
        self.size = size

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        results['img'] = [img[:self.size[0], :self.size[1], ...] for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img_fixed_size'] = self.size
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        return repr_str

@PIPELINES.register_module()
class CenterCropMultiViewImage(object):
    """Crop the image
    Args:
        size (tuple, optional): Fixed crop size.
    """

    def __init__(self, size=None):
        self.size = size

    def __call__(self, results):
        """Call function to crop images."""
        crop_offsets = []
        
        for i, img in enumerate(results['img']):
            height, width = img.shape[:2]
            crop_height, crop_width = self.size

            # Calculate the center crop coordinates
            start_x = max((width - crop_width) // 2, 0)
            start_y = max((height - crop_height) // 2, 0)
            
            crop_offsets.append((start_x, start_y))

            # Crop the image centered
            results['img'][i] = img[start_y:start_y + crop_height, 
                                start_x:start_x + crop_width, ...]

        results['img_shape'] = [img.shape for img in results['img']]
        results['img_fixed_size'] = self.size
        
        # FIX: Update lidar2img matrices (crop shifts principal point)
        if 'lidar2img' in results:
            updated_lidar2img = []
            for (start_x, start_y), l2i in zip(crop_offsets, results['lidar2img']):
                offset_matrix = np.eye(4, dtype=np.float32)
                offset_matrix[0, 2] = -start_x  # Shift cx left
                offset_matrix[1, 2] = -start_y  # Shift cy up
                updated_lidar2img.append(offset_matrix @ l2i)
            results['lidar2img'] = updated_lidar2img
            print(f"DEBUG CenterCropMultiViewImage: Applied crop offset {crop_offsets[0]}")
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size})'
        return repr_str

@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        np.random.shuffle(self.scales)
        rand_scale = self.scales[0]
        img_shape = results['img_shape']
        y_size = int(img_shape[0] * rand_scale)
        x_size = int(img_shape[1] * rand_scale) 
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size, y_size), return_scale=False) for img in results['img']]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        # print(results['img_shape'])
        results['gt_bboxes_3d'].tensor[:, :6] *= rand_scale
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class HorizontalRandomFlipMultiViewImage(object):

    def __init__(self, flip_ratio=0.5, flip_bev=False):
        self.flip_ratio = 0.5
        self.flip_bev = flip_bev

    def __call__(self, results):
        if np.random.rand() >= self.flip_ratio:
            return results
        results = self.flip_bbox(results)
        if self.flip_bev:
            results = self.flip_bev_cam_params(results)
        else:
            results = self.flip_cam_params(results)
        results = self.flip_img(results)
        return results

    def flip_img(self, results, direction='horizontal'):
        results['img'] = [mmcv.imflip(img, direction) for img in results['img']]
        return results

    def flip_cam_params(self, results):
        flip_factor = np.eye(4)
        flip_factor[1, 1] = -1
        lidar2cam = [l2c @ flip_factor for l2c in results['lidar2cam']]
        w = results['img_shape'][0][1]
        lidar2img = []
        for cam_intrinsic, l2c in zip(results['cam_intrinsic'], lidar2cam):
            cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
            lidar2img.append(cam_intrinsic @ l2c)
        results['lidar2cam'] = lidar2cam
        results['lidar2img'] = lidar2img
        return results

    def flip_bev_cam_params(self, results):
        flip_factor = np.eye(4)
        flip_factor[1, 1] = -1
        lidar2img = [l2i @ flip_factor for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        return results

    # def flip_bev_cam_params(self, results):
    #     """Flip in BEV (LiDAR space) and update image coordinates"""
        
    #     # Get image width
    #     w = results['img_shape'][0][1]
        
    #     # BEV flip in LiDAR space (flip y-axis)
    #     flip_factor = np.eye(4)
    #     flip_factor[1, 1] = -1
        
    #     updated_lidar2img = []
    #     for l2i in results['lidar2img']:
    #         l2i = np.array(l2i)
            
    #         # Apply BEV flip in 3D space
    #         if l2i.shape == (4, 4):
    #             l2i_flipped = l2i[:3, :] @ flip_factor
    #         else:
    #             l2i_flipped = l2i @ flip_factor
            
    #         # Apply image flip (update principal point)
    #         # Create a copy to avoid modifying original
    #         l2i_copy = l2i_flipped.copy()
            
    #         # Flip x-coordinates: x_new = w - x_old
    #         # This affects the first row of the matrix
    #         l2i_copy[0, :] = -l2i_copy[0, :]  # Flip x
    #         l2i_copy[0, 2] = w + l2i_copy[0, 2]  # Offset by width (corrects cx)
            
    #         if l2i.shape == (4, 4):
    #             # Reconstruct 4x4
    #             l2i_new = np.eye(4, dtype=np.float32)
    #             l2i_new[:3, :] = l2i_copy
    #             updated_lidar2img.append(l2i_new)
    #         else:
    #             updated_lidar2img.append(l2i_copy)
        
    #     results['lidar2img'] = updated_lidar2img
    #     return results

    def flip_bbox(self, input_dict, direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
        return input_dict