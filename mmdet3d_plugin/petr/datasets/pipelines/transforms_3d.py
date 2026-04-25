import mmcv
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from PIL import Image

from mmdet3d.core.bbox import LiDARInstance3DBoxes


@PIPELINES.register_module()
class PadMultiViewImagePETR(object):
    """Pad multi-view images to a uniform fixed size and reflect that in
    ``img_shape``/``pad_shape``.

    Unlike ``PadMultiViewImage`` (which keeps the per-camera shape distinct so
    detector heads can mask invalid borders), PETR's head requires a single
    ``pad_shape`` and a single ``img_shape`` shared across the batch, which is
    what this pad emits.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        assert size is not None or size_divisor is not None
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        imgs = results['img']
        if self.size is not None:
            target_h, target_w = self.size
        else:
            heights = [img.shape[0] for img in imgs]
            widths = [img.shape[1] for img in imgs]
            target_h = int(
                np.ceil(max(heights) / self.size_divisor) * self.size_divisor)
            target_w = int(
                np.ceil(max(widths) / self.size_divisor) * self.size_divisor)

        padded_imgs = []
        for img in imgs:
            padded = mmcv.impad(
                img, shape=(target_h, target_w), pad_val=self.pad_val)
            padded_imgs.append(padded)

        results['img'] = padded_imgs
        # PETR head expects a single (h, w, c) tuple replicated across views.
        common_shape = padded_imgs[0].shape
        results['img_shape'] = [common_shape for _ in padded_imgs]
        results['pad_shape'] = [common_shape for _ in padded_imgs]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(size={self.size}, '
                f'size_divisor={self.size_divisor}, pad_val={self.pad_val})')


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    """Random resize / crop / flip / rotate per-image augmentation.

    Mirrors PETR's official ``ResizeCropFlipImage`` but works against the
    legacy ``lidar2img`` field used by WHALES instead of ``cam2img`` /
    ``lidar2cam``.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def __call__(self, results):
        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        new_lidar2img = []
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            ida_mat_4 = np.eye(4, dtype=np.float32)
            ida_mat_4[:3, :3] = ida_mat.numpy()
            l2i = np.array(results['lidar2img'][i], dtype=np.float32)
            new_lidar2img.append(ida_mat_4 @ l2i)

        results['img'] = new_imgs
        results['lidar2img'] = new_lidar2img
        common_shape = new_imgs[0].shape
        results['img_shape'] = common_shape
        results['pad_shape'] = common_shape
        return results

    def _get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    """Apply BEV rotation / scaling to gt boxes and the camera extrinsics."""

    def __init__(
        self,
        rot_range=(-0.3925, 0.3925),
        scale_ratio_range=(0.95, 1.05),
        translation_std=(0, 0, 0),
        reverse_angle=False,
        training=True,
    ):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        rot_angle = np.random.uniform(*self.rot_range)
        self._rotate_lidar2img(results, rot_angle)
        if self.reverse_angle:
            rot_angle *= -1
        results['gt_bboxes_3d'].rotate(np.array(rot_angle))

        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self._scale_lidar2img(results, scale_ratio)
        results['gt_bboxes_3d'].scale(scale_ratio)

        if not self.reverse_angle:
            gt = results['gt_bboxes_3d'].tensor.numpy()
            gt[:, 6] -= 2 * rot_angle
            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                gt, box_dim=gt.shape[-1])
        return results

    @staticmethod
    def _rotate_lidar2img(results, angle):
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        rot_inv = np.linalg.inv(rot_mat)
        new_l2i = []
        for l2i in results['lidar2img']:
            new_l2i.append(np.array(l2i, dtype=np.float64) @ rot_inv)
        results['lidar2img'] = new_l2i

    @staticmethod
    def _scale_lidar2img(results, scale_ratio):
        scale = np.diag([scale_ratio, scale_ratio, scale_ratio, 1.0])
        scale_inv = np.linalg.inv(scale)
        new_l2i = []
        for l2i in results['lidar2img']:
            new_l2i.append(np.array(l2i, dtype=np.float64) @ scale_inv)
        results['lidar2img'] = new_l2i
