# Copyright (c) OpenMMLab. All rights reserved.
"""Monocular 3D detection dataset for WHALES.

Loads mono3D COCO JSON files produced by
``tools/data_converter/whales_converter.export_2d_annotation``.  Each COCO
``image`` entry corresponds to one (scene, frame, agent, camera) tuple and
each annotation's ``bbox_cam3d`` is already in the specific camera's OpenCV
frame (after the cam_yaw-aware rewrite in the converter).

The heavy lifting (``_parse_ann_info`` / ``pre_pipeline`` / ``show`` /
``format_results``) is inherited from :class:`NuScenesMonoDataset`; we
override just the class list, the 4-camera grouping in ``_format_bbox`` and
the WHALES-specific evaluation backend.
"""

import tempfile
from os import path as osp

import mmcv
import numpy as np
import pyquaternion

from mmdet.datasets import DATASETS

from .nuscenes_mono_dataset import (NuScenesMonoDataset, cam_nusc_box_to_global,
                                    global_nusc_box_to_cam,
                                    nusc_box_to_cam_box3d, output_to_nusc_box)


@DATASETS.register_module()
class WhalesMonoDataset(NuScenesMonoDataset):
    """Monocular 3D detection on the WHALES dataset.

    Args:
        class_range (dict, optional): per-class evaluation range in metres
            (same convention as :class:`WhalesDataset`). Defaults to
            ``{'Vehicle': 50, 'Pedestrian': 40, 'Cyclist': 40}``.
        num_cameras_per_frame (int): how many camera views make up a single
            agent-frame (ego and RSU both produce 4 views in WHALES).
        ``**kwargs`` are forwarded to :class:`NuScenesMonoDataset`.
    """

    CLASSES = ('Vehicle', 'Pedestrian', 'Cyclist')
    # WHALES doesn't track nuScenes-style attributes (moving / parked / …);
    # we just round-trip the class name.
    DefaultAttribute = {
        'Vehicle': 'Vehicle',
        'Pedestrian': 'Pedestrian',
        'Cyclist': 'Cyclist',
    }

    def __init__(self,
                 class_range=None,
                 num_cameras_per_frame=4,
                 with_velocity=False,
                 **kwargs):
        super().__init__(with_velocity=with_velocity, **kwargs)
        self.num_cameras_per_frame = int(num_cameras_per_frame)
        self.class_range = class_range or {
            'Vehicle': 50, 'Pedestrian': 40, 'Cyclist': 40,
        }
        if self.eval_detection_configs is not None:
            self.eval_detection_configs.class_range = self.class_range
            self.eval_detection_configs.class_names = list(
                self.class_range.keys())
        # WHALES has no velocity labels; emit an 8-dim camera box
        # (cx, cy, cz, l, h, w, ry, vx=0, vz=0).
        self.bbox_code_size = 9

    def get_attr_name(self, attr_idx, label_name):
        """WHALES has no nuScenes-style attributes; echo the class name."""
        return label_name

    # ------------------------------------------------------------------
    # Format detection results
    # ------------------------------------------------------------------
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Merge per-camera predictions into one dict keyed by sample token.

        Mirrors :meth:`NuScenesMonoDataset._format_bbox` but uses
        ``self.num_cameras_per_frame`` instead of the hard-coded 6, and
        skips the nuScenes attribute plumbing since WHALES has none.
        """
        from mmcv import Config
        from mmdet3d.core import (bbox3d2result, box3d_multiclass_nms,
                                  xywhr2xyxyr)
        from ..core.bbox import CameraInstance3DBoxes

        cam_num = self.num_cameras_per_frame
        mapped_class_names = self.CLASSES
        nusc_annos = {}
        boxes_per_frame = []
        attrs_per_frame = []

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            if sample_id % cam_num == 0:
                boxes_per_frame = []
                attrs_per_frame = []

            boxes, attrs = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes, attrs = cam_nusc_box_to_global(
                self.data_infos[sample_id], boxes, attrs,
                mapped_class_names, self.eval_detection_configs,
                self.eval_version)
            boxes_per_frame.extend(boxes)
            attrs_per_frame.extend(attrs)

            if (sample_id + 1) % cam_num != 0:
                continue

            # End of a frame: NMS across all cameras of the same agent-frame.
            frame_anchor_id = sample_id + 1 - cam_num
            boxes = global_nusc_box_to_cam(
                self.data_infos[frame_anchor_id], boxes_per_frame,
                mapped_class_names, self.eval_detection_configs,
                self.eval_version)
            cam_boxes3d, scores, labels = nusc_box_to_cam_box3d(boxes)
            nms_cfg = Config(dict(
                use_rotate_nms=True, nms_across_levels=False,
                nms_pre=4096, nms_thr=0.05, score_thr=0.01,
                min_bbox_size=0, max_per_frame=500))
            cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
            attrs_tensor = labels.new_tensor(list(attrs_per_frame))
            boxes3d, scores, labels, attrs_tensor = box3d_multiclass_nms(
                cam_boxes3d.tensor, cam_boxes3d_for_nms, scores,
                nms_cfg.score_thr, nms_cfg.max_per_frame, nms_cfg,
                mlvl_attr_scores=attrs_tensor)
            cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
            det = bbox3d2result(cam_boxes3d, scores, labels, attrs_tensor)
            boxes, attrs = output_to_nusc_box(det)
            boxes, attrs = cam_nusc_box_to_global(
                self.data_infos[frame_anchor_id], boxes, attrs,
                mapped_class_names, self.eval_detection_configs,
                self.eval_version)

            annos = []
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                attr = self.get_attr_name(attrs[i], name)
                annos.append(dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    # WHALES has no velocity labels; zero it out.
                    velocity=[0.0, 0.0],
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                ))
            nusc_annos.setdefault(sample_token, []).extend(annos)

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump({'meta': self.modality, 'results': nusc_annos}, res_path)
        return res_path

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='img_bbox'):
        """Evaluate a single result file via the shared WhalesEval backend."""
        from mmdet3d_plugin.datasets.whales_eval import WhalesEval
        from tools.data_converter.whales import Whales

        output_dir = osp.join(*osp.split(result_path)[:-1])
        whales = Whales(version=self.version, dataroot=self.data_root,
                        verbose=False, train_test='test')
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            '': 'val',
            'none': 'val',
            'v1.0': 'val',
        }
        evaluator = WhalesEval(
            whales=whales,
            whales_dataset=self,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map.get(self.version, 'val'),
            output_dir=output_dir,
            verbose=False)
        evaluator.main(render_curves=False)

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = {}
        metric_prefix = f'{result_name}_Whales'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = round(v, 4)
            for k, v in metrics['label_tp_errors'][name].items():
                detail[f'{metric_prefix}/{name}_{k}'] = round(v, 4)
            for k, v in metrics['tp_errors'].items():
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = round(v, 4)
        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Write predictions to a nuScenes-style JSON."""
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'Length mismatch: {len(results)} results vs {len(self)} samples')

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        keys = list(results[0].keys()) if results else []
        if 'img_bbox' not in keys and 'pts_bbox' not in keys:
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = {
                name: self._format_bbox(
                    [out[name] for out in results],
                    osp.join(jsonfile_prefix, name))
                for name in keys
            }
        return result_files, tmp_dir

    # ------------------------------------------------------------------
    # evaluate()
    # ------------------------------------------------------------------
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=('img_bbox',),
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Run the WHALES evaluator on predictions."""
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = {}
            for name in result_names:
                if name not in result_files:
                    continue
                ret = self._evaluate_single(result_files[name],
                                            result_name=name)
                results_dict.update(ret)
        else:
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict
