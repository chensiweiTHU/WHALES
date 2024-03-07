# modified from nuScenes dev-kit.
# Code written by Siwei Chen, 2024

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from pyquaternion import Quaternion
# from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, \
    DetectionMetricDataList
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from nuscenes.utils.geometry_utils import points_in_box
def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get('sample', sample_token)['sample_annotation']
        bikerack_recs = [ann for ann in sample_anns if
                         ann['type'] == 'static_object.bicycle_rack']
        bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__(class_field) in ['Cyclist']:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes
def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field


class DetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""
    
    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)
        DETECTION_NAMES = ['Vehicle','Pedestrian','Cyclist',]
        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        # assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
        #     'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])
from mmdet3d.datasets import DolphinsDataset
from tools.data_converter.dolphins import Dolphins
class DolphinsEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 dolphins: Dolphins,
                 dolphins_dataset: DolphinsDataset,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.dophins = dolphins
        self.dataset = dolphins_dataset
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = self.load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = self.load_gt(self.dophins, self.eval_set, DetectionBox, verbose=verbose)
        # mean_gt_box_h = sum([self.gt_boxes.all[i].translation[2] for i in range(2000)])/2000
        # mean_pred_box_h = sum([self.pred_boxes.all[i].translation[2] for i in range(2000)])/2000
        # print('Mean gt box height: ',mean_gt_box_h)
        # print('Mean pred box height: ',mean_pred_box_h)
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens).intersection(set(self.pred_boxes.sample_tokens)), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        # centerpoint =True
        self.pred_boxes = self.add_center_dist(dolphins, self.pred_boxes)#,not centerpoint)
        self.gt_boxes = self.add_center_dist(dolphins, self.gt_boxes)
        valuate = True
        if valuate:
            P_num = 0
            In_P_num = 0
            for box in self.gt_boxes.all:
                if box.detection_name == 'Pedestrian':
                    P_num += 1
                    if box.ego_dist < 40:
                        In_P_num += 1
            print('Total {} Pedesrtians gt'.format(P_num))
            print('Total {} Pedesrtians in 40m gt'.format(In_P_num))
            P_num = 0
            In_P_num = 0
            for box in self.pred_boxes.all:
                if box.detection_name == 'Pedestrian':
                    P_num += 1
                    if box.ego_dist < 40:
                        In_P_num += 1
            print('Total {} Pedesrtians detected'.format(P_num))
            print('Total {} Pedesrtians in 40m detected'.format(In_P_num))
        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(dolphins, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(dolphins, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def add_center_dist(self, dolphins: Dolphins,
                        eval_boxes: EvalBoxes, mode="pointpillars"):
        """
        Adds the cylindrical (xy) center distance from ego vehicle to each box.
        :param nusc: The NuScenes instance.
        :param eval_boxes: A set of boxes, either GT or predictions.
        :return: eval_boxes augmented with center distances.
        """
        for sample_token in eval_boxes.sample_tokens:
            sample = dolphins.sample[dolphins._token2ind['sample'][sample_token]]
            # dolphins.get('sample', sample_token)
            # sd_record = dolphins.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            # pose_record = dolphins.get('ego_pose', sd_record['ego_pose_token'])

            for box in eval_boxes[sample_token]:
                # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
                # Note that the z component of the ego pose is 0.
                # if mode == "pointpillars":
                if mode == 'centerpoint':
                    translation = (box.translation[0] - sample['veh_locations'][sample['veh_id']][0]-4,
                                    box.translation[1] - sample['veh_locations'][sample['veh_id']][1],
                                    box.translation[2] - sample['veh_locations'][sample['veh_id']][2]+1.8)
                    if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                        box.translation = translation
                    else:
                        raise NotImplementedError
                if sample['veh_id']<sample['vehicle_num']-1:
                    ego_translation = (box.translation[0] - sample['veh_locations'][sample['veh_id']][0],
                                    box.translation[1] - sample['veh_locations'][sample['veh_id']][1],
                                    box.translation[2] - sample['veh_locations'][sample['veh_id']][2]+1.8)
                else:
                    ego_translation = (box.translation[0] - sample['veh_locations'][sample['veh_id']][0],
                                    box.translation[1] - sample['veh_locations'][sample['veh_id']][1],
                                    box.translation[2] - sample['veh_locations'][sample['veh_id']][2])
                # elif mode=="centerpoint":
                #     ego_translation = (box.translation[0] - sample['veh_locations'][sample['veh_id']][0]-4,
                #                     box.translation[1] - sample['veh_locations'][sample['veh_id']][1],
                #                     box.translation[2] - sample['veh_locations'][sample['veh_id']][2]+3.6)
                if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                    box.ego_translation = ego_translation
                else:
                    raise NotImplementedError
        return eval_boxes

    def load_prediction(self, result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
            -> Tuple[EvalBoxes, Dict]:
        """
        Loads object predictions from file.
        :param result_path: Path to the .json result file provided by the user.
        :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
        :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
        :param verbose: Whether to print messages to stdout.
        :return: The deserialized results and meta data.
        """

        # Load from file and check that the format is correct.
        with open(result_path) as f:
            data = json.load(f)
        assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                                'See https://www.nuscenes.org/object-detection for more information.'

        # Deserialize results and get meta data.
        "modify the token to str"
        #data['results']['1'][0]['sample_token']
        for frame_t in data['results'].keys():
            for box_i in range(len(data['results'][frame_t])):
                data['results'][frame_t][box_i]['sample_token'] = str(data['results'][frame_t][box_i]['sample_token'])
                for i in range(3):
                    if data['results'][frame_t][box_i]['size'][i] <= 0:
                        data['results'][frame_t][box_i]['size'][i] = -data['results'][frame_t][box_i]['size'][i]+0.01
        all_results = EvalBoxes.deserialize(data['results'], box_cls)
        meta = data['meta']
        if verbose:
            print("Loaded results from {}. Found detections for {} samples."
                .format(result_path, len(all_results.sample_tokens)))

        # Check that each sample has no more than x predicted boxes.
        for sample_token in all_results.sample_tokens:
            assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
                "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

        return all_results, meta

    def load_gt(self, dolph: Dolphins, eval_split: str, box_cls,  verbose: bool = False) -> EvalBoxes:
        """
        Loads ground truth boxes from DB.
        :param nusc: A NuScenes instance.
        :param eval_split: The evaluation split for which we load GT boxes.
        :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
        :param verbose: Whether to print messages to stdout.
        :return: The GT boxes.
        """
        # Init.
        # if box_cls == DetectionBox:
        #     attribute_map = {a['token']: a['name'] for a in nusc.attribute}

        if verbose:
            print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, dolph.version))
        # Read out all sample_tokens in DB.
        # sample_tokens_all = [s['token'] for s in nusc.sample]
        # assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

        # Only keep samples from this split.
        from nuscenes.eval.detection.utils import category_to_detection_name
        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()

        # Check compatibility of split with nusc_version.
        # version = nusc.version
        # if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        #     assert version.endswith('trainval'), \
        #         'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
        # elif eval_split in {'mini_train', 'mini_val'}:
        #     assert version.endswith('mini'), \
        #         'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
        # elif eval_split == 'test':
        #     assert version.endswith('test'), \
        #         'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
        # else:
        #     raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
        #                     .format(eval_split))

        if eval_split == 'test':
            # Check that you aren't trying to cheat :).
            assert len(dolph.sample_annotation) > 0, \
                'Error: You are trying to evaluate on the test set but you do not have the annotations!'

        # sample_tokens = []
        # for sample_token in sample_tokens_all:
        #     scene_token = nusc.get('sample', sample_token)['scene_token']
        #     scene_record = nusc.get('scene', scene_token)
        #     if scene_record['name'] in splits[eval_split]:
        #         sample_tokens.append(sample_token)

        all_annotations = EvalBoxes()
        len(self.dataset )

        # Load annotations and filter predictions and annotations.
        tracking_id_set = set()
        sample_tokens_all = []
        for scene in dolph.scenes:
            steps = len(dolph.frames[scene])
            interval = dolph.config[scene]['world']['save_interval']
            time_interval = 0.1 * interval
            vehicle_num = dolph.scenen[scene]['vehicle_num']
            for step in range(1,steps+1):
                for v_id in range(vehicle_num+1):
                    sample_token = f'{scene}_{step*interval}_{v_id}'
                    sample_tokens_all.append(sample_token)
                    sample_boxes = []
                    for sample_annotation in dolph.frames[scene][f'{scene}_{step*interval}']['sample_annotation']:
                        # boxes = list(self.get_box(curr_anno)) 
                        if box_cls == DetectionBox:
                            # Get label name in detection task and filter unused labels.
                            detection_name = sample_annotation['type']
                            if detection_name is None:
                                continue
                            velocity = np.array(
                                    dolph.box_velocity(scene, interval, step, sample_annotation, time_interval)[:2]) 
                            sample_boxes.append(
                                box_cls(
                                    sample_token=sample_token,
                                    translation=sample_annotation['location'],
                                    size=sample_annotation['size'],
                                    rotation=sample_annotation['rotation'],
                                    velocity=velocity,
                                    num_pts=1000,
                                    detection_name=detection_name,
                                    detection_score=-1.0,  # GT samples do not have a score.
                                    attribute_name=''
                                )
                            )
                        elif box_cls == TrackingBox:
                            # Use nuScenes token as tracking id.
                            tracking_id = sample_annotation['instance_token']
                            tracking_id_set.add(tracking_id)

                            # Get label name in detection task and filter unused labels.
                            # Import locally to avoid errors when motmetrics package is not installed.
                            from nuscenes.eval.tracking.utils import category_to_tracking_name
                            tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                            if tracking_name is None:
                                continue

                            sample_boxes.append(
                                box_cls(
                                    sample_token=sample_token,
                                    translation=sample_annotation['translation'],
                                    size=sample_annotation['size'],
                                    rotation=sample_annotation['rotation'],
                                    velocity=dolph.box_velocity(sample_annotation['token'])[:2],
                                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                                    tracking_id=tracking_id,
                                    tracking_name=tracking_name,
                                    tracking_score=-1.0  # GT samples do not have a score.
                                )
                            )
                        else:
                            raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)
                    all_annotations.add_boxes(sample_token, sample_boxes)

        if verbose:
            print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))
        self.sample_tokens_all = sample_tokens_all
        return all_annotations

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            print('Accumulating metric data for class: %s' % class_name)
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th,True)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.dophins,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary




if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
