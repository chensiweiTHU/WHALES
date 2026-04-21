from collections import defaultdict

from mmdet.datasets.builder import PIPELINES


def _ewma(values, decay):
    """Recency-weighted mean of ``values``.

    The last element gets weight 1, the one before it ``decay``, and so on
    backwards. ``decay == 1.0`` collapses to a plain arithmetic mean.
    """
    n = len(values)
    if n == 0:
        return 0.0
    if decay == 1.0:
        return float(sum(values)) / n
    # age[i] = (n - 1 - i), so weight[i] = decay ** (n - 1 - i).
    w = [decay ** (n - 1 - i) for i in range(n)]
    total_w = sum(w)
    return sum(v * wi for v, wi in zip(values, w)) / total_w
from .loading import LoadPointsFromFile
import numpy as np
from mmdet3d.core.points import BasePoints, get_points_type
from pyquaternion import Quaternion
import torch
import copy
@PIPELINES.register_module()
class LoadPointsFromCooperativeAgents(LoadPointsFromFile):
    """Load Points From File.
    Load sunrgbd and scannet points from file.

    Args: same as LoadPointsFromFile
    """
    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        super().__init__(coord_type, load_dim, use_dim, shift_height, use_color, file_client_args)
    def call(self, results):
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
    def __call__(self, results:dict):
        """Call LoadPointsFromCooperativeAgents function to load points data from cooperative agents.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        if 'cooperative_agents' in results.keys():
            cooperative_results = results['cooperative_agents']
            for veh_token in cooperative_results.keys():
                veh_results = cooperative_results[veh_token]
                veh_results = self.call(veh_results)
                cooperative_results[veh_token] = veh_results
            results['cooperative_agents'] = cooperative_results
        return results

@PIPELINES.register_module()
class RawlevelPointCloudFusion(object):
    def __init__(self, fusion_dims=[]):
        """the fusion_dims contains the potential fusion dims,
        we will call the functions and add more dims to points
        """
        self.fusion_dims = fusion_dims

    def __call__(self, results:dict):
        if 'cooperative_agents' in results.keys():
            cooperative_results = results['cooperative_agents']
            points = results['points']
            points_class = type(points)
            points = points.tensor.numpy()
            attribute_dims = None
            cp_points_list = [points]
            transmitted_data_size = 0
            for veh_token in cooperative_results.keys():
                assert veh_token != results['veh_token']
                cp_points = copy.deepcopy(cooperative_results[veh_token]['points'])
                # convert the points to the same coordinate system:
                # aux veh -> world -> ego veh
                turn_matrix = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
                r_matrix = Quaternion(cooperative_results[veh_token]['ego2global_rotation']).rotation_matrix
                t_vector = cooperative_results[veh_token]['ego2global_translation']
                matrix_c = np.eye(4)
                matrix_c[:3, :3] = r_matrix
                matrix_c[:3, 3] = t_vector

                r_matrix = Quaternion(results['ego2global_rotation']).rotation_matrix
                t_vector = results['ego2global_translation']
                matrix_e = np.eye(4)
                matrix_e[:3, :3] = r_matrix
                matrix_e[:3, 3] = t_vector

                cp_points = cp_points.tensor.numpy().T
                if 'cooperative_datasize_limit'  in results.keys():
                    data_size_limit = results['cooperative_datasize_limit'][veh_token]
                else:
                    data_size_limit = np.inf
                if cp_points.nbytes > data_size_limit:
                    point_num = int(data_size_limit / (cp_points.itemsize*cp_points.shape[0]))
                    # random sample the points
                    idx = np.random.choice(cp_points.shape[1], point_num, replace=False)
                    cp_points = cp_points[:, idx]
                    assert cp_points.nbytes <= data_size_limit
                transmitted_data_size += cp_points.nbytes
                cp_points = turn_matrix@np.linalg.inv(matrix_e) @ matrix_c @turn_matrix@ cp_points
                cp_points = cp_points.T
                cp_points_list.append(cp_points)
            points = np.concatenate(cp_points_list, axis=0)
            points = points_class(
                points, points_dim=points.shape[-1],
                attribute_dims=attribute_dims)

            results['points'] = points
            results['transmitted_data_size'] = transmitted_data_size
        return results
    
@PIPELINES.register_module()
class AgentScheduling(object):
    def __init__(self,
                 mode="full_communication",
                 submode="",
                 basic_data_limit=2e6,
                 timesteps=1000,
                 window_size=10,
                 scheduling_range=[-50, -50, -5, 50, 50, 3],
                 ucb_c=0.5,
                 mass_beta=1.0,
                 cmass_alpha=1.0,
                 cmass_beta=0.5,
                 cmass_topology_radius=2.0,
                 cahs_num_agents=1,
                 cahs_score_weight=0.1,
                 cahs_history_window=10,
                 cahs_decay=0.7):
        """
        Args:
            mode (str): top-level communication model. One of
                'full_communication', 'groupcast', 'unicast', 'cahs',
                'ucb' (or 'UCB'), 'mass', 'mass-modified', 'c-mass'.
            submode (str): algorithm sub-selection within a ``mode``.
                For ``unicast``: 'closest_first' or 'random'.
                For ``groupcast``: 'full_random'.
            basic_data_limit (float): per-agent bandwidth cap in bytes.
            scheduling_range (list[float]): [xmin, ymin, zmin, xmax, ymax, zmax]
                used by ``withinRange`` / ``withinEgoRange``.
            ucb_c (float): UCB1 exploration constant.
            mass_beta (float): MASS UCB coefficient on sqrt(unseen_time).
            cmass_alpha (float): C-MASS topological-uncertainty weight.
            cmass_beta (float): C-MASS UCB (unexplored-time) weight.
            cmass_topology_radius (float): metres used to match box centres
                when building empirical first- and second-order topologies.
            cahs_num_agents (int): how many agents CAHS should schedule per
                frame. 1 for the single-agent variant in the WHALES paper,
                any positive integer for top-K multi-agent scheduling, or
                -1 to schedule every candidate with non-zero historical
                coverage.
            cahs_score_weight (float): tie-breaking weight applied to the
                agent's historical detection-confidence when ranking by
                coverage. ``score = coverage + w * confidence``.
            cahs_history_window (int): number of past frames to keep in the
                per-agent buffer feeding the historical-coverage estimate.
            cahs_decay (float): recency-weighting factor in (0, 1]. Observations
                from ``k`` frames ago contribute ``cahs_decay ** k``; the
                aggregate is a weighted mean. Equivalent to
                discounted-UCB / constant-step-size smoothing. Set to 1.0
                for a uniform sliding-window mean.
        """
        self.mode = mode
        self.submode = submode
        self.basic_data_limit = basic_data_limit
        self.current_timestep = 0
        self.scheduling_range = scheduling_range
        self.c = ucb_c
        self.beta = mass_beta
        self.cmass_alpha = cmass_alpha
        self.cmass_beta = cmass_beta
        self.cmass_topology_radius = cmass_topology_radius
        self.cahs_num_agents = int(cahs_num_agents)
        self.cahs_score_weight = float(cahs_score_weight)
        self.cahs_history_window = int(cahs_history_window)
        self.cahs_decay = float(cahs_decay)
        if not (0.0 < self.cahs_decay <= 1.0):
            raise ValueError(
                f'cahs_decay must be in (0, 1]; got {self.cahs_decay}')

        self.current_scene = None
        self.history_schedules = dict()        # timestep -> [agent, ...]
        self.agent_schedule_count = defaultdict(int)   # agent -> #times picked
        self.agent_schedule_last = dict()              # agent -> last timestep
        self.agent_gain_sum = defaultdict(float)       # agent -> cumulative gain

        # MASS / MASS-modified state
        self.ego_history_agent_results = {}
        self.ego_cp_history = {}
        self.history_agent_results = dict()
        self.agent_performance_history = {}

        # C-MASS state
        self.cmass_topology_first = defaultdict(dict)   # agent -> {obj_key: w}
        self.cmass_topology_second = defaultdict(dict)  # (i, j) -> {obj_key: w}

        # CAHS state
        self.cahs_coverage_history = defaultdict(list)  # agent -> [coverage, ...]
        self.cahs_score_history = defaultdict(list)     # agent -> [avg_score, ...]

    def __call__(self, results: dict):
        if self.current_scene != results['scene_token']:
            self.current_scene = results['scene_token']
            self.history_schedules = dict()
            self.agent_schedule_count = defaultdict(int)
            self.agent_schedule_last = dict()
            self.agent_gain_sum = defaultdict(float)
            self.history_agent_results = dict()
            self.agent_performance_history = {}
            self.ego_cp_history = {}
            self.cmass_topology_first = defaultdict(dict)
            self.cmass_topology_second = defaultdict(dict)
            self.cahs_coverage_history = defaultdict(list)
            self.cahs_score_history = defaultdict(list)
        self.current_timestep = int(results['frame_token'])

        mode = self.mode
        submode = self.submode

        if mode == "full_communication":
            cooperative_limit = {veh: np.inf
                                 for veh in results['cooperative_agents']}
            results['cooperative_datasize_limit'] = cooperative_limit
        elif mode == "groupcast":
            if submode == "full_random":
                # Bernoulli(0.5) per agent.
                cooperative_veh_tokens = list(results['cooperative_agents'].keys())
                choose_mask = np.random.choice(
                    [0, 1], len(cooperative_veh_tokens), p=[0.5, 0.5])
                cooperative_veh_tokens = [
                    cooperative_veh_tokens[i]
                    for i in range(len(cooperative_veh_tokens))
                    if choose_mask[i] == 1
                ]
                results['cooperative_veh_tokens'] = cooperative_veh_tokens
                cooperative_results = {
                    veh: results['cooperative_agents'][veh]
                    for veh in cooperative_veh_tokens
                }
                cooperative_datasize_limit = {
                    veh: self.basic_data_limit for veh in cooperative_veh_tokens
                }
                results['cooperative_agents'] = cooperative_results
                results['cooperative_datasize_limit'] = cooperative_datasize_limit
        elif mode == "unicast":
            # Pick exactly one cooperative agent.
            distancegraph = results['sample_info']['cav_distance_graph']
            v_id = int(results['veh_token'])
            # Work on a copy so we don't mutate the sample_info dict that
            # lives in the dataset's info cache.
            distance2ego = np.asarray(distancegraph[v_id], dtype=float).copy()
            assert distance2ego[v_id] == 0 or distance2ego[v_id] == np.inf
            distance2ego[v_id] = np.inf

            cooperative_agent = None
            if submode == "closest_first":
                if np.isfinite(distance2ego).any():
                    cooperative_agent = str(int(np.argmin(distance2ego)))
            elif submode == 'random':
                # first choose candidates within scheduling range
                max_range = max(self.scheduling_range)
                candidates = [str(i) for i in range(len(distance2ego))
                              if distance2ego[i] < max_range]
                if candidates:
                    cooperative_agent = str(np.random.choice(candidates))

            if cooperative_agent is not None:
                self._commit_selection(results, cooperative_agent)
            else:
                results['cooperative_veh_tokens'] = []
                results['cooperative_agents'] = {}
                results['cooperative_datasize_limit'] = {}
        elif mode == "cahs":
            selected = self._pick_cahs(results)
            if selected:
                self._commit_multiple(results, selected)
            else:
                results['cooperative_veh_tokens'] = []
                results['cooperative_agents'] = {}
                results['cooperative_datasize_limit'] = {}

        elif mode == "ucb" or mode == "UCB":
            cooperative_agent = self._pick_ucb(results)
            if cooperative_agent is not None:
                self._commit_selection(results, cooperative_agent)

        elif mode == "c-mass":
            cooperative_agent = self._pick_cmass(results)
            if cooperative_agent is not None:
                self._commit_selection(results, cooperative_agent)

        elif mode == "mass-modified":
            self.updateEgoCPHistory(results)
            available_agents = list(results['cooperative_agents'].keys())

            cooperative_agent = None
            best_stats = None
            max_ucb_score = -float('inf')

            def _stats_for(agent):
                """Return (last_seen_time, last_seen_gain, last_seen_score)."""
                if agent in self.agent_performance_history:
                    p = self.agent_performance_history[agent]
                    return (p['last_seen_time'], p['last_seen_gain'],
                            p['last_seen_score'])
                last = self.ego_cp_history[agent][-1]
                avg_score = (float(np.mean(last['scores']))
                             if last['scores'] else 0.0)
                return (last['timestep'],
                        last['num_vehicles_within_range'],
                        avg_score)

            # Cold-start: pick any agent we've never observed in ego_cp_history.
            for agent in available_agents:
                if agent not in self.ego_cp_history:
                    cooperative_agent = agent
                    best_stats = (self.current_timestep, 0, 0.0)
                    break
            else:
                for agent in available_agents:
                    if agent not in self.ego_cp_history:
                        continue
                    last_seen_time, last_seen_gain, last_seen_score = \
                        _stats_for(agent)
                    ucb_score = (last_seen_gain + self.beta * np.sqrt(
                        max(0, self.current_timestep - last_seen_time)))
                    if ucb_score > max_ucb_score:
                        max_ucb_score = ucb_score
                        cooperative_agent = agent
                        best_stats = (last_seen_time, last_seen_gain,
                                      last_seen_score)

            if cooperative_agent is not None:
                self._commit_selection(results, cooperative_agent)
                self.agent_performance_history[cooperative_agent] = {
                    'last_seen_time': best_stats[0],
                    'last_seen_gain': best_stats[1],
                    'last_seen_score': best_stats[2],
                }


        elif mode == "mass":
            agents_data = self.withinRange(results)
            available_agents = list(results['cooperative_agents'].keys())

            cooperative_agent = None
            max_ucb_score = -float('inf')

            # Cold-start: pull any agent we've never observed a gain from.
            for agent in available_agents:
                if agent not in self.history_agent_results:
                    cooperative_agent = agent
                    break
            else:
                for agent in available_agents:
                    stats = self.history_agent_results[agent]
                    last_seen_time = stats['last_seen_time']
                    last_seen_gain = stats['last_seen_gain']
                    ucb_score = last_seen_gain + self.beta * np.sqrt(
                        max(0, self.current_timestep - last_seen_time))
                    if ucb_score > max_ucb_score:
                        max_ucb_score = ucb_score
                        cooperative_agent = agent

            if cooperative_agent is not None:
                self._commit_selection(results, cooperative_agent)
                gain = (agents_data.get(cooperative_agent, {})
                        .get('num_vehicles_within_range', 0))
                self.history_agent_results[cooperative_agent] = {
                    'last_seen_time': self.current_timestep,
                    'last_seen_gain': gain,
                }

        self.history_schedules[self.current_timestep] = results['cooperative_veh_tokens']
        return results                

    def withinRange(self, results):
        agents_data = {}
        hist_res = results['history_results']
        for veh_id, veh_data in hist_res.items():
            if veh_id == results['veh_token']:
                continue
            turn_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            r_matrix_c = Quaternion(results['cooperative_agents'][veh_id]['ego2global_rotation']).rotation_matrix
            t_vector_c = results['cooperative_agents'][veh_id]['ego2global_translation']
            matrix_c = np.eye(4)
            matrix_c[:3, :3] = r_matrix_c
            matrix_c[:3, 3] = t_vector_c

            # Transformation matrix from vehicle coordinate to global coordinate for the ego vehicle
            r_matrix_e = Quaternion(results['ego2global_rotation']).rotation_matrix
            t_vector_e = results['ego2global_translation']
            matrix_e = np.eye(4)
            matrix_e[:3, :3] = r_matrix_e
            matrix_e[:3, 3] = t_vector_e

            agent_info = {'scores': [], 'num_vehicles_within_range': 0}
            if veh_data and 'pts_bbox' in veh_data[0] and 'boxes_3d' in veh_data[0]['pts_bbox']:
                bbox_params_list = veh_data[0]['pts_bbox']['boxes_3d']
                scores_list = veh_data[0]['pts_bbox']['scores_3d']

                vehicles_within_range = 0

                for idx, bbox_params in enumerate(bbox_params_list):

                    bbox_params = bbox_params[:7]
                    
                    dx, dy, dz = bbox_params[3:6] / 2
                    corners_local = np.array([
                        [-dx, -dy, -dz], [-dx, dy, -dz], [dx, dy, -dz], [dx, -dy, -dz],
                        [-dx, -dy, dz], [-dx, dy, dz], [dx, dy, dz], [dx, -dy, dz]
                    ])
                    
                    rx = bbox_params[6]
                    rotation_matrix = np.array([
                        [np.cos(rx), -np.sin(rx), 0],
                        [np.sin(rx), np.cos(rx), 0],
                        [0, 0, 1]
                    ])
                    
                    center = bbox_params[:3]
                    corners_global = np.dot(corners_local, rotation_matrix.T) + center.cpu().numpy()

                    corners_homogeneous = np.ones((corners_global.shape[0], 4))
                    corners_homogeneous[:, :3] = corners_global

                    corners_in_e = turn_matrix @ np.linalg.inv(matrix_e) @ matrix_c @ turn_matrix @ corners_homogeneous.T
                    corners_in_e = corners_in_e.T
                    
                    all_within_range = np.all(
                        (corners_in_e[:, 0] >= self.scheduling_range[0]) & (corners_in_e[:, 0] <= self.scheduling_range[3]) &
                        (corners_in_e[:, 1] >= self.scheduling_range[1]) & (corners_in_e[:, 1] <= self.scheduling_range[4]) &
                        (corners_in_e[:, 2] >= self.scheduling_range[2]) & (corners_in_e[:, 2] <= self.scheduling_range[5])
                    )
                    
                    if all_within_range and scores_list[idx].item() > 0.8:
                        vehicles_within_range += 1
                        agent_info['scores'].append(scores_list[idx].item())

                agent_info['num_vehicles_within_range'] = vehicles_within_range

            agents_data[veh_id] = agent_info
        return agents_data
    
    def withinEgoRange(self, results):
        """
        Determine the number of detected objects by the ego vehicle that fall within a specified range.
        This mimics withinRange but specifically focuses on detections by the ego vehicle.
        """
        ego_id = results['veh_token']
        detections = results.get('history_results', {}).get(ego_id, [])
        ego_data = {'scores': [], 'num_vehicles_within_range': 0}

        if detections:
            for detection in detections:
                if 'pts_bbox' in detection and 'boxes_3d' in detection['pts_bbox']:
                    bbox_params_list = detection['pts_bbox']['boxes_3d']
                    scores_list = detection['pts_bbox']['scores_3d']

                    for idx, bbox_params in enumerate(bbox_params_list):
                        # Apply a score threshold to filter detections
                        if scores_list[idx] > 0.8:
                            is_within_range = self.checkWithinEgoRange(bbox_params)
                            if is_within_range:
                                ego_data['num_vehicles_within_range'] += 1
                                ego_data['scores'].append(scores_list[idx])

        return ego_data

    def checkWithinEgoRange(self, bbox_params):
        dx, dy, dz = bbox_params[3:6] / 2
        corners_local = np.array([
            [-dx, -dy, -dz], [-dx, dy, -dz], [dx, dy, -dz], [dx, -dy, -dz],
            [-dx, -dy, dz], [-dx, dy, dz], [dx, dy, dz], [dx, -dy, dz]
        ])
        
        rx = bbox_params[6]
        rotation_matrix = np.array([
            [np.cos(rx), -np.sin(rx), 0],
            [np.sin(rx), np.cos(rx), 0],
            [0, 0, 1]
        ])
        
        center = bbox_params[:3]
        corners_global = np.dot(corners_local, rotation_matrix.T) + center.cpu().numpy()

        all_within_range = np.all(
            (corners_global[:, 0] >= self.scheduling_range[0]) & (corners_global[:, 0] <= self.scheduling_range[3]) &
            (corners_global[:, 1] >= self.scheduling_range[1]) & (corners_global[:, 1] <= self.scheduling_range[4]) &
            (corners_global[:, 2] >= self.scheduling_range[2]) & (corners_global[:, 2] <= self.scheduling_range[5])
        )
        if all_within_range:
            return True
        return False

    def updateEgoCPHistory(self, results):
        """Record the ego's detection result attributed to each scheduled agent.

        For every agent that was scheduled 5 frames ago (by this scheduler)
        we append the ego's current-frame detection snapshot to
        ``self.ego_cp_history[agent]`` so ``mass-modified`` can score it
        later.  The 5-frame delay is the heuristic "time for the coop
        data to land and be fused" used by the original code.
        """
        available_agents = list(results['cooperative_agents'].keys())
        ego_detections = self.withinEgoRange(results)

        lookup_step = self.current_timestep - 5
        if lookup_step < 1:
            return
        prev_scheduled = self.history_schedules.get(lookup_step, [])
        if not prev_scheduled:
            return
        for agent_id in available_agents:
            if agent_id in prev_scheduled:
                self.ego_cp_history.setdefault(agent_id, []).append({
                    'timestep': self.current_timestep,
                    'scores': ego_detections['scores'],
                    'num_vehicles_within_range':
                        ego_detections['num_vehicles_within_range'],
                })

    # ------------------------------------------------------------------
    # CAHS: Coverage-Aware Historical Scheduler (single- or multi-agent)
    # ------------------------------------------------------------------
    def _pick_cahs(self, results):
        """Pick the top-``cahs_num_agents`` cooperative agent(s) by
        exponentially-recency-weighted historical coverage of the ego's
        scheduling range.

        For each candidate agent the current-frame coverage
        (``num_vehicles_within_range``) and mean detection score are
        appended to a per-agent ring buffer of length
        ``cahs_history_window``.  The aggregate is the
        ``cahs_decay``-weighted mean: the most recent observation gets
        weight 1, the frame before it ``cahs_decay``, and so on
        backwards - i.e. a standard discounted-UCB / EMA smoother.

        Ranking key:

            score(i) = ewma(coverage_history[i])
                       + cahs_score_weight * ewma(score_history[i])

        Cold-start agents (no history yet) are ranked ahead of everyone
        with ``+inf`` so they get pulled first.
        """
        agents_data = self.withinRange(results)
        ego_token = results['veh_token']
        candidates = [a for a in results['cooperative_agents']
                      if a != ego_token]
        if not candidates:
            return []

        w = self.cahs_history_window
        decay = self.cahs_decay
        ranked = []
        for agent in candidates:
            data = agents_data.get(agent, {})
            coverage = data.get('num_vehicles_within_range', 0)
            scores = data.get('scores', [])
            avg_score = float(np.mean(scores)) if scores else 0.0

            cov_hist = self.cahs_coverage_history[agent]
            score_hist = self.cahs_score_history[agent]
            cov_hist.append(float(coverage))
            score_hist.append(avg_score)
            if w > 0:
                del cov_hist[:-w]
                del score_hist[:-w]

            if len(cov_hist) <= 1:
                key = float('inf')
            else:
                key = (_ewma(cov_hist, decay)
                       + self.cahs_score_weight * _ewma(score_hist, decay))
            ranked.append((key, agent))

        ranked.sort(key=lambda kv: kv[0], reverse=True)
        if self.cahs_num_agents < 0:
            selected = [a for k, a in ranked if k > 0 or np.isinf(k)]
            if not selected:
                selected = [ranked[0][1]]
        else:
            k = max(1, self.cahs_num_agents)
            selected = [a for _, a in ranked[:k]]
        return selected

    # ------------------------------------------------------------------
    # UCB1 scheduling
    # ------------------------------------------------------------------
    def _pick_ucb(self, results):
        """UCB1 over a per-agent perception-gain signal."""
        available_agents = list(results['cooperative_agents'].keys())
        if not available_agents:
            return None
        for agent in available_agents:
            if self.agent_schedule_count[agent] == 0:
                return agent
        agents_data = self.withinRange(results)
        total_pulls = sum(self.agent_schedule_count[a] for a in available_agents)
        total_pulls = max(total_pulls, 1)
        log_total = float(np.log(total_pulls))
        best_agent, best_score = None, -float('inf')
        for agent in available_agents:
            n_i = self.agent_schedule_count[agent]
            mean_reward = self.agent_gain_sum[agent] / max(n_i, 1)
            confidence = self.c * float(np.sqrt(2.0 * log_total / n_i))
            score = mean_reward + confidence
            if score > best_score:
                best_score = score
                best_agent = agent
        gain = agents_data.get(best_agent, {}).get(
            'num_vehicles_within_range', 0) if best_agent is not None else 0
        if best_agent is not None:
            self.agent_gain_sum[best_agent] += float(gain)
        return best_agent

    # ------------------------------------------------------------------
    # C-MASS (Jia et al., 2024)
    # ------------------------------------------------------------------
    def _pick_cmass(self, results):
        """C-MASS single-pick scheduling.

        Selects one cooperative agent maximising

            h(i|ego)/B_i  +  alpha*|U(i)|/B_i  +  beta*sqrt(t - t_last(i))/B_i

        where h(i|ego) = lambda*g+(i|ego) + (1-lambda)*g(i|ego), both
        computed on empirical first- and second-order topologies built
        from results['history_results'] (see ``_cmass_update_topology``).
        """
        available_agents = list(results['cooperative_agents'].keys())
        if not available_agents:
            return None

        self._cmass_update_topology(results)

        # Cold-start: pull any agent we've never scheduled before, per the
        # paper ("each CoV is scheduled once when it enters range").
        for agent in available_agents:
            if self.agent_schedule_count[agent] == 0:
                return agent

        ego = results['veh_token']
        ego_topology = self.cmass_topology_first.get(ego, {})

        c_max = max(1, len(available_agents))
        lam = 1.0 / (c_max + 1.0)
        best_agent, best_score = None, -float('inf')
        for agent in available_agents:
            p_first = self.cmass_topology_first.get(agent, {})
            pair_key = tuple(sorted([agent, ego]))
            p_second = self.cmass_topology_second.get(pair_key, {})

            actual = sum(
                w for k, w in p_first.items() if k not in ego_topology)
            pending = actual + 0.5 * sum(
                w for k, w in p_second.items() if k not in ego_topology)

            uncertainty = len(p_second)
            last_seen = self.agent_schedule_last.get(
                agent, self.current_timestep - 1)
            unexplored = max(0, self.current_timestep - last_seen)

            h = lam * pending + (1.0 - lam) * actual
            score = (h
                     + self.cmass_alpha * uncertainty
                     + self.cmass_beta * float(np.sqrt(unexplored)))
            if score > best_score:
                best_score = score
                best_agent = agent
        return best_agent

    def _cmass_update_topology(self, results):
        """Maintain empirical 1st- and 2nd-order topologies from history.

        For every agent we have in ``results['history_results']`` we
        collect high-confidence 3D-box centres that fall in the ego's
        scheduling range, quantise them to a grid (``cmass_topology_radius``
        metres) to serve as stable object IDs, and store them as sets
        keyed by (agent,) and (agent, ego).
        """
        ego = results['veh_token']
        hist_res = results.get('history_results', {})
        if not hist_res:
            return

        ego_boxes = self._cmass_boxes_for_agent(results, ego)
        ego_keys = {self._cmass_key(b) for b in ego_boxes}
        self.cmass_topology_first[ego] = {k: 1.0 for k in ego_keys}

        for veh_id in hist_res.keys():
            if veh_id == ego:
                continue
            agent_boxes = self._cmass_boxes_for_agent(results, veh_id)
            agent_keys = {self._cmass_key(b) for b in agent_boxes}
            self.cmass_topology_first[veh_id] = {k: 1.0 for k in agent_keys}

            # Pair topology: objects jointly visible to (agent, ego) but not
            # individually.  Approximated as set intersection minus per-agent
            # unique parts.  In a single-pair single-pick setting this is
            # simply the intersection.
            joint = agent_keys & ego_keys
            pair_key = tuple(sorted([veh_id, ego]))
            self.cmass_topology_second[pair_key] = {k: 1.0 for k in joint}

    def _cmass_boxes_for_agent(self, results, agent_id):
        """Return agent's last high-confidence boxes transformed to ego frame."""
        hist = results.get('history_results', {}).get(agent_id, [])
        if not hist:
            return []
        if 'pts_bbox' not in hist[0] or 'boxes_3d' not in hist[0]['pts_bbox']:
            return []

        if agent_id == results['veh_token']:
            matrix = np.eye(4)
        else:
            agent_info = results['cooperative_agents'].get(agent_id)
            if agent_info is None:
                return []
            turn = np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            r_c = Quaternion(agent_info['ego2global_rotation']).rotation_matrix
            t_c = agent_info['ego2global_translation']
            m_c = np.eye(4); m_c[:3, :3] = r_c; m_c[:3, 3] = t_c
            r_e = Quaternion(results['ego2global_rotation']).rotation_matrix
            t_e = results['ego2global_translation']
            m_e = np.eye(4); m_e[:3, :3] = r_e; m_e[:3, 3] = t_e
            matrix = turn @ np.linalg.inv(m_e) @ m_c @ turn

        boxes = hist[0]['pts_bbox']['boxes_3d']
        scores = hist[0]['pts_bbox']['scores_3d']
        xr0, yr0, zr0, xr1, yr1, zr1 = self.scheduling_range
        out = []
        for idx, box in enumerate(boxes):
            try:
                s = float(scores[idx])
            except Exception:
                continue
            if s <= 0.8:
                continue
            params = box[:7]
            centre = np.asarray(
                params[:3].cpu().numpy() if hasattr(params[:3], 'cpu')
                else params[:3], dtype=np.float64)
            if agent_id != results['veh_token']:
                hom = np.array([centre[0], centre[1], centre[2], 1.0])
                centre = (matrix @ hom)[:3]
            if (xr0 <= centre[0] <= xr1
                    and yr0 <= centre[1] <= yr1
                    and zr0 <= centre[2] <= zr1):
                out.append(centre)
        return out

    def _cmass_key(self, centre):
        r = float(self.cmass_topology_radius)
        return (int(round(centre[0] / r)),
                int(round(centre[1] / r)),
                int(round(centre[2] / r)))

    # ------------------------------------------------------------------
    # Shared selection commit
    # ------------------------------------------------------------------
    def _commit_selection(self, results, cooperative_agent):
        """Write a single selected agent into ``results`` and update stats."""
        self._commit_multiple(results, [cooperative_agent])

    def _commit_multiple(self, results, cooperative_agents):
        """Write ``cooperative_agents`` into ``results`` and update stats."""
        full_pool = results['cooperative_agents']
        results['cooperative_veh_tokens'] = list(cooperative_agents)
        results['cooperative_agents'] = {
            agent: full_pool[agent] for agent in cooperative_agents
        }
        results['cooperative_datasize_limit'] = {
            agent: self.basic_data_limit for agent in cooperative_agents
        }
        for agent in cooperative_agents:
            self.agent_schedule_count[agent] += 1
            self.agent_schedule_last[agent] = self.current_timestep