from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets import LoadPointsFromFile
import numpy as np
from mmdet3d.core.points import BasePoints, get_points_type
from pyquaternion import Quaternion 
import torch
import pdb
import copy
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D,PointsRangeFilter,\
    GlobalRotScaleTrans,RandomFlip3D,PointShuffle
# from mmdet3d.datasets.pipelines import 
from mmcv.parallel import DataContainer as DC
from time import sleep
@PIPELINES.register_module()
class PointQuantization(object):
    def __init__(self, voxel_size, quantize_coords_range, q_delta=1.6e-5):
        self.voxel_size = np.array(voxel_size)
        self.quantize_coords_range = quantize_coords_range
        self.low_bound = np.array(quantize_coords_range[:3])
        self.q_delta = q_delta
    # 2340x2304x16 = 8.5e7 log2(8.5e7) = 26.4 4+1=5B for each point
    def __call__(self, results):
        points = results['infrastructure_points']
        points_np = points.tensor.numpy()
        points_np[:, :3] -= (self.low_bound + self.voxel_size / 2)
        points_np[:, :3] = np.around(points_np[:, :3] / self.voxel_size)
        points_np[:, :3] *= self.voxel_size
        points_np[:, :3] += (self.low_bound + self.voxel_size / 2)
        "we assume that intensity is always ≥ 0 and quantisize uniformly"
        points_np[:,3] = np.around(points_np[:,3]/self.q_delta)*self.q_delta
        points.tensor = torch.from_numpy(points_np)
        return results

@PIPELINES.register_module()
class DefaultFormatBundle3DCP(DefaultFormatBundle3D):
    def __call__(self, results):
        results = super(DefaultFormatBundle3DCP, self).__call__(results)
        if 'cooperative_agents' in results.keys():
            for key in results['cooperative_agents'].keys():
                if 'points' in results['cooperative_agents'][key].keys():
                    assert isinstance(results['cooperative_agents'][key]['points'], BasePoints)
                    results['cooperative_agents'][key]['points'] = results['cooperative_agents'][key]['points'].tensor
        return results

@PIPELINES.register_module()
class PointsRangeFilterCP(PointsRangeFilter):
    def __call__(self, results):
        results = super(PointsRangeFilterCP, self).__call__(results)
        if 'cooperative_agents' in results.keys():
            for key in results['cooperative_agents'].keys():
                if 'points' in results['cooperative_agents'][key].keys():
                    assert isinstance(results['cooperative_agents'][key]['points'], BasePoints)
                    points = results['cooperative_agents'][key]['points']
                    points_mask = points.in_range_3d(self.pcd_range)
                    clean_points = points[points_mask]
                    results['cooperative_agents'][key]['points'] = clean_points
        if 'infrastructure_points' in results.keys():
            points = results['infrastructure_points']
            points_mask = points.in_range_3d(self.pcd_range)


            clean_points = points[points_mask]
            results['infrastructure_points'] = clean_points
        return results

@PIPELINES.register_module(force=True)
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

@PIPELINES.register_module(force=True)
class ProjectCooperativePCD2ego(object):
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
            # transmitted_data_size = 0
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
                # if 'cooperative_datasize_limit'  in results.keys():
                #     data_size_limit = results['cooperative_datasize_limit'][veh_token]
                # else:
                #     data_size_limit = np.inf
                # if cp_points.nbytes > data_size_limit:
                #     point_num = int(data_size_limit / (cp_points.itemsize*cp_points.shape[0]))
                #     # random sample the points
                #     idx = np.random.choice(cp_points.shape[1], point_num, replace=False)
                #     cp_points = cp_points[:, idx]
                #     assert cp_points.nbytes <= data_size_limit
                # transmitted_data_size += cp_points.nbytes
                cp_points = turn_matrix@np.linalg.inv(matrix_e) @ matrix_c @turn_matrix@ cp_points
                cp_points = cp_points.T
                cp_points = points_class(
                    cp_points, points_dim=cp_points.shape[-1], attribute_dims=attribute_dims)
                results['cooperative_agents'][veh_token]['points'] = cp_points
            #     cp_points_list.append(cp_points)
            # points = np.concatenate(cp_points_list, axis=0)
            vis=False
            if vis:
                from mmdet3d.core.visualizer.show_result import show_result
                show_result(points, None,None,'workdirs/',results['sample_idx']+'raw-level-fusion', False, False)
                print('raw-level-fusion-visualized!!!!!!!!!!')
                import time 
                time.sleep(5)
            points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
            # results['points'] = points
            # results['transmitted_data_size'] = transmitted_data_size
        if 'infrastructure_points' in results.keys(): # DAIR-V2X
            "project inf points to vehicle coordinate system in raw data"
            infrastructure_points = results['infrastructure_points']
            points_class = type(infrastructure_points)
            infrastructure_points = infrastructure_points.tensor
            device = infrastructure_points.device

            rotatation_matrix = torch.tensor(results['inf2veh']['rotation']).to(device).float()
            translation = torch.tensor(results['inf2veh']['translation']).to(device).float()
            transfrom_matrix = torch.zeros((4,4)).to(device)
            transfrom_matrix[0:3,0:3] = rotatation_matrix
            transfrom_matrix[0:3,3] = translation.T
            transfrom_matrix[3,3] = 1
            points_inf_feat = infrastructure_points[:,3:]
            points_inf = infrastructure_points[:,:3]
            points_inf = torch.cat([points_inf,torch.ones(points_inf.shape[0],1).to(device)],dim=1)
            points_inf = transfrom_matrix@points_inf.T
            points_inf = points_inf.T
            points_inf = points_inf[:,0:3]
            infrastructure_points = torch.cat([points_inf,points_inf_feat],dim=1)
            results['infrastructure_points'] = points_class(
            infrastructure_points, points_dim=infrastructure_points.shape[-1], attribute_dims=None)
        return results

@PIPELINES.register_module(force=True)
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
            visualization = False
            if visualization:
                save_path = './visual/'
                "save raw points and fused points in .bin"
                points.tofile(save_path+results['sample_idx']+'raw.bin')
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
                if visualization:
                    cp_points.T.tofile(save_path+results['sample_idx']+veh_token+'.bin')
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
                if visualization:
                    # time.sleep(0.3)
                    sleep(0.3)
                    # print('cp_points',cp_points.shape)
                    cp_points = cp_points.astype(np.float32)
                    cp_points.tofile(save_path+results['sample_idx']+'projected'+veh_token+'.bin')
            points = np.concatenate(cp_points_list, axis=0)
            if visualization:
                points=points.astype(np.float32)
                points.tofile(save_path+results['sample_idx']+'fused.bin')
            vis=False
            if vis:
                from mmdet3d.core.visualizer.show_result import show_result
                show_result(points, None,None,'workdirs/',results['sample_idx']+'raw-level-fusion',False,False)
                print('raw-level-fusion-visualized!!!!!!!!!!')
                import time 
                time.sleep(5)
            points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

            results['points'] = points
            results['transmitted_data_size'] = transmitted_data_size
        return results
    
@PIPELINES.register_module(force=True)
class AgentScheduling(object):
    def __init__(self, mode="full_communication", submode="", basic_data_limit=2e6, timesteps=1000, window_size=10, scheduling_range = [-50, -50, -5, 50, 50, 3]):
        "the mode indciates the communication model used in the cooperative perception"
        "the submode indicates the scheduling algorithm used in the cooperative perception"
        "the basic_data_limit indicates the basic data limit for each vehicle"
        self.mode = mode
        self.submode = submode
        self.basic_data_limit = basic_data_limit
        # Discount factor for discounted UCB
        self.current_timestep = 0
        self.scheduling_range = scheduling_range
        self.UCB_scores = {}
        self.c = 0.5
        self.beta = 1
        self.history_schedules=dict()
        self.current_scene = None
        self.ego_history_agent_results = {} 
        self.ego_cp_history = {}
        self.history_agent_results = dict()
        self.agent_performance_history = {}

    def __call__(self, results: dict):
        # results[]
        if 'transmitted_data_size' not in results.keys():
            results['transmitted_data_size'] = 0
        if self.current_scene != results['scene_token']:
            self.current_scene = results['scene_token']
            self.history_schedules = dict()
        self.current_timestep = int(results['frame_token'])
        if self.mode == "full_communication":
            self.basic_data_limit = np.inf
            cooperative_limit = dict()
            cooperative_results = results['cooperative_agents']
            for veh_token in cooperative_results.keys():
                cooperative_limit[veh_token] = self.basic_data_limit
            results['cooperative_datasize_limit'] = cooperative_limit
        elif self.mode=="groupcast":
            if self.submode=="full_random":
                self.basic_data_limit = np.inf
                "evry agent has 50% chance to be chosen"
                cooperative_veh_tokens = list(results['cooperative_agents'].keys())
                choose_mask = np.random.choice([0,1], len(cooperative_veh_tokens), p=[0.5, 0.5])
                cooperative_veh_tokens = [cooperative_veh_tokens[i] for i in range(len(cooperative_veh_tokens)) \
                                          if choose_mask[i]==1]
                results['cooperative_veh_tokens'] = cooperative_veh_tokens
                cooperavtive_datasize_limit = dict()
                cooperative_results = dict()
                for veh_token in cooperative_veh_tokens:
                    cooperavtive_datasize_limit[veh_token] = self.basic_data_limit
                    cooperative_results[veh_token] = results['cooperative_agents'][veh_token]
                results['cooperative_agents'] = cooperative_results
                results['cooperative_datasize_limit'] = cooperavtive_datasize_limit
        elif self.mode == "unicast":
            "we can only choose one vehicle to communicate"
            if self.submode == "closest":
                distancegraph = results['sample_info']['cav_distance_graph']
                v_id = int(results['veh_token'])
                distance2ego = distancegraph[v_id]
                assert distance2ego[v_id] == 0 or distance2ego[v_id] == np.inf
                distance2ego[v_id] = np.inf
                cooperative_agent = str(np.argmin(distance2ego))
                results['cooperative_veh_tokens'] = [cooperative_agent]
                results['cooperative_agents'] = {
                    cooperative_agent: results['cooperative_agents'][cooperative_agent]
                }
                results['cooperative_datasize_limit'] = {
                    cooperative_agent:self.basic_data_limit
                }
            if self.submode == 'random':
                # first choose candidates within shceduling range
                distancegraph = results['sample_info']['cav_distance_graph']
                v_id = int(results['veh_token'])
                distance2ego = distancegraph[v_id]
                assert distance2ego[v_id] == 0 or distance2ego[v_id] == np.inf
                distance2ego[v_id] = np.inf
                candidates = []
                for id in range(len(distance2ego)):
                    if distance2ego[id] < max(self.scheduling_range):
                        candidates.append(str(id))
                if len(candidates) > 0:
                    cooperative_agent = np.random.choice(candidates)
                    results['cooperative_veh_tokens'] = [cooperative_agent]
                    results['cooperative_agents'] = {
                        cooperative_agent: results['cooperative_agents'][cooperative_agent]
                    }
                    results['cooperative_datasize_limit'] = {
                        cooperative_agent:self.basic_data_limit
                    }
                else: # still get closest in no candidates in range
                    cooperative_agent = str(np.argmin(distance2ego))
                    results['cooperative_veh_tokens'] = [cooperative_agent]
                    results['cooperative_agents'] = {
                        cooperative_agent: results['cooperative_agents'][cooperative_agent]
                    }
                    results['cooperative_datasize_limit'] = {
                        cooperative_agent:self.basic_data_limit
                    }
            if self.submode == "best_agent":
                if 'history_results' not in results.keys():
                    pass
                agents_data = self.withinRange(results)
                veh_token = results['veh_token']
                scene_token = results['scene_token']
                best_agent = None
                max_vehicles = -1
                max_score = -1

                for agent, data in agents_data.items():
                    num_vehicles = data['num_vehicles_within_range']
                    avg_score = np.mean(data['scores']) if data['scores'] else 0
                    if (num_vehicles > max_vehicles or (num_vehicles == max_vehicles and avg_score > max_score)) and agent != veh_token:
                        best_agent = agent
                        max_vehicles = num_vehicles
                        max_score = avg_score
                results['cooperative_veh_tokens'] = [best_agent]
                results['cooperative_agents'] = {
                    best_agent: results['cooperative_agents'][best_agent]
                }
                results['cooperative_datasize_limit'] = {
                    best_agent: self.basic_data_limit
                }
            # elif self.submode == "UCB":
            #     pass

            # elif self.submode == "mass-modified":
            #     self.updateEgoCPHistory(results)
            #     available_agents = list(results['cooperative_agents'].keys())

            #     cooperative_agent = None
            #     max_ucb_score = -float('inf')

            #     for agent in available_agents:
            #         if agent not in self.ego_cp_history:
            #             cooperative_agent = agent
            #             break
            #     else:
            #         for agent in available_agents:
            #             if agent in self.ego_cp_history:
            #                 # Retrieve last_seen_time and last_seen_gain from the dedicated dictionary
            #                 if agent in self.agent_performance_history:
            #                     performance_history = self.agent_performance_history[agent]
            #                     last_seen_time = performance_history['last_seen_time']
            #                     last_seen_gain = performance_history['last_seen_gain']
            #                     last_seen_score = performance_history['last_seen_score']
            #                 else:
            #                     # Fallback to the most recent record if not found in the performance history
            #                     last_cp_performance = self.ego_cp_history[agent][-1]
            #                     last_seen_time = last_cp_performance['timestep']
            #                     last_seen_gain = last_cp_performance['num_vehicles_within_range']
            #                     last_seen_score = np.mean(last_cp_performance['scores']) if last_cp_performance['scores'] else 0

            #                     # Update the dedicated dictionary

            #             # Calculate UCB score
            #             ucb_score = last_seen_gain + self.beta * np.sqrt(self.current_timestep - last_seen_time)

            #             if ucb_score > max_ucb_score:
            #                 max_ucb_score = ucb_score
            #                 cooperative_agent = agent
            #             # print(f"Agent {agent} has UCB score {ucb_score} at timestep {self.current_timestep}")

            #     if cooperative_agent is not None:
            #         # Update the results with the selected cooperative agent information
            #         results['cooperative_veh_tokens'] = [cooperative_agent]
            #         results['cooperative_agents'] = {
            #             cooperative_agent: results['cooperative_agents'][cooperative_agent]
            #         }
            #         results['cooperative_datasize_limit'] = {
            #             cooperative_agent: self.basic_data_limit
            #         }

            #         self.agent_performance_history[agent] = {
            #             'last_seen_time': last_seen_time,
            #             'last_seen_gain': last_seen_gain,
            #             'last_seen_score': last_seen_score,
            #         }


            elif self.submode == "mass":
                last_cp_agent = self.history_schedules.get(self.current_timestep - 5, None)
                agents_data = self.withinRange(results)
                available_agents = list(results['cooperative_agents'].keys())

                cooperative_agent = None
                max_ucb_score = float('-inf')

                # Loop over available agents to find a suitable cooperative agent
                for agent in available_agents:
                    if agent not in self.history_schedules:
                        cooperative_agent = agent
                        break
                else:
                    for agent in available_agents:
                        if agent not in self.history_agent_results:
                            self.history_agent_results[agent] = {
                                'last_seen_time': self.current_timestep,
                                'last_seen_gain': agents_data[agent]['num_vehicles_within_range']
                            }

                        last_seen_time = self.history_agent_results[agent]['last_seen_time']
                        last_seen_gain = self.history_agent_results[agent]['last_seen_gain']

                        # Calculate UCB score
                        ucb_score = last_seen_gain + self.beta * np.sqrt(self.current_timestep - last_seen_time)

                        # Apply penalty if this agent was the last cooperative agent chosen
                        if agent == last_cp_agent:
                            ucb_score *= 0.9  # Penalize by reducing its score by 10%

                        if ucb_score > max_ucb_score:
                            max_ucb_score = ucb_score
                            cooperative_agent = agent

                if cooperative_agent is not None:
                    results['cooperative_veh_tokens'] = [cooperative_agent]
                    results['cooperative_agents'] = {
                        cooperative_agent: results['cooperative_agents'][cooperative_agent]
                    }
                    results['cooperative_datasize_limit'] = {
                        cooperative_agent: self.basic_data_limit
                    }

                    self.history_agent_results[cooperative_agent] = {
                        'last_seen_time': self.current_timestep,
                        'last_seen_gain': agents_data[cooperative_agent]['num_vehicles_within_range']
                    }
            # elif self.submode == "mass":
            #     last_cp_agent = self.history_schedules.get(self.current_timestep - 5, None)
            #     agents_data = self.withinRange(results)
            #     # ego_agent = results['veh_token']
            #     available_agents = list(results['cooperative_agents'].keys())

            #     cooperative_agent = None

            #     for agent in available_agents:
            #         if agent not in self.history_schedules:
            #             cooperative_agent = agent
            #             break
            #     else:
            #         for agent in available_agents:
            #             if agent not in self.history_agent_results:
            #                 self.history_agent_results[agent] = {
            #                     'last_seen_time': self.current_timestep,
            #                     'last_seen_gain': agents_data[agent]['num_vehicles_within_range']
            #                 }
            #             last_seen_time = self.history_agent_results[agent]['last_seen_time']
            #             last_seen_gain = self.history_agent_results[agent]['last_seen_gain']

            #             ucb_score = last_seen_gain + self.beta * np.sqrt(self.current_timestep - last_seen_time)

            #             if ucb_score > max_ucb_score:
            #                 max_ucb_score = ucb_score
            #                 cooperative_agent = agent

            #     if cooperative_agent is not None:
            #         results['cooperative_veh_tokens'] = [cooperative_agent]
            #         results['cooperative_agents'] = {
            #             cooperative_agent: results['cooperative_agents'][cooperative_agent]
            #         }
            #         results['cooperative_datasize_limit'] = {
            #             cooperative_agent: self.basic_data_limit
            #         }

            #         self.history_agent_results[cooperative_agent] = {
            #             'last_seen_time': self.current_timestep,
            #             'last_seen_gain': agents_data[cooperative_agent]['num_vehicles_within_range']
            #         }

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
        """
        Update the historical cooperative planning results for the ego agent.
        This includes recording the timestep and the ego's detection results at that timestep.
        """
        # ego_id = results['veh_token']
        available_agents = list(results['cooperative_agents'].keys())
        ego_detections = self.withinEgoRange(results)  # Fetch current detections

        for agent_id in available_agents:
            if self.current_timestep > 10:
                print(f"Agent ID: {agent_id}, History Schedules: {self.history_schedules}")
                if agent_id == self.history_schedules[self.current_timestep - 5]:
                    self.ego_cp_history[agent_id].append({
                        'timestep': self.current_timestep,
                        'scores': ego_detections['scores'],
                        'num_vehicles_within_range': ego_detections['num_vehicles_within_range']
                    })

@PIPELINES.register_module()
class GlobalRotScaleTransCP(GlobalRotScaleTrans):
    """We perform same data augmentation for the cooperative agents, 
    this can only be done after points in cooperative agents are transformed 
    to the ego vehicle coordinate system
    """
    def _trans_bbox_points(self, input_dict):
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        ######
        input_dict['points'].translate(trans_factor)
        ######
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)
        if 'cooperative_agents' in input_dict.keys():
            for key in input_dict['cooperative_agents'].keys():
                if 'points' in input_dict['cooperative_agents'][key].keys():
                    assert isinstance(input_dict['cooperative_agents'][key]['points'], BasePoints)
                    input_dict['cooperative_agents'][key]['points'].translate(trans_factor)
        if 'infrastructure_points' in input_dict.keys():
            input_dict['infrastructure_points'].translate(trans_factor)
    
    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        if len(input_dict['bbox3d_fields']) == 0:
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T
            #####
            "rotation for cooperative agents in the same way"
            if 'cooperative_agents' in input_dict.keys():
                for key in input_dict['cooperative_agents'].keys():
                    if 'points' in input_dict['cooperative_agents'][key].keys():
                        assert isinstance(input_dict['cooperative_agents'][key]['points'], BasePoints)
                        rot_mat_T = input_dict['cooperative_agents'][key]['points'].rotate(noise_rotation)
            if 'infrastructure_points' in input_dict.keys():
                rot_mat_T = input_dict['infrastructure_points'].rotate(noise_rotation)
            #####
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T
                #####
                "we only need to rotate points of CP agents"
                if 'cooperative_agents' in input_dict.keys():
                    for key in input_dict['cooperative_agents'].keys():
                        if 'points' in input_dict['cooperative_agents'][key].keys():
                            assert isinstance(input_dict['cooperative_agents'][key]['points'], BasePoints)
                            rot_mat_T = input_dict['cooperative_agents'][key]['points'].rotate(-noise_rotation)
                            # note: rotate points is reverse to rotate box and point function
                if 'infrastructure_points' in input_dict.keys():
                    rot_mat_T = input_dict['infrastructure_points'].rotate(noise_rotation)
                #####

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        #####
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points
        # ego operation
        #####
        if 'cooperative_agents' in input_dict.keys():
            for key in input_dict['cooperative_agents'].keys():
                if 'points' in input_dict['cooperative_agents'][key].keys():
                    assert isinstance(input_dict['cooperative_agents'][key]['points'], BasePoints)
                    cp_points = input_dict['cooperative_agents'][key]['points']
                    cp_points.scale(scale)
                    if self.shift_height:
                        assert 'height' in cp_points.attribute_dims.keys(), \
                            'setting shift_height=True but points have no height attribute'
                        cp_points.tensor[:, cp_points.attribute_dims['height']] *= scale
                    input_dict['cooperative_agents'][key]['points'] = cp_points
        if 'infrastructure_points' in input_dict.keys():
            input_dict['infrastructure_points'].scale(scale)
            if self.shift_height:
                assert 'height' in input_dict['infrastructure_points'].attribute_dims.keys(), \
                    'setting shift_height=True but points have no height attribute'
                input_dict['infrastructure_points'].tensor[:, input_dict['infrastructure_points']\
                                                           .attribute_dims['height']] *= scale
        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

@PIPELINES.register_module()
class RandomFlip3DCP(RandomFlip3D):
    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                ####
                'ego filp points'
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
                ####
                "filp points of cooperative agents in the same way, points only"
                if 'cooperative_agents' in input_dict.keys():
                    for key in input_dict['cooperative_agents'].keys():
                        if 'points' in input_dict['cooperative_agents'][key].keys():
                            assert isinstance(input_dict['cooperative_agents'][key]['points'], BasePoints)
                            input_dict['cooperative_agents'][key]['points'].flip(direction)
                if 'infrastructure_points' in input_dict.keys():
                    input_dict['infrastructure_points'].flip(direction)
            else:
                input_dict[key].flip(direction)
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['ori_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

@PIPELINES.register_module()
class PointShuffleCP(PointShuffle):
    def __call__(self, input_dict):
        input_dict = super().__call__(input_dict)
        if "infrastructure_points" in input_dict.keys():
            idx_inf = input_dict['infrastructure_points'].shuffle()
        return input_dict
    
@PIPELINES.register_module()
class GlobalTransCP(object):
    def __init__(self, trans_factor):
        self.trans_factor = np.array(trans_factor)
    def __call__(self, input_dict):
        input_dict['points'].translate(self.trans_factor)
        if "infrastructure_points" in input_dict.keys():
            input_dict['infrastructure_points'].translate(self.trans_factor)
        # input_dict['pcd_trans'] = self.trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(self.trans_factor)
        return input_dict