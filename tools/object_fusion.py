import mmcv
import torch
import numpy as np
import pyquaternion
import copy

from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet3d.core import bbox3d2result, xywhr2xyxyr
def get_object_fusion_results(results,dataset,test_cfg,device):
    new_results = []
    for i, oldresult in enumerate(results):
        result = copy.deepcopy(oldresult)
        assert len(result['img_metas'])==1
        if 'cooperative_agents' in result['img_metas'][0]:
            ego_token = result['img_metas'][0]['ego_agent']['sample_idx']
            ego_scene,ego_frame,ego_veh=dataset.get_scene_frame_veh(ego_token)
            ego2global_translation = result['img_metas'][0]['ego_agent']['ego2global_translation']
            ego2global_rotation = result['img_metas'][0]['ego_agent']['ego2global_rotation']
            boxes = [result['pts_bbox']['boxes_3d']]
            labels = [result['pts_bbox']['labels_3d']]
            scores = [result['pts_bbox']['scores_3d']]
            for veh_token in result['img_metas'][0]['cooperative_agents'].keys():
                veh_full_token = dataset.fuse_token(ego_scene,ego_frame,veh_token)
                cp_index = dataset.token_map[veh_full_token]
                veh_result = copy.deepcopy(results[cp_index])
                veh_box = veh_result['pts_bbox']['boxes_3d']
                assert veh_full_token == veh_result['img_metas'][0]['ego_agent']['sample_idx']
                if len(veh_box)>0:
                    veh2global_translation = veh_result['img_metas'][0]['ego_agent']['ego2global_translation']
                    veh2global_rotation = veh_result['img_metas'][0]['ego_agent']['ego2global_rotation']
                    veh_box = box2nuscworld(veh_box,veh2global_translation,veh2global_rotation)
                    veh_box = nuscworld2box(veh_box,ego2global_translation,ego2global_rotation)
                    boxes.append(veh_box)
                    labels.append(veh_result['pts_bbox']['labels_3d'])
                    scores.append(veh_result['pts_bbox']['scores_3d'])
                    # 9dims, float32
                    result['img_metas'][0]['transmitted_data_size']+= 9 * len(veh_box) * 4
            if len(boxes)>1:
                result['pts_bbox'] = merge_results(boxes, scores, labels, test_cfg,device)  
                
            # print(result)  
        new_results.append(result)
    return new_results

def box2nuscworld(box,ego2global_translation,ego2global_rotation):
    box.flip('horizontal')
    box.rotate(-pyquaternion.Quaternion(ego2global_rotation).yaw_pitch_roll[0])
    box.translate(np.array(ego2global_translation))
    return box

def nuscworld2box(box,ego2global_translation,ego2global_rotation):
    box.translate(-np.array(ego2global_translation))
    box.rotate(-pyquaternion.Quaternion(ego2global_rotation).inverse.yaw_pitch_roll[0])
    box.flip('horizontal')
    return box

def merge_results(recovered_bboxes, recovered_scores, recovered_labels, test_cfg, device):
    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
    aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
    aug_scores = torch.cat(recovered_scores, dim=0)
    aug_labels = torch.cat(recovered_labels, dim=0)

    # load into device
    aug_bboxes = aug_bboxes.to(device)
    aug_bboxes_for_nms = aug_bboxes_for_nms.to(device)
    aug_scores = aug_scores.to(device)
    aug_labels = aug_labels.to(device)

    # TODO: use a more elegent way to deal with nms
    if test_cfg.use_rotate_nms:
        nms_func = nms_gpu
    else:
        nms_func = nms_normal_gpu

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class nms when merge bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)

    for class_id in range(torch.max(aug_labels).item() + 1):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]
        if len(bboxes_nms_i) == 0:
            continue
        selected = nms_func(bboxes_nms_i, scores_i, test_cfg.nms_thr)

        merged_bboxes.append(bboxes_i[selected, :])
        merged_scores.append(scores_i[selected])
        merged_labels.append(labels_i[selected])

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = torch.cat(merged_scores, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.max_num, len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]
    merged_labels = merged_labels[order]

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)
