from functools import partial
import torch
import torch.nn as nn
from spconv.core import ConvAlgo
from mmdet3d.models.builder import BACKBONES
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from ...models.model_utils.pruning_block_confidence import DynamicFocalPruningDownsample
from .spconv_backbone_voxelnext_sps import SparseSequentialBatchdict,PostActBlock,\
    SparseBasicBlock
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from ...models.backbones_3d.focal_sparse_conv.focal_sparse_utils import FocalLoss
# from .spconv_sps_former import SPSPruningFormer
"we use sparse pruning to choose important points"


@BACKBONES.register_module()
class VoxelResSPSAttnQuantiseizer(nn.Module):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "spconv", "spconv"]
    
    def __init__(self, input_channels, grid_size, spconv_kernel_sizes=[3,3,3,3], downsample_pruning_ratio = [0.5, 0.5, 0.5, 0, 0], \
                 channels=[16,32,64,128,128], point_cloud_range=[-3, -46.08, 0, 1, 46.08, 92.16], **kwargs):
        super().__init__()
        # self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        grid_size = np.array(grid_size) # Z, Y, X
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.downsample_pruning_ratio = downsample_pruning_ratio
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.point_cloud_range = point_cloud_range
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = PostActBlock
        self.conv1 = SparseSequentialBatchdict(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = SparseSequentialBatchdict(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, \
                padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', voxel_stride=1,\
                conv_type=self.downsample_type[0], pruning_ratio=self.downsample_pruning_ratio[0],loss_mode="focal_sprs",\
                point_cloud_range=self.point_cloud_range),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )
        self.conv3 = SparseSequentialBatchdict(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), voxel_stride=2,\
                indice_key='spconv3', conv_type=self.downsample_type[1], pruning_ratio=self.downsample_pruning_ratio[1],loss_mode="focal_sprs",point_cloud_range=self.point_cloud_range),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = SparseSequentialBatchdict(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), voxel_stride=4,\
                indice_key='spconv4', conv_type=self.downsample_type[2], pruning_ratio=self.downsample_pruning_ratio[2],loss_mode="focal_sprs",point_cloud_range=self.point_cloud_range),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }
        self.forward_ret_dict = {}
        self.backbone_model = None
        self.focal_loss = FocalLoss()

        self.importance_conv = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        # self.spsformer = SPSPruningFormer(channel=channels[4],)

    # def bound_backbone(self,backbone):
    #     self.backbone_model = backbone
    
    def find_important_conv_coords(self, x, voxel_importance, pruning_ratio, voxel_stride=1):
        _, indices = voxel_importance[:,0].view(-1,).sort()
        indices_im = indices[int(voxel_importance.shape[0]*pruning_ratio):] #  比例越高越压缩
        important_conv_coords = x.indices.float().clone()
        important_conv_coords[:,1:] = (x.indices[:,1:] + voxel_importance[:,1:]) * voxel_stride
        important_conv_coords = important_conv_coords[indices_im]#[:,1:] keep batch index
        # importance_mask = torch.isin(batch_dict['voxel_coords'].int(), important_conv_coords.int()).all(dim=1)
        # important_voxels = batch_dict['voxels'][importance_mask]
        # important_voxel_coords = batch_dict['voxel_coords'][importance_mask]
        return important_conv_coords

    def attn_prune_unimportant(self, important_conv_coords, batch_dict):
        batch_size = batch_dict['batch_size']
        batch_index = batch_dict['voxel_coords'][:,0]
        batch_coord_importance = []
        for b in range(batch_size):
            batch_index_mask = batch_index == b
            query = batch_dict['voxel_coords'][batch_index_mask][:,1:].float()
            keys = important_conv_coords[important_conv_coords[:,0]==b][:,1:].float()
            values = keys
            # print("query:", query.shape, "keys:", keys.shape, "values:", values.shape)
            attn_scores = torch.matmul(query, keys.transpose(-2,-1))/torch.sqrt(torch.tensor(keys.shape[-1]))
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
            coords_weight = torch.matmul(attn_weights, values)
            # print("coords_weight:", coords_weight.shape)
            coords_importance = self.importance_conv(coords_weight)
            batch_coord_importance.append(coords_importance)
        batch_coord_importance = torch.cat(batch_coord_importance)
        return batch_coord_importance



    def calulate_focal_loss(self,x, important_coords, voxel_importance, batch_dict, voxel_stride=1):
        spatial_indices = important_coords[:, 1:] * voxel_stride
        spatial_shape = torch.tensor(x.spatial_shape,device=x.features.device).float()
        spatial_shape[0] = spatial_shape[0] - 1 # we added 1 before
        point_cloud_range = torch.Tensor(self.point_cloud_range).cuda()
        voxel_size = (point_cloud_range[3:] - point_cloud_range[:3])/spatial_shape
        voxels_3d = spatial_indices * voxel_size + point_cloud_range[:3]
        # print("voxel_stride:", voxel_stride, "point_cloud_range:", self.point_cloud_range, "voxel_size:", self.voxel_size)
        batch_size = x.batch_size
        mask_voxels = []
        box_of_pts_cls_targets = []
        
        for b in range(batch_size):
            if True:
                index=important_coords[:, 0]
                batch_index = index == b
                mask_voxel = voxel_importance[batch_index]
                if torch.max(batch_index)==False:
                    continue
                voxels_3d_batch = voxels_3d[batch_index].unsqueeze(0)
                mask_voxels.append(mask_voxel)
                gt_boxes = batch_dict['gt_boxes'][b, :, :7].unsqueeze(0)
                box_of_pts_batch = points_in_boxes_gpu(voxels_3d_batch[:, :, self.inv_idx], gt_boxes).squeeze(0)
                box_of_pts_cls_targets.append(box_of_pts_batch>=0)
        
        loss_box_of_pts = 0
        if True:
            mask_voxels = torch.cat(mask_voxels).squeeze(-1)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            # print("mask_voxels", mask_voxels.shape, "box_of_pts_cls_targets:",box_of_pts_cls_targets.sum())
            mask_voxels_two_classes = torch.cat([1-mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            # print("mask_voxels_two_classes", mask_voxels_two_classes.shape, "box_of_pts_cls_targets:",box_of_pts_cls_targets.shape)
            loss_box_of_pts = self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())

        return loss_box_of_pts

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        batch_dict['loss_box_of_pts_sprs']=0

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor) # 41X1440X1440x16 33954 
        x_conv1, batch_dict = self.conv1(x, batch_dict) # 41X1440X1440x16 33954 
        x_conv2, batch_dict = self.conv2(x_conv1, batch_dict)# 21X720X720x32 42292
        voxel_importance_conv1 = batch_dict['voxel_importance']

        x_conv3, batch_dict = self.conv3(x_conv2, batch_dict) # 11X360X360x64 25372
        voxel_importance_conv2 = batch_dict['voxel_importance']

        x_conv4, batch_dict = self.conv4(x_conv3, batch_dict) # 6X180X180x128  11492
        voxel_importance_conv3 = batch_dict['voxel_importance']

        pruning_ratio = self.downsample_pruning_ratio
        # return x_conv2, batch_dict
        # if self.backbone_model is None:
        #     x = self.conv_input(input_sp_tensor) # 41X1440X1440x16 33954 
        #     x_conv1, batch_dict = self.conv1(x, batch_dict) # 41X1440X1440x16 33954 
        #     pruning_ratio = self.downsample_pruning_ratio[0]
        # else:
        #     x = self.backbone_model.conv_input(input_sp_tensor) # 41X1440X1440x16 33954
        #     x_conv1, batch_dict = self.backbone_model.conv1(x, batch_dict) # 41X1440X1440x16 33954
        #     pruning_ratio = self.backbone_model.downsample_pruning_ratio[0]
        # x_features = x_conv1.features
        # x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
        # sigmoid = nn.Sigmoid()
        # voxel_importance = sigmoid(x_attn_predict.view(-1, 1))
        important_conv1_coords = self.find_important_conv_coords(x_conv1, voxel_importance_conv1, pruning_ratio[0], voxel_stride=1)
        important_conv2_coords = self.find_important_conv_coords(x_conv2, voxel_importance_conv2, pruning_ratio[1], voxel_stride=2)
        important_conv3_coords = self.find_important_conv_coords(x_conv3, voxel_importance_conv3, pruning_ratio[2], voxel_stride=4)
        important_conv_coords = torch.cat([important_conv1_coords, important_conv2_coords, important_conv3_coords], dim=0)
        batch_coord_importance = self.attn_prune_unimportant(important_conv_coords, batch_dict)
        batch_dict['voxel_importance'] = batch_coord_importance
        _, indices = batch_coord_importance.view(-1,).sort()
        indices_im = indices[int(batch_coord_importance.shape[0]*pruning_ratio[3]):]
        important_coords = batch_dict['voxel_coords'][indices_im]
        important_voxels = batch_dict['voxels'][indices_im]
        kept_coord_importance = batch_coord_importance[indices_im]
        loss_box_of_pts_sprs = self.calulate_focal_loss(x_conv1, important_coords, kept_coord_importance, batch_dict)
        batch_dict['loss_box_of_pts_sprs_pruning'] = loss_box_of_pts_sprs
        vis_importance = False
        if vis_importance:
            all_coords = x_conv1.indices[:,1:]
            important_coords = important_coords.cpu().numpy()
            all_coords = all_coords.cpu().numpy()
            important_coords.tofile('visual/important_coords.bin')#,important_coords)
            all_coords.tofile('visual/all_coords.bin')
        
        return important_coords,important_voxels, batch_dict
