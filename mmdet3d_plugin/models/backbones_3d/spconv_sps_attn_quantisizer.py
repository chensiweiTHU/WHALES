from functools import partial
import torch
import torch.nn as nn
from spconv.core import ConvAlgo
from mmdet3d.models.builder import BACKBONES
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from ...models.model_utils.pruning_block import DynamicFocalPruningDownsample
from .spconv_backbone_voxelnext_sps import SparseSequentialBatchdict,PostActBlock,\
    SparseBasicBlock
from .spconv_sps_former import SPSPruningFormer
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
                padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', \
                conv_type=self.downsample_type[0], pruning_ratio=self.downsample_pruning_ratio[0],loss_mode="focal_sprs",\
                point_cloud_range=self.point_cloud_range),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )
        self.conv3 = SparseSequentialBatchdict(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), \
                indice_key='spconv3', conv_type=self.downsample_type[1], pruning_ratio=self.downsample_pruning_ratio[1],loss_mode="focal_sprs",point_cloud_range=self.point_cloud_range),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = SparseSequentialBatchdict(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), \
                indice_key='spconv4', conv_type=self.downsample_type[2], pruning_ratio=self.downsample_pruning_ratio[2],loss_mode="focal_sprs",point_cloud_range=self.point_cloud_range),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = SparseSequentialBatchdict(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), \
                indice_key='spconv5', conv_type=self.downsample_type[3], pruning_ratio=self.downsample_pruning_ratio[3],point_cloud_range=self.point_cloud_range),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        
        self.conv6 = SparseSequentialBatchdict(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), \
                indice_key='spconv6', conv_type=self.downsample_type[4], pruning_ratio=self.downsample_pruning_ratio[4],point_cloud_range=self.point_cloud_range),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }
        self.forward_ret_dict = {}
        self.backbone_model = None
        self.spsformer = SPSPruningFormer(channel=channels[4],)

    # def bound_backbone(self,backbone):
    #     self.backbone_model = backbone

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
        x_conv3, batch_dict = self.conv3(x_conv2, batch_dict)# 11X360X360x64 25474
        x_conv4, batch_dict = self.conv4(x_conv3, batch_dict)# 6X180X180x128 10729
        x_conv5, batch_dict = self.conv5(x_conv4, batch_dict)# 3X90X90x128 4860
        x_conv6, batch_dict = self.conv6(x_conv5, batch_dict)# 2X45X45x128 1958

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat(x_conv4.features, x_conv5.features, x_conv6.features))
        x_conv4.indices = torch.cat(x_conv4.indices, x_conv5.indices, x_conv6.indices)

        pruning_ratio = self.downsample_pruning_ratio[0]
        x_conv1_former, batch_dict = self.spsformer(x_conv1, x_conv4, pruning_ratio, batch_dict)
        voxel_importance = batch_dict['voxel_importance']
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
        _, indices = voxel_importance.view(-1,).sort()
        indices_im = indices[int(voxel_importance.shape[0]*pruning_ratio):] #  比例越高越压缩
        indices_nim = indices[:int(voxel_importance.shape[0]*pruning_ratio)]
        important_coords = x_conv1.indices[indices_im]#[:,1:] keep batch index
        unimportant_coords = x_conv1.indices[indices_nim]#[:,1:]
        important_voxels = batch_dict['voxels'][indices_im]
        unimportant_voxels = batch_dict['voxels'][indices_nim]
        vis_importance = False
        if vis_importance:
            all_coords = x_conv1.indices[:,1:]
            important_coords = important_coords.cpu().numpy()
            all_coords = all_coords.cpu().numpy()
            important_coords.tofile('visual/important_coords.bin')#,important_coords)
            all_coords.tofile('visual/all_coords.bin')
        
        return important_coords, unimportant_coords,important_voxels,unimportant_voxels, batch_dict
