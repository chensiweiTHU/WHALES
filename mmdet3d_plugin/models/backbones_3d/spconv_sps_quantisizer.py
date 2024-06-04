from functools import partial
import torch
import torch.nn as nn
from spconv.core import ConvAlgo
from mmdet3d.models.builder import BACKBONES
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from ...models.model_utils.pruning_block import DynamicFocalPruningDownsample
from .spconv_backbone_voxelnext2d_sps import SparseSequentialBatchdict,PostActBlock,\
    SparseBasicBlock
"we use sparse pruning to choose important points"


@BACKBONES.register_module()
class VoxelResBackBone8xVoxelNeXtSPS(nn.Module):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "spconv", "spconv"]
    downsample_pruning_ratio = [0.5,]
    def __init__(self, input_channels, grid_size, spconv_kernel_sizes=[3], \
                 channels=[16], out_channel=128, **kwargs):
        super().__init__()
        # self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        grid_size = np.array(grid_size) # Z, Y, X
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        # block = PostActBlock

        self.conv1 = SparseSequentialBatchdict(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        # self.conv2 = SparseSequentialBatchdict(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type=self.downsample_type[0], pruning_ratio=self.downsample_pruning_ratio[0],loss_mode="focal_sprs"),
        #     # SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        #     # SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        # )

        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }
        self.forward_ret_dict = {}


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
        vis_importance = True
        if vis_importance:
            pruning_ratio = 0.8
            x_features = x_conv1.features
            x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
            sigmoid = nn.Sigmoid()
            voxel_importance = sigmoid(x_attn_predict.view(-1, 1))
            _, indices = voxel_importance.view(-1,).sort()
            indices_im = indices[int(voxel_importance.shape[0]*pruning_ratio):]
            indices_nim = indices[:int(voxel_importance.shape[0]*pruning_ratio)]
            important_coords = x_conv1.indices[indices_im][:,1:]
            all_coords = x_conv1.indices[:,1:]
            important_coords = important_coords.cpu().numpy()
            all_coords = all_coords.cpu().numpy()
            important_coords.tofile('visual/important_coords.bin')#,important_coords)
            all_coords.tofile('visual/all_coords.bin')
        
        return batch_dict
