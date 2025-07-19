from functools import partial
import torch
import torch.nn as nn
from spconv.core import ConvAlgo
from mmdet3d.models.builder import BACKBONES
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from ...models.model_utils.pruning_block_confidence import DynamicFocalPruningDownsample


class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        for k, module in self._modules.items():
            if module is None:
                continue
            input, batch_dict = module(input, batch_dict)
        return input, batch_dict

class PostActBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, pruning_ratio=0.5, point_cloud_range=[-3, -46.08, 0, 1, 46.08, 92.16], #[0, -46.08, -3, 92.16, 46.08, 1],
                   voxel_stride=1,conv_type='subm', norm_fn=None, loss_mode=None, algo=ConvAlgo.Native, downsample_pruning_mode="topk"):
        super().__init__()
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        if conv_type == 'subm':
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        elif conv_type == 'spconv':
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key)
        elif conv_type == 'inverseconv':
            self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        elif conv_type == "dynamicdownsample_attn":
            self.conv = DynamicFocalPruningDownsample(in_channels, out_channels, kernel_size, stride=stride, padding=padding, indice_key=indice_key, bias=False, voxel_stride=voxel_stride,
                pruning_ratio=pruning_ratio, pred_mode="learnable", pred_kernel_size=3, loss_mode=loss_mode, algo=algo, pruning_mode=downsample_pruning_mode, point_cloud_range=point_cloud_range)
        else:
            raise NotImplementedError

        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, batch_dict):
        if isinstance(self.conv, (DynamicFocalPruningDownsample,)):
            x, batch_dict = self.conv(x, batch_dict)
        else:
            x = self.conv(x)
        x = replace_feature(x, self.bn(x.features))
        x = replace_feature(x, self.relu(x.features))
        return x, batch_dict


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, batch_dict):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out, batch_dict

@BACKBONES.register_module()
class VoxelResBackBone8xVoxelNeXtSPS(nn.Module):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "spconv", "spconv"]
    downsample_pruning_ratio = [0.5, 0.5, 0.5, 0., 0.]
        # def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
    #     super().__init__()
    #     self.model_cfg = model_cfg
    #     norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

    #     spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
    #     channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
    #     out_channel = model_cfg.get('OUT_CHANNEL', 128)
    def __init__(self, input_channels, grid_size, spconv_kernel_sizes=[3, 3, 3, 3], \
                 channels=[16, 32, 64, 128, 128], out_channel=128, point_cloud_range=[-3, -46.08, 0, 1, 46.08, 92.16], **kwargs):
        super().__init__()
        # self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        grid_size = np.array(grid_size) # Z, Y, X
    # def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
    #     super().__init__()
        # self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # spconv_kernel_sizes = model_cfg.get('spconv_kernel_sizes', [3, 3, 3, 3])
        # channels = model_cfg.get('channels', [16, 32, 64, 128, 128])
        # out_channel = model_cfg.get('out_channel', 128)

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
        )

        self.conv2 = SparseSequentialBatchdict(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), voxel_stride=1,\
                indice_key='spconv2', conv_type=self.downsample_type[0], pruning_ratio=self.downsample_pruning_ratio[0],loss_mode="focal_sprs",point_cloud_range=self.point_cloud_range),
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
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }
        self.forward_ret_dict = {}

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

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
        vis_importance = False
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
            important_coords.tofile('visual/important_coords.bin')#,important_coords
            all_coords.tofile('visual/all_coords.bin')
        x_conv2, batch_dict = self.conv2(x_conv1, batch_dict)# 21X720X720x32 42292
        x_conv3, batch_dict = self.conv3(x_conv2, batch_dict)# 11X360X360x64 25474
        x_conv4, batch_dict = self.conv4(x_conv3, batch_dict)# 6X180X180x128 10729
        x_conv5, batch_dict = self.conv5(x_conv4, batch_dict)# 3X90X90x128 4860
        x_conv6, batch_dict = self.conv6(x_conv5, batch_dict)# 2X45X45x128 1958

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out) #180X180X128, 14105
        out = self.shared_conv(out) #180X180X128, 14105

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
