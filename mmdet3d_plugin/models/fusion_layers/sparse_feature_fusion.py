import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch
import torch.nn as nn
from spconv.core import ConvAlgo
from mmdet3d.models.builder import FUSION_LAYERS
import numpy as np
from ...utils.spconv_utils import replace_feature, spconv
from mmdet3d.models import builder
from ..confidence_aware.communication import Communication
from ..confidence_aware.how2comm_preprocess import How2commPreprocess
from mmdet.models.utils import build_transformer

@FUSION_LAYERS.register_module()
class SparseCoF(nn.Module):
    def __init__(self, in_channels, out_channels, sparse_backbone_layer=None, comm_cfg=None, how2comm_cfg=None, transformer_cfg=None):
        super(SparseCoF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparse_convs = builder.build_backbone(sparse_backbone_layer)
        self.channel_fusion = spconv.SparseSequential(
            spconv.SparseConv2d(64, 64, 7, 1, 3,indice_key='subm1'),
        )
        self.communication = Communication(comm_cfg)
        self.how2comm = How2commPreprocess(how2comm_cfg)
        self.fuse_modules = build_transformer(transformer_cfg)

    def rearrange_feature_shape(self, x):

        return x

    def forward(self, x_veh, x_inf):
        batch_size = x_veh['encoded_spconv_tensor'].batch_size
        ups_fused = []
        ups_common = []
        ups_exclusive = []
        hm_veh = self.sparse_convs.hm_layer(x_veh['encoded_spconv_tensor'])
        hm_inf = self.sparse_convs.hm_layer(x_inf['encoded_spconv_tensor'])
        
        confidence_lists = self.communication(hm_veh, hm_inf)

        x1 = self.sparse_convs.block(x_veh['encoded_spconv_tensor'])
        x2 = self.sparse_convs.block(x_inf['encoded_spconv_tensor'])
            

        sparse_feats_1, sparse_feats_2, comm_loss, comm_rates = self.how2comm.communication(
            x1, x2, confidence_lists)
        sparse_feats_1 = self.channel_fusion(sparse_feats_1)
        sparse_feats_2 = self.channel_fusion(sparse_feats_2)
            
        x_fusion, output_lists = self.fuse_modules(sparse_feats_1, sparse_feats_2, confidence_lists)
        x_fusion = self.sparse_convs.deblock(x_fusion)
        ups_fused.append(x_fusion)
        x_common = self.sparse_convs.deblock(output_lists[0])
        ups_common.append(x_common)
        x_exclusive = self.sparse_convs.deblock(output_lists[1])
        ups_exclusive.append(x_exclusive)

        return ups_fused, ups_common, ups_exclusive, comm_loss, comm_rates