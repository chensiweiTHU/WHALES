# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import spconv.pytorch as spconv
from spconv.pytorch import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
                   SparseConvTranspose3d, SparseInverseConv2d,
                   SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.core import SparseConvTensor, scatter_nd
from mmcv.cnn import CONV_LAYERS
CONV_LAYERS.register_module('SparseConv2d', module=SparseConv2d)
CONV_LAYERS.register_module('SparseConv3d', module=SparseConv3d)
CONV_LAYERS.register_module('SparseConvTranspose2d', module=SparseConvTranspose2d)
CONV_LAYERS.register_module('SparseConvTranspose3d', module=SparseConvTranspose3d)
CONV_LAYERS.register_module('SparseInverseConv2d', module=SparseInverseConv2d)
CONV_LAYERS.register_module('SparseInverseConv3d', module=SparseInverseConv3d)
CONV_LAYERS.register_module('SubMConv2d', module=SubMConv2d)
CONV_LAYERS.register_module('SubMConv3d', module=SubMConv3d)
# from .conv import
__all__ = [
    'SparseConv2d',
    'SparseConv3d',
    'SubMConv2d',
    'SubMConv3d',
    'SparseConvTranspose2d',
    'SparseConvTranspose3d',
    'SparseInverseConv2d',
    'SparseInverseConv3d',
    'SparseModule',
    'SparseSequential',
    'SparseMaxPool2d',
    'SparseMaxPool3d',
    'SparseConvTensor',
    'scatter_nd',
]
