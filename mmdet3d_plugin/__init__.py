# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

try:
    import mmdet3d.core as _mmdet3d_core  # noqa: F401
except Exception:
    pass

from mmcv.utils import Registry as _Registry

_orig_register_module = _Registry._register_module


def _register_with_force(self, module_class, module_name=None, force=False):
    return _orig_register_module(
        self, module_class, module_name=module_name, force=True)


_Registry._register_module = _register_with_force

import mmdet
import mmseg
from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.3.8'
mmcv_maximum_version = '1.4.0'
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

mmdet_minimum_version = '2.14.0'
mmdet_maximum_version = '3.0.0'
mmdet_version = digit_version(mmdet.__version__)
assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version <= digit_version(mmdet_maximum_version)), \
    f'MMDET=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version}, ' \
    f'<={mmdet_maximum_version}.'

mmseg_minimum_version = '0.14.1'
mmseg_maximum_version = '1.0.0'
mmseg_version = digit_version(mmseg.__version__)
assert (mmseg_version >= digit_version(mmseg_minimum_version)
        and mmseg_version <= digit_version(mmseg_maximum_version)), \
    f'MMSEG=={mmseg.__version__} is used but incompatible. ' \
    f'Please install mmseg>={mmseg_minimum_version}, ' \
    f'<={mmseg_maximum_version}.'

__all__ = ['__version__', 'short_version']
