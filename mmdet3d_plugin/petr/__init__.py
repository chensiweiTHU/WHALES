# PETR for WHALES (mmcv 1.4 / mmdet 2.14 / mmdet3d 0.17.1).
#
# Adapted from https://github.com/megvii-research/PETR (and the mmdet3d
# 1.x port at projects/PETR) to fit the legacy registry / API stack used
# by the rest of this codebase.
from .models import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
