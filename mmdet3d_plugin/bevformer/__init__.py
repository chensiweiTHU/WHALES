"""BEVFormer port for the WHALES plugin.

Imports here trigger registration of all BEVFormer modules
(MODELS / HEADS / TRANSFORMER / ATTENTION / BBOX_CODERS / ...) into mmdet's
global registries when ``mmdet3d_plugin.bevformer`` is imported.
"""

from .core import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
from .bevformer import *  # noqa: F401,F403
