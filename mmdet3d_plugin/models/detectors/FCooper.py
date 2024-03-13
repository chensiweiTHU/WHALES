from .OpenCood_detector import OpenCoodDetector
from .get_bev import get_bev_FCooper
from mmdet.models import DETECTORS
from types import MethodType
@DETECTORS.register_module(force=True)
class FCooper(OpenCoodDetector):
    def __init__(self, **kwargs):
        super(FCooper, self).__init__(**kwargs)
        self.opencood_model.get_bev = MethodType(get_bev_FCooper, self.opencood_model)


