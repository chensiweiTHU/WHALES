from .OpenCood_detector import OpenCoodDetector
from .get_bev import get_bev_V2VNet
from mmdet.models import DETECTORS
from types import MethodType
@DETECTORS.register_module(force=True)
class V2VNet(OpenCoodDetector):
    def __init__(self, **kwargs):
        super(V2VNet, self).__init__(**kwargs)
        self.opencood_model.get_bev = MethodType(get_bev_V2VNet, self.opencood_model)


