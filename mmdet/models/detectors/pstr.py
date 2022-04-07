# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .detr_reid import DETRReID


@DETECTORS.register_module()
class PSTR(DETRReID):

    def __init__(self, *args, **kwargs):
        super(DETRReID, self).__init__(*args, **kwargs)
