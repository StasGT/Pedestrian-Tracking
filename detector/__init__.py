from .YOLOv3 import YOLOv3
from .MMDet import MMDet
from .YOLOv5 import YOLOv5
from .Mask_RCNN import Mask_RCNN

__all__ = ['build_detector']


def build_detector(cfg, use_cuda, segment=False):
    if cfg.USE_MMDET:
        return MMDet(cfg.MMDET.CFG, cfg.MMDET.CHECKPOINT,
                     score_thresh=cfg.MMDET.SCORE_THRESH,
                     is_xywh=True, use_cuda=use_cuda)
    elif cfg.USE_SEGMENT:
        return Mask_RCNN(segment, num_classes=cfg.MASKRCNN.NUM_CLASSES, box_thresh=cfg.MASKRCNN.BOX_THRESH,
                         label_json_path=cfg.MASKRCNN.LABEL, weight_path=cfg.MASKRCNN.WEIGHT)

    else:
        return Mask_RCNN(segment, num_classes=cfg.MASKRCNN.NUM_CLASSES, box_thresh=cfg.MASKRCNN.BOX_THRESH,
                         label_json_path=cfg.MASKRCNN.LABEL, weight_path=cfg.MASKRCNN.WEIGHT)
