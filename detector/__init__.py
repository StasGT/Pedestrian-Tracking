from .Mask_RCNN import Mask_RCNN

__all__ = ['build_detector']


def build_detector(segment=False):
    LABEL = './coco_classes.json'
    WEIGHT = './detector/Mask_RCNN/save_weights/maskrcnn_resnet50_fpn_coco.pth'
    NUM_CLASSES = 90
    BOX_THRESH = 0.5

    return Mask_RCNN(segment, num_classes=NUM_CLASSES, box_thresh=BOX_THRESH,
                         label_json_path=LABEL, weight_path=WEIGHT)
