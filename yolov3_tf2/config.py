from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.yolo_max_boxes = 100
__C.yolo_iou_threshold = 0.5
__C.yolo_score_threshold = 0.5