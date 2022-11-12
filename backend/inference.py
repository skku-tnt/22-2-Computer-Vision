import os
import time
import numpy as np
import cv2
import onnxruntime

from utils import preprocess, demo_postprocess, multiclass_nms
from visualization import vis
from configs import COCO_CLASSES

def inference(cv2_image):

    model_path = 'yolox_s.onnx'

    input_shape = (640, 640)
    img, ratio = preprocess(cv2_image, input_shape)

    session = onnxruntime.InferenceSession(model_path)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=False)[0]
    score_thr = 0.3

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(cv2_image, final_boxes, final_scores, final_cls_inds,
                        conf=score_thr, class_names=COCO_CLASSES)

    return origin_img
