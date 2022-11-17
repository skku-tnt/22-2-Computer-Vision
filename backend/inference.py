import os
import uuid
import numpy as np
import cv2
import onnxruntime
from tempfile import NamedTemporaryFile

from utils import preprocess, demo_postprocess, multiclass_nms
from visualization import vis, vis_mosaic
from configs import MODEL_PATH, INPUT_SHAPE, COCO_CLASSES

def inference(cv2_image, ids:list=None):

    dets = get_dets(cv2_image)
    score_thr = 0.5
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    origin_img = vis(cv2_image, final_boxes, final_scores, final_cls_inds,
                        conf=score_thr, class_names=COCO_CLASSES, ids=ids)

    return origin_img

def inference_mosaic(cv2_image, ids:list=None):

    dets = get_dets(cv2_image)
    score_thr = 0.5
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    origin_img = vis_mosaic(cv2_image, final_boxes, final_scores, final_cls_inds,
                        conf=score_thr, class_names=COCO_CLASSES, ids=ids)

    return origin_img

# return what kind of object labels are in an image
def get_label_names(cv2_image, conf = 0.3):

    dets = get_dets(cv2_image)
    _, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    final_object_lists = []

    for i in range(len(final_cls_inds)):
        cls_id = int(final_cls_inds[i])
        score = final_scores[i]
        if score < conf:
            continue
        if cls_id not in final_object_lists:
            final_object_lists.append(cls_id)

    return final_object_lists

def get_dets(cv2_image):

    img, ratio = preprocess(cv2_image, INPUT_SHAPE)

    session = onnxruntime.InferenceSession(MODEL_PATH)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], INPUT_SHAPE, p6=False)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

    return dets

def process_video(video_name : NamedTemporaryFile.__name__):

    try:
        cap= cv2.VideoCapture(video_name)
    except Exception:
        print('error')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    name = f"/storage/{str(uuid.uuid4())}_tmp.mp4"
  

    vid_writer = cv2.VideoWriter(
            name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            result_frame = inference_mosaic(frame)
            vid_writer.write(result_frame)
 
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    cap.release()
    vid_writer.release()

    name_ = name.replace('_tmp', '')
    os.system('ffmpeg -i {} -vcodec libx264 {}'.format(name, name_))

    return name_
