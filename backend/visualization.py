from turtle import pos
import cv2
import numpy as np
import uuid
from loguru import logger

temp_xy = []
def im_trim(img, x_0, y_0, x_1, y_1):
    imgtrim = img[y_0: y_1, x_0: x_1]
    return imgtrim

## 얼굴만 잘라서 저장
def vis_face(img, face_boxes, scores, cls_ids, conf=0.5, class_names=None, ids=None):

    face_boxes = face_boxes.astype(int)
    face_boxes = face_boxes.tolist()
    faces = []  
   
    for i in range(len(face_boxes)):
        imgtrim = im_trim(img, face_boxes[i][0], face_boxes[i][1],  face_boxes[i][2],  face_boxes[i][3])
        name = f"/storage/{str(uuid.uuid4())}.png"
        faces.append([i ,[face_boxes[i][0], face_boxes[i][1],  face_boxes[i][2],  face_boxes[i][3]], name])
        # 번호, 좌표들, 얼굴사진 주소
        cv2.imwrite(name, imgtrim)

    return faces

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None, ids=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]

        if score < conf:
            continue

        if ids is not None and cls_id not in ids:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def vis_mosaic(img, boxes, face_boxes, scores, cls_ids, conf=0.5, class_names=None, ids=None):
    pos_xy = []
    logger.info(face_boxes)

    if face_boxes:
        face_boxes = face_boxes[0]
        face_boxes = list(map(int, face_boxes))
        logger.info(face_boxes)
        x_mid, y_mid = (face_boxes[0] + face_boxes[2]) / 2, (face_boxes[1] + face_boxes[3]) / 2
        pos_xy.append([x_mid, y_mid])
    elif not face_boxes:
        logger.info(face_boxes)

    for i in range(len(boxes)):
        boxes[i] = list(map(int, boxes[i]))
        for j in range(len(boxes[i])):
            if boxes[i][j] == 0:
                boxes[i][j] = 1
        
    for i in range(len(boxes)):
        if pos_xy:
            logger.info('before x_mid')
            x_mid = pos_xy[0][0]
            y_mid = pos_xy[0][1]
            logger.info('before append')
            temp_xy.append((x_mid, y_mid, 20))
            logger.info('after append')
            logger.info(x_mid)
            logger.info(boxes[i])
            check_1 = False
            for j in range(len(boxes[i])):
                if boxes[i][j] == 1:
                    logger.info('1 in box')
                    check_1 = True
            if check_1 == True:
                continue
                
            if boxes[i][0] < x_mid < boxes[i][2] and boxes[i][1] < y_mid < boxes[i][3]:
                continue
            img = mosaic_area(img,  boxes[i], 0.1)
        flag = False
        '''
        for j in range(len(temp_xy)):
            x_mid, y_mid, count = temp_xy.pop(0)
            count -= 1
            if boxes[i][0] < x_mid < boxes[i][2] and boxes[i][1] < y_mid < boxes[i][3]: 
                flag = True
                continue
            if count != 0:
                temp_xy.append((x_mid, y_mid, count))
            if flag == False:
                img = mosaic_area(img,  boxes[i], 0.1)
                break
        '''
    return img
    

def mosaic(src, ratio):
    """
    ### 모자이크 기능
    :param src: 이미지 소스
    :param ratio: 모자이크 비율
    :return: 모자이크가 처리된 이미지
    """
    '''
    try:
        small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        mosaic_img = cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        except:
        break
        '''

    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic_img = cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return mosaic_img


def mosaic_area(src, outputs, ratio):
    """
    ### 부분 모자이크 기능
    :param src: 이미지 소스
    :param x: 가로축 모자이크 시작 범위
    :param y: 세로축 모자이크 시작 범위
    :param width: 모자이크 범위 넓이
    :param height: 모자이크 범위 폭
    :param ratio: 모자이크 비율
    :return: 부분 모자이크가 처리된 이미지
    """
    mosaic_area_img = src.copy()
    for i in range(len(outputs)):
        if outputs[i] < 0 :
            outputs[i] = 0
    mosaic_area_img[int(outputs[1]):int(outputs[3]), int(outputs[0]):int(outputs[2])] = mosaic(mosaic_area_img[int(outputs[1]):int(outputs[3]), int(outputs[0]):int(outputs[2])], ratio)
        
            
        
    return mosaic_area_img


def opencv_img_save(img, save_img_path, save_img_name):
    """
    ### 처리 이미지 저장 기능
    :param img: 저장할 이미지
    :param save_img_path: 이미지 저장 경로
    :param save_img_name: 저장할 이미지 명
    """
    cv2.imwrite(save_img_path + save_img_name, img)




_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)