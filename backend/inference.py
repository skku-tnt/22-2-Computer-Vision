from PIL import Image
import numpy as np
import cv2

def inference(image):
    model_name = f"yolox_s"
    model = cv2.dnn.readNetFromTorch(model_name)
    print(model)

image_name = "test.jpg"
image = np.array(Image.open(image_name))
inference(image);

