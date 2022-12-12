import os
import uuid
import cv2
from requests import request
import uvicorn
from fastapi import File
from fastapi import FastAPI, Request
from fastapi import UploadFile
import numpy as np
from PIL import Image
from loguru import logger
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
import shutil


from inference import inference, process_video, get_label_names, save_video, inference_first_frame
from utils import mkdir, decode, get_first_frame

app = FastAPI()
class Data(BaseModel):
    user: str
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

# get input image and find what objects are in the image
@app.post("/inspect_image")
def inspect_image(file: UploadFile = File(...)):
    image = cv2.cvtColor(np.array(Image.open(file.file)), cv2.COLOR_RGB2BGR)

    label_names = get_label_names(image)

    return {"data": label_names}

# video 첫번째 이미지 불러와서 인물 따오기
@app.post("/inspect_video")
def inspect_video(file: UploadFile = File(...)):
    temp = NamedTemporaryFile(delete=False)
    
    
    try:
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
        name = save_video(temp.name)
        logger.info("video save Done")
        logger.info(name)
        image = get_first_frame(name)
        label_names = get_label_names(image)
        logger.info(label_names)
        first_frame_name = f"/storage/{str(uuid.uuid4())}.png"
        cv2.imwrite(first_frame_name, image)
        logger.info('save image done')
        image, faces = inference_first_frame(image)
        logger.info("inference firest frame done")
        logger.info(name)
        infered_image = f"/storage/{str(uuid.uuid4())}.png"
        cv2.imwrite(infered_image, image)

    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)
    # use 'get_label_names' per video frame
    return {"data": label_names,'infered_image' : infered_image, 'faces': faces}

@app.post("/process_selected_labels/{ids}")
def process_selected_labels(ids: str, file: UploadFile = File(...)):

    image = cv2.cvtColor(np.array(Image.open(file.file)), cv2.COLOR_RGB2BGR)

    ids = decode(ids)
    result_image = inference(image, ids)

    output_dir = '/storage'
    mkdir(output_dir)

    name = f"/storage/{str(uuid.uuid4())}.png"
    cv2.imwrite(name, result_image)
    return {"name": name}

# get input image and return object detection result
@app.post("/image_detection")
def image_detection(file: UploadFile = File(...)):
    image = cv2.cvtColor(np.array(Image.open(file.file)), cv2.COLOR_RGB2BGR)

    #cv2 image
    result_image = inference(image)

    output_dir = '/storage'
    mkdir(output_dir)

    name = f"/storage/{str(uuid.uuid4())}.png"
    cv2.imwrite(name, result_image)
    return {"name": name}

# get input video and return object detection result
@app.post("/video_detection")
def video_detection(file: UploadFile = File(...)):
    temp = NamedTemporaryFile(delete=False)

    try:
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
        logger.info('read video done')
        name = process_video(temp.name)
        logger.info('video process done')
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)
        
    return {"name": name}

@app.post("/make_face_folders")
async def make_face_folders(data: Request):
    data = await data.json()
   
    name = data['name']
    if data is not None:
        folder_name = f"/app/live_face_recognition/photos/{str(uuid.uuid4())}"
        os.mkdir(folder_name)
        file_name = folder_name + '/' + name
        shutil.move(name, folder_name)

    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)