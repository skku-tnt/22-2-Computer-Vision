import os
import uuid
import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image

from tempfile import NamedTemporaryFile

from inference import inference, process_video, get_label_names, process_selected_labels
from utils import mkdir

class customdata():
	ids : list

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

# get input image and find what objects are in the image
@app.post("/inspect_image")
def inspect_image(file: UploadFile = File(...)):
    image = cv2.cvtColor(np.array(Image.open(file.file)), cv2.COLOR_RGB2BGR)

    label_names = get_label_names(image)

    return {"data": label_names}

@app.post("/inspect_video")
def inspect_video(file: UploadFile = File(...)):
    

    pass

@app.post("/process_selected_labels")
def process_selected_labels(data: customdata, file: UploadFile = File(...)):
    image = cv2.cvtColor(np.array(Image.open(file.file)), cv2.COLOR_RGB2BGR)

    # form : {"ids" : [1, 2, 3]}
    received = data.dict()
    ids = received["ids"]

    #cv2 image
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
                f.write(contents);
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
        
        name = process_video(temp.name)
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)
        
    return {"name": name}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)