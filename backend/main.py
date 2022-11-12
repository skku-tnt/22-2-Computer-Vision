import os
import uuid
import time
import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image

import asyncio
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import configs
from inference import inference
from utils import mkdir

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

# get input image and find what objects are in the image
@app.post("/inspect_image")
def inspect_image(file: UploadFile = File(...)):
    image = cv2.cvtColor(np.array(Image.open(file.file)), cv2.COLOR_RGB2BGR)
    
    #cv2 image
    result_image = inference(image)

    output_dir = '/storage'
    mkdir(output_dir)

    name = f"/storage/{str(uuid.uuid4())}.png"
    cv2.imwrite(name, result_image)
    return {"name": name}


@app.post("/inspect_video")
def inspect_video(file: UploadFile = File(...)):
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
        
        name = process_video(temp.name)  # Pass temp.name to VideoCapture()
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        #temp.close()  # the `with` statement above takes care of closing the file
        os.remove(temp.name)
        
    return {"name": name}

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
            result_frame = inference(frame)
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


@app.post("/{style}")
async def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    model = configs.STYLES[style]
    start = time.time()
    output, resized = inference(model, image)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    cv2.imwrite(name, output)
    models = configs.STYLES.copy()
    del models[style]
    asyncio.create_task(generate_remaining_models(models, image, name))
    return {"name": name, "time": time.time() - start}

async def generate_remaining_models(models, image, name: str):
    executor = ProcessPoolExecutor()
    event_loop = asyncio.get_event_loop()
    await event_loop.run_in_executor(
        executor, partial(process_image, models, image, name)
    )

def process_image(models, image, name: str):
    for model in models:
        output, resized = inference(models[model], image)
        name = name.split(".")[0]
        name = f"{name.split('_')[0]}_{models[model]}.jpg"
        cv2.imwrite(name, output)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)