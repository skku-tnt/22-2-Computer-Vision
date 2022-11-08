import os
import uuid
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from tools.demo import main, make_parser
from yolox.exp import get_exp
from loguru import logger
import cv2
#import argparse
import time
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime

class Video(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    file_name: str
    created_at : datetime = Field(default_factory=datetime.now)
    

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/{style}")
def get_image(style: str, file: UploadFile = File(...)):
    #image = np.array(Image.open(file.file))
    
    # 폴더 생성후 받은 동영상 저장
    current_time = time.localtime()   
    cap = file.file
    vis_folder = './videos'
    save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    input_file_name = save_folder + '/input.avi'
    save_file_name = save_folder + '/output.avi'
    contents = cap.read()
    with open(os.path.join(save_folder, 'input.avi'), "wb") as fp:
        fp.write(contents)
        logger.info(file.filename)
    #return {"filenames": ['test.avi']}
    
    
    # input video불러온 후 inference
    #cap = cv2.VideoCapture(input_file_name)
    #width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    #logger.info("재생할 파일 넓이, 높이 : %d, %d"%(width, height))
    
    # 
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter(input_file_name, fourcc, fps, (int(width), int(height)))
    '''
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        #cv2.imshow('frame',frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()

    '''
    args=['video','yolox-nano','yolox-nano',input_file_name]
    logger.info(args)
    args = make_parser().parse_args(args=['video'])
    args.experiment_name = 'yolox_nano'
    args.name = 'yolox_nano'
    args.path = input_file_name
    exp = get_exp(None,'yolox_nano')
    logger.info("start")
    main(exp,args, save_file_name)
    logger.info("done")
    
    #동영상 frontend로 전달
    cap = cv2.VideoCapture(save_file_name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    name = f"/storage/{str(uuid.uuid4())}.avi"
    out = cv2.VideoWriter(name, fourcc, fps, (int(width), int(height)))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == False:
            break
    
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    logger.info('Video Out done')
    return {"name": name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)