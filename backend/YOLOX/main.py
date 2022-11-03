import os
import uuid
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image
from tools.demo import main, make_parser
from yolox.exp import get_exp
from loguru import logger

app = FastAPI()
print('hello')
args = make_parser().parse_args(args=['image'])
exp = get_exp(None,'yolox_nano')
main(exp,args)
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/{style}")
def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    #model = config.STYLES[style]
    args = make_parser().parse_args(args=['image'])
    exp = get_exp(None,'yolox_nano')
    main(exp,args)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    
    return {"name": name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)