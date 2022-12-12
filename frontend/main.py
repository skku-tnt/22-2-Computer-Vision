from PIL import Image
import requests
import streamlit as st
import json
from loguru import logger

from configs import COCO_CLASSES
from pydantic import BaseModel

class Item(BaseModel):
        pass

class Data(BaseModel):
    user: str
def main():

  st.markdown(
  """  
  <style>
  [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width: 488px;}
  [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{width: 488px; margin-left: -488px}
  </style>
  """,
  unsafe_allow_html=True,
  )

  st.title("Mosaic for you")
  file = st.file_uploader("Choose a video or image")
  logger.info(type(file))
  if file is not None:
    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):
      int_list, str_list = image_inspect(file)  ## 이미지 전용
    
      if str_list is not None:
        detect_selected_objects(int_list, str_list, file)

      button2 = st.button("Image/Video object detection")
      object_detection( file)
    
    elif file.name.lower().endswith(('.avi', '.wmv', '.mp4', '.mkv', '.mov', '.gif',)):
      int_list, str_list, infered_image, faces = video_inspect(file)  ## 비디오 전용
      if str_list is not None:
        detect_selected_objects(int_list, str_list, file, infered_image= infered_image)
 
      face_buttons = get_face_buttons(faces)
      logger.info(face_buttons)
      not_mask_face = []
      for i in (range(len(face_buttons))):
        if face_buttons[i] and (file is not None):
          not_mask_face.append(faces[i][2])
          logger.info(i)
        
      logger.info(file)
      logger.info(not_mask_face)
      if not_mask_face:
        make_face_folders(not_mask_face)
        object_detection(file)


def make_face_folders(not_mask_face):
  faces = {}
  for i in range(len(not_mask_face)):
    faces['name'] = not_mask_face[i]
  logger.info(faces)
  logger.info(type(faces))
  faces = json.dumps(faces)
  res = requests.post(f"http://backend:8080/make_face_folders", data= faces)



def get_face_buttons(faces):
  face_buttons = []
  for i in range(len(faces)):
    img_path = faces[i][2]
    i = st.button('face ' + str(i))
    face_buttons.append(i)
    logger.info(img_path)
    logger.info(faces)
    image = Image.open(img_path)
    st.image(image)
      
  return face_buttons  
 
  
def image_inspect(file):

  if (file is not None):
    files = {"file": file.getvalue()}
    ## for image detection
    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):
        res = requests.post(f"http://backend:8080/inspect_image", files=files)

        object_list = res.json()
        object_list = object_list.get("data")
        length = len(object_list)
        str_list = [COCO_CLASSES[object_list[i]] for i in range(length)]

        return object_list, str_list
    
    
def video_inspect(file):

  if (file is not None):
    files = {"file": file.getvalue()}

    if file.name.lower().endswith(('.avi', '.wmv', '.mp4', '.mkv', '.mov', '.gif')):
        res = requests.post(f"http://backend:8080/inspect_video", files=files)

        object_list = res.json()
        logger.info(object_list)
        infered_image = object_list.get("infered_image")
        faces = object_list.get('faces')
        object_list = object_list.get("data")        
        length = len(object_list)
        str_list = [COCO_CLASSES[object_list[i]] for i in range(length)]

        return object_list, str_list, infered_image, faces

def detect_selected_objects(int_list, str_list, file, infered_image = None):

  checkbox_list = []
  st.write("Select labels to blur")

  if str_list is not None:
    for i in range(len(str_list)):
      label = str_list[i]
      checkbox = st.checkbox(label=label, value=False, key=label)
      checkbox_list.append(checkbox)
  
  if st.button("confirm and detect") and (file is not None):
    return_list = []
    for i in range(len(str_list)):
      if checkbox_list[i]:
        return_list.append(int_list[i])

    files = {"file": file.getvalue()}
    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):

      id_string = encode(return_list)
      res = requests.post(f"http://backend:8080/process_selected_labels/{id_string}",files=files)

      img_path = res.json()
      image = Image.open(img_path.get("name"))
      st.image(image)
      
    elif file.name.lower().endswith(('.avi', '.wmv', '.mp4', '.mkv', '.mov', '.gif')):

      id_string = encode(return_list)
      #res = requests.post(f"http://backend:8080/process_selected_labels/{id_string}",files=files)

      img_path = infered_image #res.json()
      image = Image.open(infered_image)
      st.image(image)

def object_detection(file):
  
  if (file is not None):
    files = {"file": file.getvalue()}
    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp',  '.webp')):
      res = requests.post(f"http://backend:8080/image_detection", files=files)

      img_path = res.json()
      image = Image.open(img_path.get("name"))
      st.image(image)

    elif file.name.lower().endswith(('.avi', '.wmv', '.mp4', '.mkv', '.mov','.gif',)):
      res = requests.post(f"http://backend:8080/video_detection", files=files)
      video_path = res.json()
      video_file = open(video_path.get("name"), 'rb')
      video_bytes = video_file.read()
   
      st.video(video_bytes)

def encode(id_list: list):

    return_string = ""
    length = len(id_list)
    for i in range(length):
        return_string += str(id_list[i])
        if i < length - 1:
            return_string += "_"
    return return_string
        
if __name__ == '__main__':
  try:
    main()
  except SystemExit:
    pass