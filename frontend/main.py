from PIL import Image
import requests
import streamlit as st
import json

from configs import COCO_CLASSES

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

  if file is not None:
    int_list, str_list = image_inspect(file)
  
    if str_list is not None:
      detect_selected_objects(int_list, str_list, file)

  button2 = st.button("Image/Video object detection")
  object_detection(button2, file)

def image_inspect(file):

  if (file is not None):
    files = {"file": file.getvalue()}

    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        res = requests.post(f"http://backend:8080/inspect_image", files=files)

        object_list = res.json()
        object_list = object_list.get("data")
        length = len(object_list)
        str_list = [COCO_CLASSES[object_list[i]] for i in range(length)]

        return object_list, str_list


def detect_selected_objects(int_list, str_list, file):

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
    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):

      id_string = encode(return_list)
      res = requests.post(f"http://backend:8080/process_selected_labels/{id_string}",files=files)

      img_path = res.json()
      image = Image.open(img_path.get("name"))
      st.image(image)

def object_detection(button, file):

  if button and (file is not None):
    files = {"file": file.getvalue()}

    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
      res = requests.post(f"http://backend:8080/image_detection", files=files)

      img_path = res.json()
      image = Image.open(img_path.get("name"))
      st.image(image)

    elif file.name.lower().endswith(('.avi', '.wmv', '.mp4', '.mkv', '.mov')):
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