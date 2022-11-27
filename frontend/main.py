from PIL import Image
import requests
import streamlit as st

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

  button1 = st.button("Image inspect")
  str_lists = image_inspect(button1, file)

  res = make_checkbox(str_lists)

  button2 = st.button("Image/Video object detection")
  object_detection(button2, file)


def image_inspect(button, file):

  if button and (file is not None):
    files = {"file": file.getvalue()}

    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        res = requests.post(f"http://backend:8080/inspect_image", files=files)

        object_lists = res.json()
        object_lists = object_lists.get("data")
        length = len(object_lists)
        str_lists = [COCO_CLASSES[object_lists[i]] for i in range(length)]

        return str_lists

def make_checkbox(str_lists):

  checkbox_list = []

  if str_lists is not None:
    for i in range(len(str_lists)):
      label = str_lists[i]
      checkbox = st.checkbox(label=label, value=False, key=label)
      checkbox_list.append(checkbox)

  if checkbox_list is not None:
    for checkbox in checkbox_list:
      if checkbox:
        st.write(checkbox.label)

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

        
if __name__ == '__main__':
  try:
    main()
  except SystemExit:
    pass