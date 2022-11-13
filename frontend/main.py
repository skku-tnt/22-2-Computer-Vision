from PIL import Image
import requests
import streamlit as st

st.title("Mosaic for you")

file = st.file_uploader("Choose a video or image")

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


if st.button("Image inspect"):
  st.session_state.disabled = False
  # get image/video info from backend
  if file is not None:
    print(file.name)
    files = {"file": file.getvalue()}
    data = {'filename': file.name}

    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
      res = requests.post(f"http://backend:8080/inspect_image", files=files)

      object_lists = res.json()
      object_lists = object_lists.get("data")

      if st.radio('Select objects to mosaic', object_lists):
        st.session_state.disabled = False


if st.button("Image/Video object detection"):
  # get image/video info from backend
  if file is not None:
    print(file.name)
    files = {"file": file.getvalue()}
    data = {'filename': file.name}

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