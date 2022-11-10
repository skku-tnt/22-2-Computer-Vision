import os
from PIL import Image
import requests
import streamlit as st
import cv2

st.title("Mosaic for you")

file = st.file_uploader("Choose a video or image")

def main():

  st.sidebar.title("Settings")

  st.markdown(
  """  
  <style>
  [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width: 488px;}
  [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{width: 488px; margin-left: -488px}
  </style>
  """,
  unsafe_allow_html=True,
  )

  st.sidebar.markdown('---')
  #confidence = st.sidebar.slider('Confidence', min_value = 0.0, max_value=1.0, value=0.25)
  st.sidebar.markdown('---')

  assigne_class_id = []
  names = ["brand", "faces"]
  st.sidebar.markdown('---')
  
  stframe = st.empty()
  st.sidebar.markdown('---')

  kpi1, kpi2 = st.columns(2)

  with kpi1:
    st.markdown("**Tracked Objects**")
    kpi1_text = st.markdown('0')


if st.button("Image/Video inspect"):
  # get image/video info from backend
  if file is not None:
    print(file.name)
    files = {"file": file.getvalue()}
    data = {'filename': file.name}

    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
      res = requests.post(f"http://backend:8080/inspect_image", files=files)
    elif file.name.lower().endswith(('.avi', '.wmv', '.mp4', '.mkv', '.mov')):
      # res = requests.post(f"http://backend:8080/inspect_video", files=files)
      pass

    img_path = res.json()
    image = Image.open(img_path.get("name"))
    st.image(image, width=500)
    
    st.button("")


if st.button("Video convert"):
    if file is not None:
        files = {"file": file.getvalue()}
        res = requests.post(f"http://backend:8080/", files=files)
        cap_path = res.json()
        cap = cv2.VideoCapture(cap_path.get('name'))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        name = f"./output_tmp.mp4"
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
        
        #logger.info('Video Out done')
        os.system('ffmpeg -y -i output_tmp.mp4 -vcodec libx264 output.mp4')
        #os.system('ffmpeg -i {} -vcodec libx264 {}'.format(name, name.replace('_tmp', '')))
        video_file = open('output.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)
        
if __name__ == '__main__':
  try:
    main()
  except SystemExit:
    pass