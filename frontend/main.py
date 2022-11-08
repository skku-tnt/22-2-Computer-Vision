#from loguru import logger
import requests
import streamlit as st
import cv2
import tempfile
import os
STYLES = {
    "candy": "candy",
    "composition 6": "composition_vii",
    "feathers": "feathers",
    "la_muse": "la_muse",
    "mosaic": "mosaic",
    "starry night": "starry_night",
    "the scream": "the_scream",
    "the wave": "the_wave",
    "udnie": "udnie",
}
st.title("Mosaic for you")

image = st.file_uploader("Choose an video")
style = st.selectbox("Choose the style", [i for i in STYLES.keys()])

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

  # checkboxes
  #save_img = st.sidebar.checkbox('Save Video')
  #enable_GPU = st.sidebar.checkbox('Enable GPU')
  #custom_classes = st.sidebar.checkbox('Use Custom Classes')
  assigne_class_id = []
  names = ["brand", "faces"]
  st.sidebar.markdown('---')
  
  # custom classes
  #if custom_classes: # if custom_classes checkbox is clicked
  #  assigned_class = st.sidebar.multiselect('Select the Custom Classes', names, default='brand')
  #  for each in assigned_class:
  #    assigne_class_id.append(names.index(each))

  # Uploading videos
  video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4","mov","avi","asf","m4v"])
  DEMO_VIDEO = 'test.mp4'
  tffile = tempfile.NamedTemporaryFile(suffix='.avi',delete=False)


  # get the input video
  if not video_file_buffer:
    vid = cv2.VideoCapture(DEMO_VIDEO)
    tffile.name = DEMO_VIDEO
    dem_vid = open(tffile.name, 'rb')
    demo_bytes = dem_vid.read()

    st.sidebar.text('Input Video')
    st.sidebar.video(demo_bytes)
  else:
    tffile.write(video_file_buffer.read())
    dem_vid = open(tffile.name, 'rb')
    demo_bytes = dem_vid.read()

    st.sidebar.text('Input Video')
    st.sidebar.video(demo_bytes)
  
  print(tffile.name)
  
  stframe = st.empty()
  st.sidebar.markdown('---')

  kpi1, kpi2 = st.columns(2)

  with kpi1:
    st.markdown("**Frame Rate**")
    kpi1_text = st.markdown('0')
  
  with kpi2:
    st.markdown("**Tracked Objects**")
    kpi1_text = st.markdown('0')

# Calling YOLOX
# load_yolor_and_process_each_frame()

# st.text("Video is Processed")
# vid.release()
if st.button("Style Transfer"):
    if image is not None and style is not None:
        files = {"file": image.getvalue()}
        res = requests.post(f"http://backend:8080/{style}", files=files)
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