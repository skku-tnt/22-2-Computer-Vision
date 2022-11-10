FROM python:3.10.1-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m pip install opencv-python



COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]