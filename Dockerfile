FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 default-mysql-client -y

WORKDIR /service
ADD . /service
RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

EXPOSE 32332

CMD ["python", "service.py", "--port=32332", "--model=YCVR_big"]