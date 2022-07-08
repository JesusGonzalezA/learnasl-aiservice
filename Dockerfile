FROM python:3.8

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]