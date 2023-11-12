FROM python:3.10


WORKDIR /SRC/app


COPY SRC/requirements.txt .


RUN pip install --default-timeout=600 -r requirements.txt


COPY SRC/app.py .


CMD ["python", "./app.py"]
