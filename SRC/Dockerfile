
FROM python:3.10


WORKDIR /app


COPY requirements.txt .
RUN pip install --default-timeout=600 -r requirements.txt

COPY . .


CMD ["python", "./app.py"]
