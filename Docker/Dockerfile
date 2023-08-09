From python:3.9.13-buster

RUN apt-get update

COPY requirements.txt /tm/

RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt