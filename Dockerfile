FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /hyperadapter/

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY hyperformer hyperformer/