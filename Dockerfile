FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /hyperadapter/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY hyperformer hyperformer/