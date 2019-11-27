FROM ubuntu:latest

RUN mkdir /home/service
COPY main.py /home/service/main.py
COPY requirements.txt /tmp/requirements.txt

RUN apt -y update && apt -y install python3 python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /home/service

CMD python3 /home/service/main.py