FROM ubuntu:latest

RUN apt -y update && apt -y install python3 python3-pip

RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN mkdir /home/service
ADD data /home/service/data
COPY main.py /home/service/main.py

WORKDIR /home/service

CMD python3 /home/service/main.py