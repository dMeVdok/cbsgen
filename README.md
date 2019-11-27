# cbsgen

## a simple character-based sentence generator

based on [simple character-level RNN](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb "simple character-level RNN") by @rasbt

### running locally

```bash
pip3 install -r requirements.txt
python3 main.py
```

### docker build & run

```bash
docker build . -t cbsgen
docker run -p 8080:8080 -it cbsgen
```

or just run from the docker hub:

```bash
docker run -p 8080:8080 -it dmevdok/cbsgen:latest
```

## provided services

### GET /

A simple web interface for training/testing. Type your sentence and press "set" to train a model on it. Press "get" to generate a sentence.

### POST /train

Trains the model on sentences provided in POST body. Each sentence on the separate line.

### GET /test

Generate a sentence

### GET /state

Get JSON with service state: 
* `queue_size` -- num of sentences in queue for training, 
* `cpu` -- percentage of CPU usage
* `mem` -- percentage of RAM usage

### GET /checkpoint

Save model weights to `/home/service/model.pth` (use while running in docker)

### GET /restore

Load model weights