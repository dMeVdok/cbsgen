import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import time
import random
import string
import random
import re
import queue
import threading
import psutil

from http.server import BaseHTTPRequestHandler, HTTPServer

training_queue = queue.Queue()
queue_true_size = 0
training_active_thread = None

torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_PORTION_SIZE = 200

NUM_ITER = 20000
LEARNING_RATE = 0.005
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
NUM_HIDDEN = 1

good_chars = "~ ()!?.,;-йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁqwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"

stopping_chars = ".!?"


def char_to_tensor(text, test=False):
    if not test:
        text = "~" + text
        if len(text) == 0:
            text = " "
        if len(text) < TEXT_PORTION_SIZE + 1:
            if text[-1] not in stopping_chars:
                text = text + "."
            text = text.ljust(TEXT_PORTION_SIZE + 1)
        if len(text) > TEXT_PORTION_SIZE + 1:
            text = text[: TEXT_PORTION_SIZE + 1]
    lst = [good_chars.index(c) for c in text]
    tensor = torch.tensor(lst).long()
    return tensor


def get_train_target(text):
    text = text.replace("~", "")
    text_long = char_to_tensor(text)
    inputs = text_long[:-1]
    targets = text_long[1:]
    return inputs, targets


class RNN(torch.nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embed = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(
            input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.init_hidden = torch.nn.Parameter(torch.zeros(num_layers, 1, hidden_size))

    def forward(self, features, hidden):
        embedded = self.embed(features.view(1, -1))
        output, hidden = self.gru(embedded.view(1, 1, -1), hidden)
        output = self.fc(output.view(1, -1))
        return output, hidden

    def init_zero_state(self):
        init_hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(DEVICE)
        return init_hidden


torch.manual_seed(RANDOM_SEED)
model = RNN(len(good_chars), EMBEDDING_DIM, HIDDEN_DIM, len(good_chars), NUM_HIDDEN)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def evaluate(model, prime_str="~", predict_len=100, temperature=0.6):
    ## based on https://github.com/spro/practical-pytorch/
    ## blob/master/char-rnn-generation/char-rnn-generation.ipynb

    hidden = model.init_zero_state()
    prime_input = char_to_tensor(prime_str, test=True)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p].to(DEVICE), hidden.to(DEVICE))
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model(inp.to(DEVICE), hidden.to(DEVICE))
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = good_chars[top_i]
        predicted += predicted_char
        inp = char_to_tensor(predicted_char, test=True)

    return predicted


def train_on_one(text):
    global model
    global queue_true_size
    for i in range(10):
        hidden = model.init_zero_state()
        optimizer.zero_grad()
        loss = 0.0
        inputs, targets = get_train_target(text)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        for c in range(TEXT_PORTION_SIZE):
            outputs, hidden = model(inputs[c], hidden)
            loss += F.cross_entropy(outputs, targets[c].view(1))
        loss /= TEXT_PORTION_SIZE
        loss.backward()
        optimizer.step()
    queue_true_size -= 1

def preprocess_text(t):
    t = re.sub(r"\n+", r"\n",t)
    return t

def preprocess_line(t):
    t = ''.join([c for c in t if c in good_chars])
    return t

def add_to_queue(text):
    global training_active_thread
    global queue_true_size
    for line in preprocess_text(text).split('\n'):
        training_queue.put(preprocess_line(line))
        queue_true_size += 1
    if training_active_thread is None:
        training_active_thread = threading.Thread(target=training_thread)
        training_active_thread.start()


def training_thread():
    global training_active_thread
    while True:
        if not training_queue.empty():
            string = training_queue.get()
            train_on_one(string)
        else:
            training_active_thread = None
            return


def inference_one():
    with torch.set_grad_enabled(False):
        return (
            evaluate(model, "~", 200)[1:]
            .split("?")[0]
            .split("!")[0]
            .split(".")[0]
            .replace("~", "")
            .strip()
        )


simple_html_interface = r"""
<!DOCTYPE html>
<html>
    <head>
    <style type="text/css">
        body {
            text-align: center;
            font-family: Monospace;
        }

        textarea {
            outline: none;
            resize: none;
            width: 300px;
            height: 200px;
            border-style: solid;
            border-width: 2px;
            border-color: #000;
            font-size: 14pt;
            padding: 0.3em;
        }

        button {
            border-style: solid;
            font-family: Monospace;
            font-size: 18pt;
            padding: 0.3em;
            background-color: #999;
            color: #111;
        }
        
        div {
            margin-top: 0.6em;
        }
    </style>
    <script type="text/javascript">
    function train() {
        return fetch("/train", {
            method: 'POST',
            headers: {
                'Content-Type': 'text/plain',
            },
            body: document.getElementById('text').value,
        })
        .then(document.getElementById('text').value = "");
    }
    function test() {
        return fetch("/test", {
            method: 'GET',
        })
        .then(response => response.text().then(t => document.getElementById('text').value = t));
    }
    function get_state() {         
        return fetch("/state", {
            method: 'GET',
        })
        .then(response => response.json().then(t => document.getElementById('state').innerHTML = JSON.stringify(t)));
    }
    </script>
    </head>
    <body onload="setInterval(get_state, 1000)">
        <textarea id="text"></textarea>
        <div>
            <button onclick="train()">set</button>
            <button onclick="test()">get</button>
        </div>
        <div id="state"></div>
    </body>
</html>
"""


class S(BaseHTTPRequestHandler):
    def _set_response(self, content_type):
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            response = simple_html_interface
            self._set_response("text/html")
            self.wfile.write(response.encode("utf-8"))
            return
        if self.path == "/test":
            try:
                response = inference_one()
            except Exception:
                response = "[INFERENCE ERROR]"
            self._set_response("text/plain")
            self.wfile.write(response.encode("utf-8"))
            return
        if self.path == "/state":
            response = '{"queue_size": %d, "cpu": %f, "mem": %f}' % (
                queue_true_size,
                psutil.cpu_percent(),
                psutil.virtual_memory()._asdict()["percent"],
            )
            self._set_response("text/plain")
            self.wfile.write(response.encode("utf-8"))
            return
        if self.path == "/checkpoint":
            try:
                save_model()
                result = "[OK]"
            except Exception:
                result = "[CHECKPOINT ERROR]"
            self._set_response("text/plain")
            self.wfile.write(response.encode("utf-8"))
        if self.path == "/restore":
            try:
                load_model()
                result = "[OK]"
            except Exception:
                result = "[RESTORE ERROR]"
            self._set_response("text/plain")
            self.wfile.write(response.encode("utf-8"))            

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        response = "[OK]"
        try:
            if self.path == "/train":
                add_to_queue(post_data.decode("utf-8"))
        except Exception:
            response = "[TRAINING ERROR]"
        self._set_response("text/plain")
        self.wfile.write(response.encode("utf-8"))

    def log_message(self, format, *args):
        return


def run(server_class=HTTPServer, handler_class=S, port=8080):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    try:
        print('\033[0;32mServer is running on :8080\033[0m')
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

def save_model():
    torch.save(model.state_dict(), "model.pth")

def load_model():
    global model
    model.load_state_dict(torch.load("model.pth"))


if __name__ == "__main__":

    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()