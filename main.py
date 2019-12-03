import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import numpy as np
import time
import random
import string
import random
import re
import queue
import threading
import psutil
import glob

from http.server import BaseHTTPRequestHandler, HTTPServer

training_queue = queue.Queue()
queue_true_size = 0
training_active_thread = None

torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_PORTION_SIZE = 200

EPOCHS_PER_STRING = 1
LEARNING_RATE = 0.005
EMBEDDING_DIM = 44
HIDDEN_DIM = 44
NUM_HIDDEN = 1
DROPOUT = 0

STRLEN_REG_COEF = 0
WRDLEN_REG_COEF = 0
CHRREP_REG_COEF = 0
WRDREP_REG_COEF = 0

pretrain = False

num_iterations = 1
avg_str_len = 10
avg_word_len = 7
avg_char_repeat = 0
avg_word_repeat = 0

last_epoch_loss = 9000
losses = []
loss_std = 9000

good_chars = "~ ()!?.,;-—йцукенгшщзхъфывапролджэячсмитьбюё"

start_chars = "йцукенгшщзхфвапролджэячсмитбюё"

stopping_chars = ".!?"


def char_to_tensor(text, test=False):
    if not test:
        text = "~" + text
        if text[-1] not in stopping_chars:
            text = text + "."
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
            input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=DROPOUT
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


def evaluate(model, prime_str="~", predict_len=256, temperature=0.8):
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

def count_stat(s):
    global num_iterations
    global avg_str_len
    global avg_word_len
    global avg_char_repeat
    global avg_word_repeat
    def navg(a,v):
        return (num_iterations * a + v) / (num_iterations + 1)
    avg_str_len = navg(avg_str_len, len(s))
    avg_word_len = navg(avg_word_len, np.average([len(x) for x in s.split(' ')]))
    avg_char_repeat = navg(avg_char_repeat, count_repeats(s))
    avg_word_repeat = navg(avg_word_repeat, count_repeats(s.split(' ')))
    num_iterations += 1


def count_repeats(s):
    if len(s) <= 1: return 0
    l = s[0]
    r = 0
    for c in s[1:]:
        if c==l:
            r+=1
        l=c
    return r


def train_on_one(text, word=False):
    global model
    global queue_true_size
    global last_epoch_loss
    global losses
    global loss_std
    if not word: count_stat(text)
    for i in range(EPOCHS_PER_STRING):
        hidden = model.init_zero_state()
        optimizer.zero_grad()
        loss = 0.0
        inputs, targets = get_train_target(text)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        predicted_string = ""
        for c in range(len(text)):
            outputs, hidden = model(inputs[c], hidden)
            a, b = outputs.max(1)
            predicted_char = good_chars[np.array(b)[0]]
            predicted_string += predicted_char
            loss += F.cross_entropy(outputs, targets[c].view(1))
        loss /= len(text)
        word_len = len(predicted_string.split(' '))
        str_len = len(predicted_string)
        num_char_repeats, num_word_repeats = count_repeats(predicted_string), count_repeats(predicted_string.split(' '))
        loss += STRLEN_REG_COEF * ((str_len - avg_str_len)**2 / avg_str_len**2)
        loss += WRDLEN_REG_COEF * ((word_len - avg_word_len)**2 / avg_word_len**2)
        loss += CHRREP_REG_COEF if num_char_repeats > avg_char_repeat else 0
        loss += WRDREP_REG_COEF if num_word_repeats > avg_word_repeat else 0 
        if not word:
            last_epoch_loss = round(float(loss),5)
            losses.append(round(float(loss),5))
            loss_std = np.std(losses)
        loss.backward()
        optimizer.step()

def preprocess_text(t):
    t = t.lower()
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
    global queue_true_size
    def train_one_wrapper(string):
        if len(string)<3: return
        words = string.split(' ')
        for i in range(len(words)):
            train_on_one(string)
        for word in words:
            if len(word) > 0:
                train_on_one(word, word=True)
    while True:
        if pretrain and queue_true_size < 1000:
            pretrain100()
        if not training_queue.empty():
            string = training_queue.get()
            train_one_wrapper(string)
            queue_true_size -= 1
        else:
            training_active_thread = None
            return


def inference_one():
    with torch.set_grad_enabled(False):
        return (
            evaluate(model, "~" + random.choice(start_chars), 512)
            .split("?")[0]
            .split("!")[0]
            .split(".")[0]
            .replace("~", "")
            .strip().capitalize()
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

        #state {
            padding-top: 2em;
            width: 300px;
            margin: auto;
            word-wrap: break-word;
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
            response = '{"trained": %d, "queue_size": %d, "cpu": %f, "mem": %f, "last_epoch_loss": %f, "loss_std": %f}' % (
                num_iterations,
                queue_true_size,
                psutil.cpu_percent(),
                psutil.virtual_memory()._asdict()["percent"],
                last_epoch_loss,
                loss_std
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

def random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile, 2):
      if random.randrange(num): continue
      line = aline
    return line

def pretrain100():
    file_name = random.choice(glob.glob("./data/*.txt"))
    for i in range(5):
        with open(file_name, "r", encoding="utf-8") as file:
            add_to_queue(random_line(file))

if __name__ == "__main__":

    from sys import argv

    if len(argv) == 2:
        if argv[1] == "pretrain":
            pretrain = True
            add_to_queue('')
            run()
        else:
            run(port=int(argv[1]))
    else:
        run()