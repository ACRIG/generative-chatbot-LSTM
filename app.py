import pickle
import re
import random
import numpy as np

import torch
import torch.nn as nn

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# model_dir = '../variable_dump/261021_lstm_seq2seq_vanilla'
model_dir = './model_dir/1536_512_512_15_0.001'
device = 'cpu'

class Tokenizer():
    def __init__(self, data):
        self.data = data
        self.word2index = {}
        self.index2word = {}
        self.vocab = set()

        self.build()

    def build(self):
        for phrase in self.data:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2index['<PAD>'] = 0
        self.word2index['<UNK>'] = 1
        self.word2index['<sos>'] = 2
        self.word2index['<eos>'] = 3

        for i, word in enumerate(self.vocab):
            self.word2index[word] = i + 4

        for word, i in self.word2index.items():
            self.index2word[i] = word

    def text_to_sequence(self, text):
        sequences = []

        for word in text:
            sequences.append(self.word2index[word])

        return sequences

    def sequence_to_text(self, sequence):
        texts = []

        for token in sequence:
            texts.append(self.index2word[token])

        return texts

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N), N ==> batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (ssq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size,
                 num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape = (N) tapi kita butuh (1, N) karena decoder hanya predict 1 kata tiap predict
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape = (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape = (1, N, hidden_size)

        predictions = self.fc(outputs)
        # predictions shape = (1, N, length_vocab)

        predictions = predictions.squeeze(0)
        # predictions shape = (N, length_vocab) untuk dipassing ke loss function

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=.5):
        # source and target shape = (target_len, N)
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocabulary_size = len(answer_tokenizer.vocab) + 4

        outputs = torch.zeros(target_len, batch_size, target_vocabulary_size).to(device)

        hidden, cell = self.encoder(source)

        # ambil start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            # output shape = (N, answer_vocab_size)

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "app"
        return super().find_class(module, name)

def normalize(txt):
  txt = txt.lower()
  txt = re.sub(r"i'm", "i am", txt)
  txt = re.sub(r"he's", "he is", txt)
  txt = re.sub(r"she's", "she is", txt)
  txt = re.sub(r"that's", "that is", txt)
  txt = re.sub(r"what's", "what is", txt)
  txt = re.sub(r"where's", "where is", txt)
  txt = re.sub(r"\'ll", " will", txt)
  txt = re.sub(r"\'ve", " have", txt)
  txt = re.sub(r"\'re", " are", txt)
  txt = re.sub(r"\'d", " would", txt)
  txt = re.sub(r"won't", "will not", txt)
  txt = re.sub(r"can't", "can not", txt)
  txt = re.sub(r"a'ight", "alright", txt)
  txt = re.sub(r"n't", ' not', txt)
  return txt

def remove_non_letter(data):
  return re.sub(r'[^a-zA-Z]',' ', data)

def remove_whitespace(data):
  data = [x for x in data.split(' ') if x]
  return ' '.join(data)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)

    if len(x) > max_len:
        padded[:] = x[:max_len]

    else:
        padded[:len(x)] = x

    return padded

def respond_only(model, sentence, question, answer, device, max_length=50):
    threshold = 15

    if type(sentence) == str:
        sentence = normalize(sentence)
        sentence = remove_non_letter(sentence)
        sentence = remove_whitespace(sentence)

        tokens = [token.lower() for token in sentence.split(' ')]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, '<sos>')
    tokens.append('<eos>')

    # Go through each question token and convert to an index
    text_to_indices = []
    for token in tokens:
        if token in question.word2index.keys():
            text_to_indices.append(question.word2index[token])
        else:
            text_to_indices.append(question.word2index['<UNK>'])

    text_to_indices = pad_sequences(text_to_indices, threshold + 2)

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [answer.word2index["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == answer.word2index["<eos>"]:
            break

    answer_token = [answer.index2word[idx] for idx in outputs]
    return ' '.join(answer_token[1:-1])


with open(model_dir + '/question_tokenizer.pickle', 'rb') as handle:
    unpickler = MyCustomUnpickler(handle)
    question_tokenizer = unpickler.load()

with open(model_dir + '/answer_tokenizer.pickle', 'rb') as handle:
    unpickler = MyCustomUnpickler(handle)
    answer_tokenizer = unpickler.load()

new_model = torch.load(model_dir + '/model.pt', map_location=torch.device(device))

@app.route('/')
def hello():
    return 'Hello World'

@app.route('/bot')
def bot_ui():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(f"[DATA] {data}")
    
    question = data['question']

    answer = respond_only(new_model, str(question), question_tokenizer, answer_tokenizer, device, max_length=17)

    response = jsonify({'answer': answer})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)

