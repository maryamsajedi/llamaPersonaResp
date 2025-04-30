import random
import pandas as pd
import numpy as np
import torch

from datetime import datetime
from torch.autograd import Variable
from tqdm import tqdm
from torch import optim, nn

from torch.nn import init

random.seed(7)
path = '/source'
TrainSet= pd.read_pickle(f"{path}/final_train.pkl")
DevelopmentSet = pd.read_pickle(f"{path}/final_dev.pkl")
TestSet = pd.read_pickle(f"{path}/final_test.pkl")
def get_batch_train(batch_num, batch_size):
    start = batch_num * batch_size
    return TrainSet[start:start+batch_size]

def get_batch_Dev(dev_batch_num, dev_batch_size):
    start = dev_batch_num * dev_batch_size
    return DevelopmentSet[start:start+dev_batch_size]

def indexesFromSentence(sentence, MAX_LENGTH):
    inp = sentence.split()
    result = list(map(lambda k: vocab.get(k, 1), inp))[-MAX_LENGTH:] #1= oov
    if len(result) < MAX_LENGTH:
        result = [0]*(MAX_LENGTH - len(result)) + result
    return result

def process_train(row):
    context ,response ,label = row
    context = indexesFromSentence(context, MAX_LENGTH)
    response = indexesFromSentence(response,MAX_LENGTH)
    label = int(label)
    return context,response,label

def process_dev(row):
    context ,response , label = row
    context = indexesFromSentence(context, MAX_LENGTH)
    responseCandidate = indexesFromSentence(response,MAX_LENGTH)
    label = int(label)
    return context,responseCandidate,label

def load_glove_embeddings(filename=f'{path}/glove.6B.100d.txt'):
    lines = open(filename).readlines()
    embeddings = {}
    for line in lines:
        word = line.split()[0]
        embedding = list(map(float, line.split()[1:]))
        if word in vocab: embeddings[vocab[word]] = embedding
    return embeddings


PAD_token = 0
oov = 1
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"<oov>":1}
        self.word2count = {"<oov>":1}
        self.index2word = {PAD_token: "PAD", oov:"<oov>" }
        self.num_words  = 2  # Count SOS, EOS, PAD
    def addUtternace(self, Utterance):
        for word in Utterance.split(' '):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

WikiVoc= Voc('WikiResponseSelection')
print("Reading %s triples" % len(TrainSet))
print("Counting words...")
for sample in tqdm(TrainSet):
    WikiVoc.addUtternace(sample[0])
    WikiVoc.addUtternace(sample[1])
vocab = WikiVoc.word2index
print("\n")
print(WikiVoc.num_words, 'words in Wiki Dialogue Dataset')


dtype = torch.cuda.FloatTensor

class Encoder(nn.Module):
    def __init__( self, input_size, hidden_size, vocab_size, num_layers=1,
                 dropout=0, bidirectional=True, rnn_type='gru'):
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, input_size, sparse=False, padding_idx=0)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=num_layers,
                              dropout=dropout, bidirectional=bidirectional, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, self.hidden_size, num_layers=num_layers,
                               dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.init_weights()

    def forward(self, inps):
        embs = self.embedding(inps)
        outputs, hiddens = self.rnn(embs)
        return outputs, hiddens

    def init_weights(self):
        init.orthogonal_(self.rnn.weight_ih_l0)
        init.uniform_(self.rnn.weight_hh_l0, a=-0.01, b=0.01)
        glove_embeddings = load_glove_embeddings()
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25)
        for k,v in glove_embeddings.items():
            embedding_weights[k] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0]*self.input_size)
        del self.embedding.weight
        self.embedding.weight = nn.Parameter(embedding_weights)

class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        h_size = self.encoder.hidden_size * self.encoder.num_directions
        M = torch.FloatTensor(h_size, h_size)
        init.normal_(M)
        self.M = nn.Parameter( M, requires_grad=True)

    def forward(self, contexts, responses):
        context_os, context_hs = self.encoder(contexts)
        response_os, response_hs = self.encoder(responses)
        if self.encoder.rnn_type == 'lstm':
            context_hs = context_hs[0]
            response_hs = response_hs[0]
        results = []
        response_encodings = []
        h_size = self.encoder.hidden_size * self.encoder.num_directions
        for i in range(len(context_hs[0])):
            context_h = context_os[i][-1].view(1, h_size)
            response_h = response_os[i][-1].view(h_size, 1)
            ans = torch.mm(torch.mm(context_h, self.M), response_h)[0][0]
            results.append(torch.sigmoid(ans))
            response_encodings.append(response_h)
        results = torch.stack(results)
        return results, response_encodings
    

class EarlyStopping():
    def __init__(self, min_delta = 0, patience = 0):
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf
        self.stop_training = False
    def on_epoch_end(self, epoch, current_value):
        if np.greater(self.best, (current_value - self.min_delta)):
            self.best = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
        return self.stop_training
    

def coreConvertor(batch, device='cuda'):
    cs = []
    rs = []
    ys = []
    for c,r,y in batch:
        cs.append(torch.LongTensor(c).to(device))
        rs.append(torch.LongTensor(r).to(device))
        ys.append(torch.FloatTensor([y]).to(device))
    cs = Variable(torch.stack(cs, 0))
    rs = Variable(torch.stack(rs, 0))
    ys = Variable(torch.stack(ys, 0))
    return cs , rs , ys


MAX_LENGTH = 300
num_layers = 1
dropout = 0
input_size= 100
hidden_size=300
vocab_size=WikiVoc.num_words
bidirectional=True
rnn_type='lstm'
learning_rate = 0.001
batch_size = 256
num_batches = int(len(TrainSet) / batch_size)
num_epochs = 2
dev_batch_size = 128
num_dev_batches = int(len(DevelopmentSet) / dev_batch_size)

encoder_model = Encoder(input_size, hidden_size, vocab_size , num_layers, dropout, bidirectional, rnn_type)
encoder_model.cuda()            

model = DualEncoder(encoder_model)
model.cuda()

loss_fn = torch.nn.BCELoss() 
loss_fn.cuda()

optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
all_early_stopping = EarlyStopping(patience = 3)
all_early_stopping.stop_training =False
print("Training Started...")
for epoch in range(1,num_epochs):
    if all_early_stopping.stop_training:
        print ("Reached Early Stopping Patience at Epoch {}".format(epoch))
    print ("Training on Epoch {}".format(epoch))
    random.shuffle(TrainSet)
    for batch_num in tqdm(range(num_batches)):
        batch = get_batch_train(batch_num, batch_size)
        batch = list(map(process_train, batch))
        cs , rs , ys = coreConvertor(batch, device='cuda')
        y_preds, responses = model(cs, rs)
        y_preds=y_preds.reshape([batch_size, 1])
        loss = loss_fn(y_preds, ys)
        loss.backward()
        optimizer.step()
        del loss, batch
    print('Evaluating on Developement Set')
    LossDevList = []
    for dev_batch_num in range(num_dev_batches):
        dev = list(map(process_dev, get_batch_Dev(dev_batch_num, dev_batch_size)))
        csD , rsD , ysD = coreConvertor(dev, device='cuda')

        y_predsD, responsesD = model(csD, rsD)
        y_predsD=y_predsD.reshape([dev_batch_size, 1])
        lossD = loss_fn(y_predsD, ysD)
        LossDevList.append(round(lossD.tolist(),5))
    lossOnDev = sum(LossDevList)/len(LossDevList)
    print("Obtained loss on Development Set after {} epoch: ".format(epoch),round(lossOnDev,5))
    all_early_stopping.on_epoch_end(epoch = (epoch + 1), current_value = round(lossOnDev,5))
    if all_early_stopping.wait == 0:
        bestModel = model
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        torch.save(bestModel.state_dict(), f"{path}/SiameseRNN{rnn_type} {current_time}.pt")
    del lossD , LossDevList