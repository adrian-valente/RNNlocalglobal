import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Sigmoid
from torch.nn import Module
from torch.nn import Embedding
from torch.optim import SGD,Adam
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from numpy import vstack
from torch.utils.data import random_split
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import matplotlib

# model definition
class RNN(Module):
    # define model elements
    # n_inputs: input dimension
    # n_hidden: number of neurons per layer
    # n_layers: number of hidden layers
    def __init__(self, n_hidden, n_layers, input_dim):
        super(RNN, self).__init__()

        # hidden layers
        #self.embed = Embedding(input_dim, embedding_dim)
        self.hidden = {}
        self.act = {}
        for i in range(n_layers):
            if i==0:
                n_in = input_dim + n_hidden # embedding_dim + n_hidden
            else:
                n_in = n_hidden * 2
            # input to hidden layer
            self.hidden[i] = Linear(n_in, n_hidden)
            kaiming_uniform_(self.hidden[i].weight, nonlinearity='relu')
            # non-linearity
            self.act[i] = Tanh() # or ReLu

        # output
        self.out = Linear(n_hidden,input_dim) # dimension of output is 2
        xavier_uniform_(self.out.weight)
        #self.actout = Sigmoid()

    # forward propagate input
    def forward(self, X, hidden_layer):
        n_layers = len(hidden_layer)
        #idx = torch.argmax(X,-1)
        #embedding = self.embed(idx)
        for i in range(n_layers):
            if i == 0:
                # combine input with previous hidden
                combined = torch.cat((X, hidden_layer[i]), 1)
            else:
                # combine previous hidden with hidden
                combined = torch.cat((hidden_layer[i-1], hidden_layer[i]), 1)
            # input to hidden layer
            hidden_layer[i] = self.hidden[i](combined)
            hidden_layer[i] = self.act[i](hidden_layer[i])

        ## output
        output = self.out(hidden_layer[i])
        #output = self.actout(output)

        return output, hidden_layer


# train the model
def train_model(train_x,train_y, model):
    n_layers = len(model.hidden)
    n_hidden = model.hidden[0].weight.size()[0]
    # define the optimization
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.05)

    # enumerate epochs
    batch_size = train_x[0].size()[0]
    all_loss = []
    lossdiff = -1
    epoch = 0
    while lossdiff<0:
        # enumerate batches (xx and xY batch)
        for xi,yi in zip(train_x,train_y):
            # clear the gradients
            optimizer.zero_grad()
            # initialise hidden states
            hi = {}
            for k in range(n_layers):
                hi[k] = torch.zeros((batch_size,n_hidden))
            #loss = 0
            for j in range(xi.size()[1]):
                xij = xi[:,j,:]

                # compute the model output
                yhat,hi = model(xij,hi)

                for k in range(n_layers):
                    hi[k] = hi[k].detach()
            # calculate loss on last prediction
            loss = criterion(yhat, yi)
            loss.backward(retain_graph=True)

            # update model weights
            optimizer.step()

        all_loss.append(loss.item())
        if epoch>5:
            lossdiff = all_loss[-1]-np.mean(all_loss[-5:-1] )
        epoch +=1
    return all_loss

# evaluate the model
def evaluate_model(test_x,test_y, model):

    # this function returns the average accuracy on all input test sequences. One value per sequence
    # For now, it skips the first  items


    n_layers = len(model.hidden)
    n_hidden = model.hidden[0].weight.size()[0]

    batch_size = test_x[0].size()[0]
    num_seq = len(test_x)
    acc = []
    rand_acc = []
    for xi,yi in zip(test_x,test_y):

        # initialize hidden state
        hi = {}
        for k in range(n_layers):
            hi[k] = torch.zeros((batch_size,n_hidden))

        for j in range(xi.size()[1]):
            xij = xi[:,j,:]
            # compute the model output
            yhat,hi = model(xij,hi)

        # round to class values: this is if output is classification
        #yhat = np.argmax(yhat.detach().numpy(),-1)
        # store
        #acc.append(accuracy_score(yhat.detach().numpy(),yi))


        # if regression:
        yhat = yhat.detach().numpy()
        acc.append(np.mean((yhat-yi.numpy())**2))

    return acc


def predict(x, model):
    """
    :param x: list of tensors (ntrials x ntimesteps x ndim)
    :param model: An RNN instance
    :return:
    """
    n_layers = len(model.hidden)
    n_hidden = model.hidden[0].weight.size()[0]
    batch_size = x[0].size()[0]
    y = []
    h = []
    c = []
    for xi in x:
        # Initialize layers
        hi = {}
        for k in range(n_layers):
            hi[k] = torch.zeros((batch_size, n_hidden))
        # Loop over timesteps
        for j in range(xi.size()[1]):
            xij = xi[:, j, :]
            # compute the model output
            yhat, hi = model(xij, hi)

        yhat = yhat.detach().numpy()   # predicted stimulus
        #ci = np.round(yhat) != xij.detach().numpy()
        ##print(np.sum(ci,1))
        #ci =  ((np.sum(ci,1)==0)==False).astype(int) # predicted sameness 1=change, 0=same
        y.append(yhat)
        #c.append(ci)
        h.append(hi)
    return y, h #,c


def predict_single(x, model):
    n_layers = len(model.hidden)
    n_hidden = model.hidden[0].weight.size()[0]
    batch_size = x.shape[0]
    y = []
    h = []
    # Initialize hidden states
    hi = torch.zeros((n_layers, batch_size, n_hidden))
    # Loop over timesteps
    for j in range(x.shape[1]):
        xij = x[:, j, :]
        # compute the model output
        yhat, hi = model(xij, hi)
        y.append(yhat.detach().numpy().squeeze())
        h.append(hi.detach().numpy().squeeze().copy())
    y = np.stack(y)
    h = np.stack(h)
    return y, h


def unique_pairs(dictionary):
    '''
    create matrix with unique pairs of (one-hot) vectors in dictionary

    dictionary: N x dimensionality; matrix of N (one-hot) vectors
    '''
    vecpairs = []
    for i in range(dictionary.shape[0]):
        for j in range(dictionary.shape[0]):
            if i!=j:
                vecpairs.append(np.vstack((dictionary[j,],dictionary[i,])))
    vecpairs = np.array(vecpairs)
    return vecpairs


def unique_pairs_sequence(seq, repeats, pairs):
    """
    Create a structured sequence such as 010101, replacing 0 and 1 with (one-hot) vectors from dictionary

    seq: list of sequence structures
    repeats: repeats per sequence chunk (e.g. 4 = 4xsize of sequence chunk)
    pairs: matrix of (one-hot) vectors, shape: dic size X input size

    returns:
    """
    nseq = len(seq)
    x = np.zeros((nseq))
    input_size = pairs.shape[2]
    batch_size = pairs.shape[0]
    all_seq = []
    seq_len = []
    for s in seq:

        # pick value from dictionary:
        a = []
        b = []
        for i in range(batch_size):
            a.append(pairs[i,1,:]) # X
            b.append(pairs[i,0,:]) # Y
        a = np.array(a)
        b = np.array(b)
        # create sequence
        sequence = []
        for i,element in enumerate(s):
            if element==0:
                sequence.append(a)
            elif element==1:
                sequence.append(b)
        sequence = np.array(sequence)

        sequence = np.swapaxes(sequence,0,1)
        # add zero state
        #sequence = np.concatenate((sequence,np.zeros((batch_size,1,input_size))),1)
        out_sequence = sequence
        for r in range(repeats-1):
            # stack the sequence repeat times
            out_sequence = np.concatenate((out_sequence,sequence),1)
        all_seq.append(out_sequence)

    return all_seq


def angle_vecs(v1, v2):
    return np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
