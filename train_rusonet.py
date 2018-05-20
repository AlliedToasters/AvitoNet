import os
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import models
import argparse
import time
import math
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='lm.py')
parser.add_argument('-save_model', default='model.pt',
                    help="""Model filename to save""")
parser.add_argument('-load_model', default='',
                    help="""Model filename to load""")
parser.add_argument('-t', '--train', nargs='+', 
                    help='paths to training data', required=True)
parser.add_argument('-v', '--valid', nargs='+', 
                    help='paths to validation data')                    
parser.add_argument('-rnn_type', default='mlstm',
                    help='mlstm, lstm or gru')
parser.add_argument('-layers', type=int, default=1,
                    help='Number of layers in the rnn')
parser.add_argument('-rnn_size', type=int, default=4096,
                    help='Size of hidden states')
parser.add_argument('-embed_size', type=int, default=128,
                    help='Size of embeddings')
parser.add_argument('-seq_length', type=int, default=20,
                    help="Maximum sequence length")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Starting learning rate.""")
parser.add_argument('-dropout', type=float, default=0.1,
                    help='Dropout probability.')
parser.add_argument('-param_init', type=float, default=0.05,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-clip', type=float, default=5,
                    help="""Clip gradients at this value.""")
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')   
# GPU
parser.add_argument('-cuda', type=str, default='y',
                    help="Use CUDA (y/n")

opt = parser.parse_args()

    
    
def format_post(row):
    """Takes a row and formats it in a consistent manner
    for training and feature extraction."""
    #all fields denoted by newline
    result = ''
    result += row.region + '\n'
    result += row.city + '\n'
    result += row.parent_category_name + '\n'
    result += row.category_name + '\n'
    #parameters are seperated by tab bytes
    if not pd.isnull(row.param_1):
        result += str(row.param_1) + '\t'
    if not pd.isnull(row.param_2):
        result += str(row.param_2) + '\t'
    if not pd.isnull(row.param_3):
        result += str(row.param_3) + '\t'
    result += '\n'
    result += str(row.title)
    description = row.description
    if type(description) != type(str):
        description = str(description)
    #remove newlines from description body
    description = description.replace('\n', ' ')
    result += description
    return result

def format_post(row):
    """Takes a row and formats it in a consistent manner
    for training and feature extraction."""
    #new post denoted by newline
    result = '\n'
    #all fields denoted by newline
    result += str(row.region) + '\n'
    result += str(row.city) + '\n'
    result += str(row.parent_category_name) + '\n'
    result += str(row.category_name) + '\n'
    #parameters are seperated by tab bytes
    if not pd.isnull(row.param_1):
        result += str(row.param_1) + '\t'
    if not pd.isnull(row.param_2):
        result += str(row.param_2) + '\t'
    if not pd.isnull(row.param_3):
        result += str(row.param_3) + '\t'
    result += '\n'
    result += str(row.title)
    description = str(row.description)
    #remove newlines from description body
    description = description.replace('\n', ' ')
    result += description
    return result

def get_chunk(datagen, extra=b'', num_bytes=1344):
    """Uses a data generator to produce a formatted byte string of the
    desired length.
    """
    chunk = b''
    if len(extra) > num_bytes:
        chunk += extra[:num_bytes]
        extra = extra[num_bytes:]
        return chunk, extra
    chunk += extra
    encoded_post = b''
    remaining = 0
    while len(chunk) < num_bytes:
        slc = next(datagen)
        row = slc.iloc[0]
        encoded_post = format_post(row).encode(encoding='utf-8')
        remaining = min(len(encoded_post), num_bytes - len(chunk))
        chunk += encoded_post[:remaining]
    extra = encoded_post[remaining:]
    return chunk, extra

def init_data_gen(paths):
    """Takes csv files and produces batches."""
    gens = []
    for path in paths:
        gen = pd.read_csv(path, chunksize=1)
        gens.append(gen)
    return gens


def count_chunks(paths):
    """Generates chunks and counts them up. Returns total."""
    chunks = 0
    gens = init_data_gen(paths)
    extra = b''
    for gen in gens:
        while True:
            try:
                get_chunk(gen, extra)
                chunks += 1
            except:
                break
    return chunks

def gen_batches(paths):
    """Takes list of generators and yields batches."""
    gens = init_data_gen(paths)
    count = 0
    extra = b''
    while True:
        for gen in gens:
            while True:
                try:
                    chunk, extra = get_chunk(gen, extra)
                    yield chunk
                except StopIteration:
                    break
        yield 'stop'
        gens = init_data_gen(paths)
                
def tokenize(post):
        """Tokenizes a chunk of bytes"""
        tokens = len(post)
        #print('number of tokens: ', tokens)

        ids = torch.ByteTensor(tokens)
        token = 0
        for byt in post:
            ids[token] = byt
            token += 1
        #print('returning...')
        return ids

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data     

def update_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    return

def clip_gradient_coeff(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

def calc_grad_norm(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    return math.sqrt(totalnorm)

def calc_grad_norms(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    norms = []
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        norms += [modulenorm]
    return norms

def clip_gradient(model, clip):
    """Clip the gradient."""
    totalnorm = 0
    for p in model.parameters():
        p.grad.data = p.grad.data.clamp(-clip,clip)


def make_cuda(state):
    if isinstance(state, tuple):
        return (state[0].cuda(), state[1].cuda())
    else:
        return state.cuda()

def copy_state(state):
    if isinstance(state, tuple):
        return (Variable(state[0].data), Variable(state[1].data))
    else:
        return Variable(state.data)    	

def plot_hists(history):
    """Plots the optimization progress."""
    fig = plt.figure();
    plt.plot(range(len(history['loss'])), history['loss'], label='train loss');
    plt.plot(range(len(history['avg_loss'])), history['avg_loss'], label='avg train loss');
    if len(history['val_loss']) > 0:
        if type(n_batch) == type(str()):
            plt.legend();
            fig.savefig('recent_plot.png');
            plt.close(fig);
            return history
        else:
            e = len(history['val_loss'])
            plt.plot(np.arange(e)*n_batch/10, history['val_loss'], label='validation_loss');
    else:
        history['val_loss'].append(history['loss'][0])
    plt.legend();
    fig.savefig('recent_plot.png');
    plt.close(fig);
    return history

def save_model(model, embed, e, title=None):
    """Saves the model."""
    checkpoint = {
            'rnn': model,
            'embed': embed,
            #'opt': opt,
            'epoch': e
        }
    if title == None:
        save_file = ('{}s_e{}s.pt'.format(save_path, e))
    else:
        save_file = title
    print('Saving to '+ save_file)
    torch.save(checkpoint, save_file)
    return

def evaluate(hist):
    hidden_init = rnn.state0(batch_size)  		
    if cuda:
        embed.cuda()
        rnn.cuda()
        hidden_init = make_cuda(hidden_init)

    loss_avg = 0
    s = -1
    while True:
        s += 1
        start = time.time()
        post = next(val_bg)
        if type(post) == type(str()):
            global nv_batch
            nv_batch = s
            break
        toks = tokenize(post)
        batchified = batchify(toks, batch_size)
        batch = Variable(batchified.narrow(0,0,TIMESTEPS+1).long())
        hidden = hidden_init
        if cuda:
            batch = batch.cuda()

        loss = 0
        for t in range(TIMESTEPS):                  
            emb = embed(batch[t])
            hidden, output = rnn(emb, hidden)
            loss += loss_fn(output, batch[t+1])

        hidden_init = copy_state(hidden)
        loss_avg = loss_avg + loss.data[0]/TIMESTEPS
        if s % 10 == 0:
            print('v %s / %s loss %.4f loss avg %.4f time %.4f' % ( s, nv_batch, loss.data[0]/TIMESTEPS, loss_avg/(s+1), time.time()-start))
    hist['val_loss'].append((loss_avg/nv_batch).item())
    return loss_avg/nv_batch, hist

def train_epoch(epoch, hist, loss_avg):
    hidden_init = rnn.state0(batch_size)    		
    if cuda:
        embed.cuda()
        rnn.cuda()
        hidden_init = make_cuda(hidden_init)
    
    s=-1
    
    while True:
        s += 1
        embed_optimizer.zero_grad()
        rnn_optimizer.zero_grad()
        start = time.time()
        post = next(train_bg)
        if type(post) == type(str()):
            global n_batch
            n_batch = s
            break
        toks = tokenize(post)
        batchified = batchify(toks, batch_size)
        batch = Variable(batchified.narrow(0,0,TIMESTEPS+1).long())
        hidden = hidden_init
        if cuda:
            batch = batch.cuda()
        loss = 0
        for t in range(TIMESTEPS):                  
            emb = embed(batch[t])
            hidden, output = rnn(emb, hidden)
            loss += loss_fn(output, batch[t+1])


        loss.backward()

        hidden_init = copy_state(hidden)
        gn =calc_grad_norm(rnn)
        clip_gradient(rnn, clip)
        clip_gradient(embed, clip)
        embed_optimizer.step()
        rnn_optimizer.step()
        if loss_avg == 0:
            loss_avg = loss.data[0]/TIMESTEPS
        else:
            loss_avg = .99*loss_avg + .01*loss.data[0]/TIMESTEPS
        if s % 10 == 0:
            hist['loss'].append(loss.data[0]/TIMESTEPS)
            hist['avg_loss'].append(loss_avg)
            if s % 100 == 0:
                if s % 1000 == 0:
                    save_model(rnn, embed, e, title='latest.pt')
                hist = plot_hists(hist)
            print('e%s %s / %s loss %.4f loss avg %.4f time %.4f grad_norm %.4f' % (epoch, s, n_batch, loss.data[0]/TIMESTEPS, loss_avg, time.time()-start, gn))
    return hist, loss_avg


if __name__ in "__main__":
    
    save_path = opt.save_model
    load_model = opt.load_model
    train_paths = opt.train
    val_paths = opt.valid

    layers = opt.layers
    hidden_size = opt.rnn_size
    embed_size = opt.embed_size
    seq_length = opt.seq_length
    batch_size = opt.batch_size
    lr = opt.learning_rate
    learning_rate = lr
    dropout = opt.dropout
    param_init = opt.param_init
    clip = opt.clip
    seed = opt.seed

    if opt.cuda == 'y':
        cuda = True
    else:
        cuda = False
    
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    train_bg = gen_batches(train_paths)
    val_bg = gen_batches(val_paths)
    
    n_batch = 'unknown'
    nv_batch = 'unknown'
    
    input_size = embed_size
    data_size = 256
    TIMESTEPS = seq_length

    if len(load_model)>0:
        checkpoint = torch.load(load_model)
        embed = checkpoint['embed']
        rnn = checkpoint['rnn']
    else:
        embed = nn.Embedding(data_size, input_size)
        rnn = models.StackedLSTM(models.mLSTM, layers, input_size, hidden_size, data_size, dropout)

    loss_fn = nn.CrossEntropyLoss() 

    nParams = sum([p.nelement() for p in rnn.parameters()])
    print('* number of parameters: %d' % nParams)
    
    print('total number of train batches: ', n_batch)
    print('total number of validation batches: ', nv_batch)

    embed_optimizer = optim.SGD(embed.parameters(), lr=learning_rate)
    rnn_optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    
    history = dict()
    history['loss'] = []
    history['avg_loss'] = []
    history['val_loss'] = []
    e = 0
    avg_loss = 0
    for e in range(10):
        e = e
        try:
            history, avg_loss = train_epoch(e, history, avg_loss)
        except KeyboardInterrupt:
            print('Exiting from training early')
        loss_avg, history = evaluate(history)
        history = plot_hists(history)
        save_model(rnn, embed, e)
        learning_rate *= 0.7
        update_lr(rnn_optimizer, learning_rate)
        update_lr(embed_optimizer, learning_rate)