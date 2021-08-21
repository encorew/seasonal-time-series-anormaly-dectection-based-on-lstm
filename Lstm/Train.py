import torch
from torch import nn

from Lstm.LstmModel import LstmRNN
import numpy as np
import matplotlib.pyplot as plt

SERIES_DIM = 1
BATCH_SIZE = 1

GRAD_CLIP = 0.5


def gen_train_data(seq, window_len):
    train_dat = list()
    seq_len = len(seq)
    for i in range(seq_len - window_len):
        in_data = seq[i:i + window_len]
        # out_data = seq[i + 1:i + window_len + 1]
        out_data = seq[i + 1:i + window_len + 1]
        # print(out_data)
        train_dat.append((in_data, out_data))
    # print(len(train_dat))
    return train_dat


def iter_train(model, hidden, train_data, loss_function, optimizer, epoch, total_loss):
    i = 0
    for train_x, train_y in train_data:
        train_x = train_x.view(-1, BATCH_SIZE, SERIES_DIM)
        train_y = train_y.view(-1, BATCH_SIZE, SERIES_DIM)
        out, hidden = model(train_x, hidden)
        hidden = model.repackage_hidden(hidden)
        loss = loss_function(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 1:
            print("epoch:{} iter:{}".format(epoch, i))
            print("---------loss:{} total_loss{}".format(loss.item(), total_loss))
        i += 1
        if loss.item() <= 1e-13:
            # print("small")
            return False, total_loss
    return True, total_loss


def train(model, dataset, window_size, epoch, learning_rate):
    train_data = gen_train_data(torch.tensor(dataset).float(), window_size)
    print(len(train_data))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    hidden = model.init_hidden(BATCH_SIZE)
    min_epoch_loss = 1000
    # best_parameters = model.state_dict()
    # best_parameters用来记录最好的模型参数
    for _ in range(epoch):
        model.train()
        total_loss = 0
        print("==> epoch{}".format(_))
        continue_train, total_loss = iter_train(model, hidden, train_data, loss_function, optimizer, _, total_loss)
        print("last epoch loss >{}<".format(total_loss))
        if total_loss < min_epoch_loss:
            min_epoch_loss = total_loss
            best_parameters = model.state_dict()
        if not continue_train:
            print("train over!")
            break
    return model.state_dict()
