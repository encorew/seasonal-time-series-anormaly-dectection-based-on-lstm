import torch
from torch import nn
from Lstm.LstmModel import LstmRNN
import numpy as np
import matplotlib.pyplot as plt


def gen_train_data(seq, window_len):
    train_dat = list()
    seq_len = len(seq)
    print(seq_len)
    for i in range(seq_len - window_len):
        in_data = seq[i:i + window_len]
        # out_data = seq[i + 1:i + window_len + 1]
        out_data = seq[i + window_len + 1]
        # print(out_data)
        train_dat.append((in_data, out_data))
    # print(train_dat)
    return train_dat


def train(model, hidden, train_data, loss_function, optimizer):
    for train_x, train_y in train_data:
        train_x = train_x.view(-1, BATCH_SIZE, SERIES_DIM)
        train_y = train_y.view(-1, BATCH_SIZE, SERIES_DIM)
        out, hidden = model(train_x, hidden)
        loss = loss_function(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


data_len = 2000

TOTAL_DATA_NUM = 100
SERIES_DIM = 1
BATCH_SIZE = 1
HIDDEN_SIZE = 16
MAX_EPOCHS = 1000
GRAD_CLIP = 5.0
SIN_RANGE = 20
# TRAINED = False
TRAINED = True

model = LstmRNN(input_size=1, hidden_size=HIDDEN_SIZE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

hidden = model.init_hidden(BATCH_SIZE)

# generate data
if not TRAINED:
    time_step = np.linspace(0, SIN_RANGE, TOTAL_DATA_NUM)
    sin_t = np.sin(time_step)
    # test_data = torch.tensor(sin_t[train_index:]).float()
    # train_data_pair = gen_data_pair(train_data, 49)
    train_data = torch.tensor(sin_t).float()
    train_data_pair = gen_data_pair(train_data, 50)
    # 以上步骤生成(x,y)数据对
    # initialize model

    for epoch in range(MAX_EPOCHS):
        model.train()
        i = 0
        end = False
        for train_x, train_y in train_data_pair:
            train_x = train_x.view(-1, BATCH_SIZE, 1)
            train_y = train_y.view(-1, BATCH_SIZE, 1)
            out, hidden = model(train_x, hidden)
            hidden = model.repackage_hidden(hidden)

            loss = loss_function(out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch:", epoch, " iter:", i, " loss=", loss.item())
            if loss.item() < 1e-4:
                end = True
                break
            i += 1
        if end:
            break
    torch.save(model.state_dict(), 'newPlay_lstm_params.pkl')
else:
    model.load_state_dict(torch.load('window99num100.pkl'))

start = 0
time_steps = np.linspace(start, start + SIN_RANGE, TOTAL_DATA_NUM)
data = np.sin(time_steps)
data = data.reshape(TOTAL_DATA_NUM, 1)
x = torch.tensor(data[:-1]).float().view(TOTAL_DATA_NUM - 1, 1, 1)
y = torch.tensor(data[1:]).float().view(TOTAL_DATA_NUM - 1, 1, 1)

predictions = []

input = x[0:99, :, :]  # 取seq_len里面第0号数据
# input = input.view(1, 1, 1)  # input：[1,1,1]
# for i in range(x.shape[0]):  # 迭代seq_len次
#
#     pred, hidden = model(input, hidden)
#     # input = x[i, :, :].view(1, 1, 1)
#     input = pred
#     # 预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
#     predictions.append(pred.detach().numpy().ravel()[0])
# print(predictions)
input = input.view(-1, 1, 1)
pred, hidden = model(input, hidden)
predictions = pred.detach().numpy().ravel()
print(predictions.shape)
x = x.data.numpy()
y = y.data.numpy()
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[:-1], x.ravel(), c='r')  # x值
plt.scatter(time_steps[1:], y.ravel(), c='y')  # y值
plt.scatter(time_steps[1:100], predictions, c='b')  # y的预测值
plt.show()
