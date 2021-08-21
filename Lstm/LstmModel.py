from abc import ABC

import numpy as np
import torch
from torch import nn


# Define LSTM Neural Networks
class LstmRNN(nn.Module, ABC):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.5):
        super(LstmRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        # self.hidden = self.init_hidden()
        for p in self.lstm.parameters():  # 对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.hidden_size, requires_grad=True),
                torch.zeros(self.num_layers, batch, self.hidden_size, requires_grad=True))

    # 因为我们在backward的时候batch之间h是没有联系的，如果一直不detach不停地算内存会gg
    # 这里就利用detach可以隔断，相当于只保留了h的值没有保留之前的梯度
    def repackage_hidden(self, hid):
        if isinstance(hid, torch.Tensor):
            return hid.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in hid)
            # hidden有可能有多个变量，就像lstm有h和c，所以用上面这行代码把h和c都detach了

    def forward(self, seq, hidden_prev):
        seq = self.dropout(seq)
        lstm_out, hidden_prev = self.lstm(seq, hidden_prev)
        # seq is input, size (seq_len, batch, input_size), input_size is vector size
        # for example, in sentences, input_size is the size of a word vector
        # lstm_out is output, size (seq_len, batch, hidden_size)
        s, b, h = lstm_out.shape
        lstm_out = self.forwardCalculation(lstm_out.view(-1, self.hidden_size))
        lstm_out = lstm_out.view(-1, b, 1)
        # 为了能forward所以把三维变成两维的,上面的-1等于output_size
        return lstm_out, hidden_prev
