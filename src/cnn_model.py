import torch
import torch.nn as nn
import torch.nn.functional as F

from settings import *


class CNNModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

        self.c1 = nn.Conv2d(1, 8, kernel_size=2, stride=1)
        self.c2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.c3 = nn.Conv2d(16, 4, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*9*28, 128)

        self.num_layers = 2
        self.hidden_size = 256
        self.rnn_input_size = 128
        self.rnn2 = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        seq_len = x.size(1)
        p = 0
        x = x.view(x.size(0)*x.size(1), 1, x.size(2), x.size(3))
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.c3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        predictions = torch.zeros(
            lengths.sum().item(), self.output_size, device=CUDA0)

        k = 0
        for i in range(batch_size):
            y = x[k:k+lengths[i]]
            y = y.view(y.size(0), 1, y.size(1))
            h0 = torch.zeros(self.num_layers*2, 1,
                             self.hidden_size, device=CUDA0)
            out, _ = self.rnn2(y, h0)
            for j in range(lengths[i]):
                predictions[p] = self.fc(out[j][0])
                p += 1
            k += seq_len

        return predictions
