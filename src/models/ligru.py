import torch
import torch.nn as nn
import speechbrain as sb

from settings import *


class LiGRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.gru = sb.nnet.RNN.LiGRU(input_shape=[64, 500, 89],
                           hidden_size=512,
                           num_layers=3,
                           nonlinearity='tanh',
                           normalization='None',
                           bidirectional=True)
        self.gru = torch.jit.script(self.gru)
        self.fc = nn.Linear(self.gru.hidden_size*2, self.output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch, lengths):
        out, _ = self.gru(batch)
        out = self.fc(out)
        out = self.dropout(out)
        return out
        
