import torch
import torch.nn as nn

from settings import *


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=N_MELS,#N_MELS+N_CEPS,
                           hidden_size=512,
                           num_layers=8,
                           batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, batch, _lengths, _device):
        out, _ = self.rnn(batch)
        out = self.fc(out)
        out = self.dropout(out)
        return out