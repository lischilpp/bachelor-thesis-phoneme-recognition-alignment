import torch
import torch.nn as nn

from settings import *


class GRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=N_MELS,
                           hidden_size=512,
                           num_layers=3,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.rnn.hidden_size*2, self.output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch, lengths):
        out, _ = self.rnn(batch)
        out = self.fc(out)
        out = self.dropout(out)
        return out