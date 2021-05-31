import torch
import torch.nn as nn

from settings import *


class PhonemeBoundaryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=48,
                           hidden_size=256,
                           num_layers=3,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.rnn.hidden_size*2, 1)

    def forward(self, batch, lengths, device):
        h0 = torch.zeros(self.rnn.num_layers*2, batch.size(0),
                         self.rnn.hidden_size, device=device)
        out, _ = self.rnn(batch, h0)
        out = self.fc(out)
        predictions = torch.zeros(lengths.sum().item(), device=device)
        p = 0
        for i in range(batch.size(0)):
            predictions[p:p+lengths[i]] = torch.round(out[i][:lengths[i]]).flatten()
            p += lengths[i]
        
        return predictions