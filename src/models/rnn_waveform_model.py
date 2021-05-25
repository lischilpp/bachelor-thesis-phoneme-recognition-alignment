import torch
import torch.nn as nn

from settings import *


class RNNWaveformModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.num_layers1 = 2
        self.num_layers2 = 3
        self.hidden_size1 = 512
        self.hidden_size2 = 512
        self.rnn1 = nn.RNN(16, self.hidden_size1,
                           self.num_layers1, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(self.hidden_size1*2, self.hidden_size2,
                           self.num_layers1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size2*2, self.output_size)

    def forward(self, batch, lengths, device):
        counts = (lengths - 400) // 160 + 1
        predictions = torch.zeros(torch.sum(counts), self.output_size, device=device)
        p = 0
        for i, waveform in enumerate(batch):
            input = waveform[:lengths[i]].unfold(0, 400, 160).unfold(1, 16, 16)
            h01 = torch.zeros(self.num_layers1*2, input.size(0),
                                self.hidden_size1, device=device)
            out, _ = self.rnn1(input, h01)
            out = out[:, -1, :]
            out = out.unsqueeze(0)
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(4, 1,
                                self.hidden_size2, device=device)
            out2, _ = self.rnn2(out, h02)
            for j in range(counts[i]):
                predictions[p] = self.fc(out2[0][j])
                p += 1
        return predictions
