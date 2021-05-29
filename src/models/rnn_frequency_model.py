import torch
import torch.nn as nn

from settings import *


class RNNFrequencyModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(N_MELS, N_MELS)
        self.rnns1 = [
            nn.RNN(input_size=20,
                   hidden_size=256,
                   num_layers=2,
                   batch_first=True, bidirectional=True).cuda()
            for _ in range(4)
        ]
        self.rnn1_output_size = self.rnns1[0].hidden_size*2*len(self.rnns1)
        self.rnn2 = nn.GRU(input_size=self.rnn1_output_size,
                           hidden_size=512,
                           num_layers=3,
                           batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(self.rnn2.hidden_size*2, self.rnn2.hidden_size*2)
        self.fc = nn.Linear(self.rnn2.hidden_size*2, self.output_size)

    def forward(self, batch, lengths, device):
        
        batch = self.fc1(batch)
        batch = batch.unfold(2, 20, 20)
        out2 = torch.zeros(batch.size(0), batch.size(1), self.rnn1_output_size, device=device)
        for i in range(4):
            input = batch[:, :, i, :]
            h01 = torch.zeros(self.rnns1[0].num_layers*2,
                              batch.size(0),
                              self.rnns1[0].hidden_size, device=device)
            out, _ = self.rnns1[i](input, h01)
            out2[:, :, i*self.rnns1[0].hidden_size*2:(i+1)*self.rnns1[0].hidden_size*2] = out
        h0 = torch.zeros(self.rnn2.num_layers*2, batch.size(0),
                         self.rnn2.hidden_size, device=device)
        out, _ = self.rnn2(out2, h0)
        out = self.fc(self.fc2(out))
        predictions = torch.zeros(
            lengths.sum().item(), self.output_size, device=device)
        p = 0
        for i in range(batch.size(0)):
            predictions[p:p+lengths[i], :] = out[i][:lengths[i]]
            p += lengths[i]
        
        return predictions
