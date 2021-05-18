import torch
import torch.nn as nn

from settings import *


class RNNModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.num_layers1 = 2
        self.num_layers2 = 2
        self.hidden_size1 = 256
        self.hidden_size2 = 256
        # self.rnn1 = nn.RNN(SPECGRAM_N_MELS, self.hidden_size1,
        #                    self.num_layers1, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(input_size=320, hidden_size=self.hidden_size2,
                           num_layers=self.num_layers1, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size2*2, self.output_size)

    def forward(self, batch, lengths):
        predictions = torch.zeros(
            lengths.sum().item(), self.output_size, device=CUDA0)
        p = 0
        batch = batch.transpose(2, 3)
        # print(batch.shape)
        batch = batch.flatten(2)
        # print(batch.shape)
        # exit()
        for i in range(batch.size(0)):
            # feature extraction
            # single frame passed as sequence into BiRNN (many-to-one)
            # h01 = torch.zeros(self.num_layers1*2, batch.size(1),
            #                   self.hidden_size1, device=CUDA0)
            # out, _ = self.rnn1(batch[i], h01)
            # out = out[:, -1, :]
            # out2 = out.unsqueeze(0)
            x = batch[i]
            x = x.view(x.size(0), 1, x.size(1))
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(self.num_layers2*2, 1,
                              self.hidden_size2, device=CUDA0)
            out2, _ = self.rnn2(x, h02)
            for j in range(lengths[i]):
                predictions[p] = self.fc(out2[j][0])
                p += 1
        return predictions
