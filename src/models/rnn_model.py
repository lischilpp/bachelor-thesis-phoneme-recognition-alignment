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
        self.fc1 = nn.Linear(N_MELS, N_MELS)
        self.fc2 = nn.Linear(2*self.hidden_size1, 2*self.hidden_size1)
        self.rnn1 = nn.RNN(N_MELS, self.hidden_size1,
                           self.num_layers1, batch_first=True, bidirectional=True, dropout=0.5)
        self.rnn2 = nn.GRU(self.hidden_size1*2, self.hidden_size2,
                           self.num_layers1, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size2*2, self.output_size)

    def forward(self, batch, lengths):
        predictions = torch.zeros(
            lengths.sum().item() // FRAME_RESOLUTION, self.output_size, device=CUDA0)
        p = 0
        for i in range(batch.size(0)):
            specgrams = batch[i][:lengths[i]]
            specgrams = self.fc1(specgrams)
            frames = specgrams.unfold(0, FRAME_RESOLUTION,
                                      FRAME_RESOLUTION).transpose(1, 2)
            # feature extraction
            # single frame passed as sequence into BiRNN (many-to-one)
            h01 = torch.zeros(self.num_layers1*2, frames.size(0),
                              self.hidden_size1, device=CUDA0)
            out, _ = self.rnn1(frames, h01)
            out = out[:, -1, :]
            out2 = out.unsqueeze(0)
            out2 = self.fc2(out2)
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(self.num_layers2*2, 1,
                              self.hidden_size2, device=CUDA0)
            out2, _ = self.rnn2(out2, h02)
            for j in range(lengths[i] // FRAME_RESOLUTION):
                predictions[p] = self.fc(out2[0][j])
                p += 1
        return predictions
