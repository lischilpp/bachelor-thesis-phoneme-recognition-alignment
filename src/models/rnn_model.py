import torch
import torch.nn as nn

from settings import *


class RNNModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.num_layers1 = 2
        self.num_layers2 = 3
        self.hidden_size1 = 512
        self.hidden_size2 = 512
        self.fc1 = nn.Linear(N_MELS, N_MELS)
        self.fc2 = nn.Linear(2*self.hidden_size1, 2*self.hidden_size1)
        self.rnn1 = nn.RNN(N_MELS, self.hidden_size1,
                           self.num_layers1, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(self.hidden_size1*2, self.hidden_size2,
                           self.num_layers1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size2*2, self.output_size)

    def forward(self, batch, lengths, device):
        predictions = torch.zeros(
            lengths.sum().item() // FRAME_RESOLUTION, self.output_size, device=device)
        p = 0
        batch_frames = batch.unfold(1, FRAME_RESOLUTION,
                                    FRAME_RESOLUTION).transpose(2, 3)
        for i in range(batch.size(0)):
            frames = batch_frames[i][:lengths[i] // FRAME_RESOLUTION]
            # feature extraction
            # single frame passed as sequence into BiRNN (many-to-one)
            h01 = torch.zeros(self.num_layers1*2, frames.size(0),
                              self.hidden_size1, device=device)
            out, _ = self.rnn1(frames, h01)
            out = out[:, -1, :].unsqueeze(0)
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(self.num_layers1*2, 1,
                              self.hidden_size2, device=device)
            out, _ = self.rnn2(out, h02)
            out = self.fc(out[0])
            out_size = out.size(0)
            predictions[p:p+out_size, :] = out
            p += out_size
        return predictions
