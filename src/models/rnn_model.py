import torch
import torch.nn as nn

from settings import *


class RNNModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(N_MELS, N_MELS)
        self.rnn1 = nn.RNN(input_size=N_MELS,
                           hidden_size=512,
                           num_layers=3,
                           batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(2*self.rnn1.hidden_size, 2*self.rnn1.hidden_size)
        self.rnn2 = nn.GRU(input_size=self.rnn1.hidden_size*2,
                           hidden_size=512,
                           num_layers=3,
                           batch_first=True, bidirectional=True)
        self.fc3 = nn.Linear(self.rnn2.hidden_size*2, self.output_size)

    def forward(self, batch, lengths, device):
        predictions = torch.zeros(
            lengths.sum().item() // FRAME_RESOLUTION, self.output_size, device=device)
        p = 0
        batch_frames = batch.unfold(1, FRAME_RESOLUTION,
                                    FRAME_RESOLUTION).transpose(2, 3)
        for i in range(batch.size(0)):
            frames = self.fc1(batch_frames[i][:lengths[i] // FRAME_RESOLUTION])
            # feature extraction
            # single frame passed as sequence into BiRNN (many-to-one)
            h01 = torch.zeros(self.rnn1.num_layers*2, frames.size(0),
                              self.rnn1.hidden_size, device=device)
            out, _ = self.rnn1(frames, h01)
            out = self.fc2(out[:, -1, :]).unsqueeze(0)
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(self.rnn1.num_layers*2, 1,
                              self.rnn2.hidden_size, device=device)
            out, _ = self.rnn2(out, h02)
            out = self.fc3(out[0])
            out_size = out.size(0)
            predictions[p:p+out_size, :] = out
            p += out_size
        return predictions
