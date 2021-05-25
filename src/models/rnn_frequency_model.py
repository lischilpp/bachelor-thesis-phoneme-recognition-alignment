import torch
import torch.nn as nn

from settings import *


class RNNFrequencyModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.num_layers1 = 2
        self.num_layers2 = 3
        self.hidden_size1 = 128
        self.hidden_size2 = 512
        self.fc1 = nn.Linear(N_MELS, N_MELS)
        self.fc2 = nn.Linear(2*self.hidden_size1, 2*self.hidden_size1)
        self.rnns1 = [
            nn.RNN(16, self.hidden_size1,
                   self.num_layers1, batch_first=True, bidirectional=True).cuda()
            for _ in range(4)
        ]
        self.rnn2 = nn.GRU(1024, self.hidden_size2,
                           self.num_layers2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size2*2, self.output_size)

    def forward(self, batch, lengths, device):
        predictions = torch.zeros(
            lengths.sum().item() // FRAME_RESOLUTION, self.output_size, device=device)
        p = 0
        for i in range(batch.size(0)):
            fbank = batch[i][:lengths[i]]
            frames = fbank.unfold(0, FRAME_RESOLUTION,
                                  FRAME_RESOLUTION).transpose(1, 2)
            frames = frames.unfold(2, 16, 16)

            out2 = torch.zeros(frames.size(0), 1024, device=device)

            for j in range(4):
                h01 = torch.zeros(self.num_layers1*2, frames.size(0),
                              self.hidden_size1, device=device)
                input = frames[:, :, j, :]
                out, _ = self.rnns1[j](input, h01)
                out = out[:, -1, :]
                out2[:, 256*j:256*(j+1)] = out
                out = out.unsqueeze(0)
            
            out2 = out2.unsqueeze(0)
            
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(2*self.num_layers2, 1,
                              self.hidden_size2, device=device)
            out2, _ = self.rnn2(out2, h02)
            for j in range(lengths[i] // FRAME_RESOLUTION):
                predictions[p] = self.fc(out2[0][j])
                p += 1
        return predictions
