import torch
import torch.nn as nn

from settings import *


class EncoderDecoderModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.enc1 = nn.RNN(input_size=N_MELS,
                                       hidden_size=64,
                                       num_layers=2, batch_first=True, bidirectional=True)
        self.enc2 = nn.RNN(input_size=self.enc1.hidden_size*2,
                                       hidden_size=64,
                                       num_layers=2, batch_first=True, bidirectional=True)
        self.dec1 = nn.GRU(input_size=self.enc2.hidden_size*2,
                                       hidden_size=256,
                                       num_layers=2, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(self.dec1.hidden_size*2, self.output_size*4)        

    def forward(self, batch, lengths, device):
        predictions = torch.zeros(
            lengths.sum().item() // FRAME_RESOLUTION, self.output_size, device=device)
        p = 0
        for i in range(batch.size(0)):
            fbank = batch[i][:lengths[i]]
            frames = fbank.unfold(0, FRAME_RESOLUTION,
                                  FRAME_RESOLUTION).transpose(1, 2)
            h_enc1 = torch.zeros(self.enc1.num_layers*2, frames.size(0),
                              self.enc1.hidden_size, device=device)
            out, _ = self.enc1(frames, h_enc1)
            out = out[:, -1, :]
            out = out.unfold(0, 4, 4).transpose(1, 2)
            h_enc2 = torch.zeros(self.enc2.num_layers*2, out.size(0),
                              self.enc2.hidden_size, device=device)
            out, _ = self.enc2(out, h_enc2)
            out = out[:, -1, :]
            out = out.unsqueeze(0)
            h_dec1 = torch.zeros(self.dec1.num_layers*2, out.size(0),
                              self.dec1.hidden_size, device=device)
            out, _ = self.dec1(out, h_dec1)
            out = self.fc1(out.squeeze())
            out = out.unfold(1, self.output_size, self.output_size).flatten(0, 1)
            predictions[p:p+out.size(0), :] = out
            p += out.size(0)
        return predictions
