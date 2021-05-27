import torch
import torch.nn as nn

from settings import *


class EncoderDecoderModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        

    def forward(self, batch, lengths, device):
        predictions = torch.zeros(
            lengths.sum().item() // FRAME_RESOLUTION, self.output_size, device=device)
        p = 0
        for i in range(batch.size(0)):
            fbank = batch[i][:lengths[i]]
            frames = fbank.unfold(0, FRAME_RESOLUTION,
                                  FRAME_RESOLUTION).transpose(1, 2)
            
            for j in range(lengths[i] // FRAME_RESOLUTION):
                predictions[p] = self.fc(out2[0][j])
                p += 1
        return predictions
