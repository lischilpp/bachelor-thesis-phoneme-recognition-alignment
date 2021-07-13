import torch
import torch.nn as nn

from settings import *


class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.encoder = nn.GRU(input_size=N_MELS,
                           hidden_size=256,
                           num_layers=3,
                           batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(input_size=N_MELS,
                           hidden_size=256,
                           num_layers=3,
                           batch_first=True, bidirectional=True)

    def forward(self, batch):
        out, _ = self.encoder(batch)
        return out