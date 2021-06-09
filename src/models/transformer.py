import math
import torch
import torch.nn as nn
from torch.nn import (
    TransformerEncoder, 
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer)

from settings import *


class TransformerModel(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.ninp = N_MELS
        self.nhid = 200
        self.nlayers = 2
        self.nhead = 2
        self.dropout = 0.2
        self.pos_encoder = PositionalEncoding(self.ninp, self.dropout)
        encoder_layer = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.nlayers)
        self.decoder = nn.Linear(self.ninp, output_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src_mask = self.generate_square_subsequent_mask(src.size(0)).cuda()
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)