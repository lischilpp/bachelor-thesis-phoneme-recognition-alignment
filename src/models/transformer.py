import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from settings import *


class TransformerModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = 1000
        self.ninp = N_MELS
        self.nhid = 2048
        self.nlayers = 4
        self.nhead = 4
        self.dropout = 0.1
        self.linear1 = nn.Linear(N_MELS, self.ninp)
        self.pos_encoder = PositionalEncoding(self.ninp, self.dropout, self.max_seq_len)
        encoder_layer = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.nlayers)
        self.decoder = nn.Linear(self.ninp, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_padding_mask(self, lengths, padded_length, device):
        mask = torch.zeros(lengths.size(0), padded_length,
                           device=device)
        for i in range(lengths.size(0)):
            mask[i, lengths[i]:] = 1
        return mask.bool()

    def get_nopeek_mask(self, size, device):
        return torch.full((size, size), float('-inf'), device=device).triu(1)

    def forward(self, src, lengths, device):
        src = self.linear1(src)
        src_mask = self.get_nopeek_mask(src.size(1), device)
        padding_mask = self.get_padding_mask(lengths, src.size(1), device)
        src = src.transpose(0, 1) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, src_mask, padding_mask)
        out = self.decoder(out)
        return out.transpose(0, 1)

    def set_train(self):
        self.transformer_encoder.train()
    
    def set_eval(self):
        self.transformer_encoder.eval()


# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1000):
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
