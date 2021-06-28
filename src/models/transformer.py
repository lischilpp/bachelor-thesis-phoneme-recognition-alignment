import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from settings import *


class TransformerModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = 40
        self.ninp = 256
        self.nhid = 2048
        self.nlayers = 4
        self.nhead = 4
        self.dropout = 0.1
        self.linear1 = nn.Linear(N_MELS, self.ninp)
        self.pos_encoder = PositionalEncoding(self.ninp, self.dropout, self.max_seq_len)
        encoder_layer = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.nlayers)
        self.decoder = nn.Linear(self.ninp, num_classes)
        self.dropout_layer = nn.Dropout(0.5)

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
        src = self.linear1(src).transpose(0, 1)
        splits = src.split(self.max_seq_len, dim=0)
        src_mask = self.get_nopeek_mask(self.max_seq_len, device)
        output = torch.zeros(src.size(0), src.size(1), self.ninp, device=device)
        for i, split in enumerate(splits):
            nopeek_mask = src_mask
            padding_mask = None
            if split.size(0) < self.max_seq_len:
                padding_mask = self.get_padding_mask(lengths, split.size(0), device)
                nopeek_mask = src_mask[:split.size(0), :split.size(0)]
            split = split * math.sqrt(self.ninp)
            split = self.pos_encoder(split)
            out = self.transformer_encoder(split,
                                           nopeek_mask,
                                           src_key_padding_mask=padding_mask)
            output[i*self.max_seq_len:(i+1)*self.max_seq_len] = out

        output = self.decoder(output)
        output = output.transpose(0, 1)
        output = self.dropout_layer(output)
        return output


# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
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
