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

    def __init__(self, num_classes):
        super().__init__()
        d_model = N_MELS
        self.d_model = d_model
        pos_dropout = 0.1
        max_seq_length = 1000
        nhead = 2
        num_encoder_layers = 2
        num_decoder_layers = 2
        dim_feedforward=1024
        trans_dropout = 0.1

        self.embed_tgt = nn.Embedding(num_classes, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def get_padding_mask(self, lengths, padded_length, device):
        mask = torch.zeros(lengths.size(0), padded_length, device=device).bool()
        for i in range(lengths.size(0)):
            mask[i, lengths[i]:] = True
        return mask

    def get_nopeek_mask(self, size, device):
        return torch.full((size, size), float('-inf'), device=device).triu(1)

    def forward(self, src, tgt, lengths, device):
        src = src.transpose(0, 1)
        src_key_padding_mask = self.get_padding_mask(lengths, src.size(0), device)
        tgt_key_padding_mask = src_key_padding_mask.clone()
        memory_key_padding_mask = src_key_padding_mask.clone()
        tgt_mask = self.get_nopeek_mask(tgt.size(1), device)
        src = self.pos_enc(src * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt).transpose(0, 1) * math.sqrt(self.d_model))

        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        out = self.fc(out).transpose(0, 1)
        return out


# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
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