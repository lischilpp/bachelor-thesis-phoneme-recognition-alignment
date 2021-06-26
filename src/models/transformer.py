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
        self.num_classes = num_classes
        self.d_model = N_MELS
        self.pos_dropout = 0.1
        self.max_seq_length = 1000
        self.nhead = 2
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dim_feedforward = 2048
        self.trans_dropout = 0.1

        self.embed_tgt = nn.Embedding(self.num_classes+2, self.d_model)
        self.pos_enc = PositionalEncoding(
            self.d_model,
            self.pos_dropout,
            self.max_seq_length)

        self.transformer = nn.Transformer(
            self.d_model,
            self.nhead,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.dim_feedforward,
            self.trans_dropout)
        self.linear = nn.Linear(self.d_model, self.num_classes+2)

    def get_padding_mask(self, lengths, padded_length, device):
        mask = torch.zeros(lengths.size(0), padded_length,
                           device=device)
        for i in range(lengths.size(0)):
            mask[i, lengths[i]:] = 1
        return mask.bool()

    def get_nopeek_mask(self, size, device):
        return torch.full((size, size), float('-inf'), device=device).triu(1)

    def forward(self, src, tgt, lengths, device):
        self.transformer.train()
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        src_padding_mask = self.get_padding_mask(
            lengths, src_len, device)
        tgt_padding_mask = self.get_padding_mask(
            lengths, tgt_len, device)
        src_nopeek_mask = self.get_nopeek_mask(src_len, device)
        tgt_nopeek_mask = self.get_nopeek_mask(tgt_len, device)
        src = self.pos_enc(src.transpose(0, 1) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt.transpose(0, 1)) * math.sqrt(self.d_model))

        out = self.transformer(src,
                               tgt,
                               src_mask=src_nopeek_mask,
                               tgt_mask=tgt_nopeek_mask,
                               src_key_padding_mask=src_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask,
                               memory_key_padding_mask=src_padding_mask)

        out = self.linear(out.transpose(0, 1))
        return out

    def evaluate_input(self, src, lengths, device):
        self.transformer.eval()
        
        batch_size = src.size(0)
        seq_len = src.size(1)
        src_padding_mask = self.get_padding_mask(
            lengths, seq_len, device)
        tgt_padding_mask = self.get_padding_mask(
            lengths, seq_len, device)
        src_nopeek_mask = self.get_nopeek_mask(seq_len, device)
        tgt_nopeek_mask = self.get_nopeek_mask(seq_len, device)
        src = self.pos_enc(src.transpose(0, 1) * math.sqrt(self.d_model))

        src = self.transformer.encoder(src,
                                       mask=src_nopeek_mask,
                                       src_key_padding_mask=src_padding_mask)

        tgt = torch.zeros(seq_len+1, batch_size, dtype=torch.long, device=device)
        tgt[0, :] = self.num_classes
        for i in range(1, seq_len+1):
            tgt_embed = self.embed_tgt(tgt[:i].transpose(0, 1)).transpose(0, 1)
            tgt_enc = self.pos_enc(tgt_embed * math.sqrt(self.d_model))

            out = self.transformer.decoder(tgt=tgt_enc,
                                           memory=src,
                                           tgt_mask=tgt_nopeek_mask[:i, :i],
                                           memory_mask=src_nopeek_mask[:i, :],
                                           tgt_key_padding_mask=tgt_padding_mask[:, :i],
                                           memory_key_padding_mask=src_padding_mask[:, :seq_len])

            tgt[i] = self.linear(out[-1]).log_softmax(-1).argmax(dim=1)

        return tgt[1:].transpose(0, 1)


# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
