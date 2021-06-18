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
        self.dim_feedforward = 1024
        self.trans_dropout = 0.1

        self.embed_tgt = nn.Embedding(self.num_classes+1, self.d_model)
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
        self.fc = nn.Linear(self.d_model, self.num_classes+1)

    def get_padding_mask(self, lengths, padded_length, device):
        mask = torch.zeros(lengths.size(0), padded_length,
                           device=device).bool()
        for i in range(lengths.size(0)):
            mask[i, lengths[i]:] = True
        return mask

    def get_nopeek_mask(self, size, device):
        return torch.full((size, size), float('-inf'), device=device).triu(1)

    def forward(self, src, tgt, lengths, device):
        self.transformer.train()
        src = src.transpose(0, 1)
        # start_token_feature_tensor = torch.full((src.size(0), src.size(1), 1), N_MELS, device=device)
        # src = torch.cat((start_token_feature_tensor, src), dim=2)
        key_padding_mask = self.get_padding_mask(
            lengths, src.size(0), device)
        src = self.pos_enc(src * math.sqrt(self.d_model))
        tgt_mask = self.get_nopeek_mask(tgt.size(1), device)
        tgt = self.pos_enc(self.embed_tgt(tgt).transpose(
            0, 1) * math.sqrt(self.d_model))

        out = self.transformer(src,
                               tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=key_padding_mask,
                               tgt_key_padding_mask=key_padding_mask,
                               memory_key_padding_mask=key_padding_mask)

        out = self.fc(out).transpose(0, 1)
        return out

    def evaluate_input(self, src, lengths, device):
        self.transformer.eval()
        src = src.transpose(0, 1)
        # start_token_tensor = torch.full((src.size(0), src.size(1), 1), N_MELS, device=device)
        # src = torch.cat((start_token_tensor, src), dim=2)
        batch_size = src.size(1)
        key_padding_mask = self.get_padding_mask(
            lengths, src.size(0), device)
        memory_mask = self.get_nopeek_mask(src.size(0), device)
        tgt_mask = self.get_nopeek_mask(src.size(0), device)

        src = self.pos_enc(src * math.sqrt(self.d_model))
        src = self.transformer.encoder(src, 
                                       src_key_padding_mask=key_padding_mask)

        inputs = torch.zeros(src.size(0), batch_size, dtype=torch.long, device=device)
        inputs[0, :] = self.num_classes
        for i in range(1, src.size(0)):
            tgt = self.pos_enc(self.embed_tgt(inputs[:i].transpose(0, 1)).transpose(0, 1) * math.sqrt(self.d_model))
            # print(tgt.shape)
            # exit()
            # print('--')
            # print(tgt.shape)
            # print(tgt_mask[:i, :i])
            # print(key_padding_mask[:, :i])
            # print(key_padding_mask.shape)
            # if i > 10:
            #     exit()

            out = self.transformer.decoder(tgt=tgt,
                                           memory=src,
                                           tgt_mask=tgt_mask[:i, :i],
                                        #    memory_mask=memory_mask[:i, :],
                                           tgt_key_padding_mask=key_padding_mask[:, :i],
                                           memory_key_padding_mask=key_padding_mask)

            inputs[i] = self.fc(out).softmax(0)[-1].argmax(dim=1)

        return inputs.transpose(0, 1)


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
