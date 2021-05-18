from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from settings import *


class CNNModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

        self.bn3 = nn.BatchNorm2d(3)
        self.bn8 = nn.BatchNorm2d(8)
        self.bn32 = nn.BatchNorm2d(32)

        self.c1_4 = nn.Conv2d(1, 8, kernel_size=4, stride=1)
        self.c1_8 = nn.Conv2d(1, 8, kernel_size=8, stride=2)
        self.c1_16 = nn.Conv2d(1, 8, kernel_size=16, stride=4)
        self.c1_32 = nn.Conv2d(1, 8, kernel_size=32, stride=8)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.c2_4 = nn.Conv2d(8, 3, kernel_size=5, stride=3)
        self.c2_8 = nn.Conv2d(8, 3, kernel_size=3, stride=2)
        self.c2_16 = nn.Conv2d(8, 3, kernel_size=1, stride=1)
        self.c2_32 = nn.Conv2d(8, 3, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(5040, 512)

        self.num_layers = 2
        self.hidden_size = 256
        self.rnn_input_size = 128 + 512

        self.rnn = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, batch, lengths):
        x = batch[:, None, :, :]
        # feature extraction
        r1_4 = self.dropout(self.maxpool(F.relu(self.bn8(self.c1_4(x)))))
        r1_8 = self.dropout(self.maxpool(F.relu(self.bn8(self.c1_8(x)))))
        r1_16 = self.dropout(self.maxpool(F.relu(self.bn8(self.c1_16(x)))))
        r1_32 = self.dropout(self.maxpool(F.relu(self.bn8(self.c1_32(x)))))

        r2_4 = self.dropout(F.relu(self.bn3(self.c2_4(r1_4))))
        r2_8 = self.dropout(F.relu(self.bn3(self.c2_8(r1_8))))
        r2_16 = self.dropout(F.relu(self.bn3(self.c2_16(r1_16))))
        r2_32 = self.dropout(F.relu(self.bn3(self.c2_32(r1_32))))

        max_len = torch.max(lengths)
        p = 0
        predictions = torch.zeros(
            lengths.sum().item() // SPECTROGRAM_FRAME_LENGTH, self.output_size, device=CUDA0)

        for i, x in enumerate(batch):
            # remove padding
            len_ratio = lengths[i] / max_len
            rx_4 = r2_4[i, :, :int(r2_4.size(2) * len_ratio), :]
            rx_8 = r2_8[i, :, :int(r2_8.size(2) * len_ratio), :]
            rx_16 = r2_16[i, :, :int(r2_16.size(2) * len_ratio), :]
            rx_32 = r2_32[i, :, :int(r2_32.size(2) * len_ratio), :]

            # normalize length
            r_4 = torchvision.transforms.Resize((90, 9))(rx_4)
            r_8 = torchvision.transforms.Resize((67, 6))(rx_8)
            r_16 = torchvision.transforms.Resize((67, 6))(rx_16)
            r_32 = torchvision.transforms.Resize((33, 2))(rx_32)

            r = torch.cat((r_4.flatten(), r_8.flatten(),
                          r_16.flatten(), r_32.flatten()))

            r = self.fc1(r)

            frames = x[:lengths[i]].unfold(
                0, SPECTROGRAM_FRAME_LENGTH, SPECTROGRAM_FRAME_LENGTH).flatten(1)

            inp = torch.cat((frames, r.repeat(frames.size(0), 1)), 1)
            inp = inp.view(inp.size(0), 1, inp.size(1))

            h0 = torch.zeros(self.num_layers*2, 1,
                             self.hidden_size, device=CUDA0)
            out, _ = self.rnn(inp, h0)
            for j in range(frames.size(0)):
                predictions[p] = self.fc(out[j][0])
                p += 1
        return predictions
