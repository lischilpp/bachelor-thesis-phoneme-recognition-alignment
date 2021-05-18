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

        self.bn4 = nn.BatchNorm2d(4)
        self.bn8 = nn.BatchNorm2d(8)
        self.bn32 = nn.BatchNorm2d(32)

        self.c1_4 = nn.Conv2d(1, 8, kernel_size=4, stride=1)
        self.c1_8 = nn.Conv2d(1, 8, kernel_size=8, stride=2)
        self.c1_16 = nn.Conv2d(1, 8, kernel_size=16, stride=4)
        self.c1_32 = nn.Conv2d(1, 8, kernel_size=32, stride=8)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.c2_4 = nn.Conv2d(8, 32, kernel_size=4, stride=2)
        self.c2_8 = nn.Conv2d(8, 32, kernel_size=3, stride=2)
        self.c2_16 = nn.Conv2d(8, 32, kernel_size=3, stride=1)
        self.c2_32 = nn.Conv2d(8, 32, kernel_size=2, stride=1)

        self.c3_4 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.c3_8 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.c3_16 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.c3_32 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

        self.fc1 = nn.Linear(10740, 1024)

        self.num_layers = 2
        self.hidden_size = 256
        self.rnn_input_size = 128 + 1024

        self.rnn = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, batch, lengths):
        x = batch[:, None, :, :]
        # feature extraction
        r1_4 = self.maxpool(F.relu(self.bn8(self.c1_4(x))))
        r1_8 = self.maxpool(F.relu(self.bn8(self.c1_8(x))))
        r1_16 = self.maxpool(F.relu(self.bn8(self.c1_16(x))))
        r1_32 = self.maxpool(F.relu(self.bn8(self.c1_32(x))))
        r2_4 = F.relu(self.bn32(self.c2_4(r1_4)))
        r2_8 = F.relu(self.bn32(self.c2_8(r1_8)))
        r2_16 = F.relu(self.bn32(self.c2_16(r1_16)))
        r2_32 = F.relu(self.bn32(self.c2_32(r1_32)))
        r3_4 = F.relu(self.bn4(self.c3_4(r2_4)))
        r3_8 = F.relu(self.bn4(self.c3_8(r2_8)))
        r3_16 = F.relu(self.bn4(self.c3_16(r2_16)))
        r3_32 = F.relu(self.bn4(self.c3_32(r2_32)))

        max_len = torch.max(lengths)
        p = 0
        predictions = torch.zeros(
            lengths.sum().item() // SPECTROGRAM_FRAME_LENGTH, self.output_size, device=CUDA0)

        for i, x in enumerate(batch):
            # remove padding
            len_ratio = lengths[i] / max_len
            rx_4 = r3_4[i, :, :int(r3_4.size(2) * len_ratio), :]
            rx_8 = r3_8[i, :, :int(r3_8.size(2) * len_ratio), :]
            rx_16 = r3_16[i, :, :int(r3_16.size(2) * len_ratio), :]
            rx_32 = r3_32[i, :, :int(r3_32.size(2) * len_ratio), :]

            # normalize length
            r_4 = torchvision.transforms.Resize((140, 14))(rx_4)
            r_8 = torchvision.transforms.Resize((70, 6))(rx_8)
            r_16 = torchvision.transforms.Resize((68, 4))(rx_16)
            r_32 = torchvision.transforms.Resize((33, 1))(rx_32)

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
