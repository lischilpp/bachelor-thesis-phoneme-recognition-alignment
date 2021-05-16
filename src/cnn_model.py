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
        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)

        self.cp1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.cp2 = nn.Conv2d(16, 32, kernel_size=(16, 2), stride=1)
        self.cp3 = nn.Conv2d(32, 8, kernel_size=(16, 2), stride=2)

        self.c1 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.c2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(32, 8, kernel_size=2, stride=1)
        self.c4 = nn.Conv2d(8, 1, kernel_size=5, stride=1)

        self.num_layers = 2
        self.hidden_size = 256
        self.rnn_input_size = 256+266

        self.rnn2 = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, batch, lengths):
        p = 0
        predictions = torch.zeros(
            lengths.sum().item(), self.output_size, device=CUDA0)
        for i, x in enumerate(batch):
            x = x[:lengths[i]]
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
            x = F.relu(self.bn16(self.cp1(x)))
            x = F.relu(self.bn32(self.cp2(x)))
            x = F.relu(self.bn8(self.cp3(x)))
            s = x.view(1, 8, 16, x.size(0) * 2)
            s = F.relu(self.bn16(self.c1(s)))
            s = F.relu(self.bn32(self.c2(s)))
            s = F.max_pool2d(s, 2)
            s = F.relu(self.bn8(self.c3(s)))
            s = F.relu(self.c4(s))
            s = torchvision.transforms.Resize((1, 266))(s)
            s = s.flatten()

            x = x.flatten(1)
            x = torch.cat((x, s.repeat(x.size(0), 1)), 1)
            x = x.view(x.size(0), 1, x.size(1))
            h0 = torch.zeros(self.num_layers*2, 1,
                             self.hidden_size, device=CUDA0)
            out, _ = self.rnn2(x, h0)
            for j in range(x.size(0)):
                predictions[p] = self.fc(out[j][0])
                p += 1

        return predictions
