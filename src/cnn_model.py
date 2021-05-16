import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from settings import *


class CNNModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

        self.c1 = nn.Conv2d(1, 32, kernel_size=16, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(64, 8, kernel_size=(4, 2), stride=(2, 1))
        self.c4 = nn.Conv2d(8, 1, kernel_size=(4, 2), stride=1)

        self.num_layers = 2
        self.hidden_size = 256
        self.rnn_input_size = 3*64+3*80

        self.rnn2 = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, batch, lengths):
        batch_size = batch.size(0)
        s = batch.view(batch_size, 1, batch.size(1)
                       * batch.size(2), batch.size(3))
        s = F.relu(self.bn1(self.c1(s)))
        s = F.relu(self.bn2(self.c2(s)))
        s = F.max_pool2d(s, 2)
        s = F.relu(self.c3(s))
        s = F.relu(self.c4(s))
        s = torchvision.transforms.Resize((80, 3))(s)
        s = s.view(s.size(0), -1)

        predictions = torch.zeros(
            lengths.sum().item(), self.output_size, device=CUDA0)

        p = 0
        for i, x in enumerate(batch):
            x = x.flatten(1)
            x = torch.cat((x, s[i].repeat(x.size(0), 1)), 1)
            x = x.view(x.size(0), 1, x.size(1))
            h0 = torch.zeros(self.num_layers*2, 1,
                             self.hidden_size, device=CUDA0)
            out, _ = self.rnn2(x, h0)
            for j in range(lengths[i]):
                predictions[p] = self.fc(out[j][0])
                p += 1

        return predictions
