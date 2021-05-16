import torch
import torch.nn as nn

from settings import *


class DynamicKernel():
    def __init__(self, func, width, height, padding=0):
        self.func = func
        self.width = width
        self.height = height
        self.padding = nn.ZeroPad2d(padding)

    def apply(self, input):
        print(input.shape)
        print(self.padding(input).unfold(0, self.height, self.height).shape)
        rows = self.padding(input).unfold(0, self.height, self.height).unfold(
            1, self.width, self.width)
        print(rows.shape)
        out = torch.tensor([[self.func(e) for e in row]
                           for row in rows], device=CUDA0)
        return out


class CNNModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.nns8 = [nn.Linear(8*8, 2*2) for _ in range(8)]
        self.kernels8 = [DynamicKernel(
            self.nns8[i], 8, 8) for i in range(8)]
        self.nns16 = [nn.Linear(16*16, 4*4) for _ in range(4)]
        self.kernels16 = [DynamicKernel(
            self.nns16[i], 16, 16) for i in range(4)]
        self.nns32 = [nn.Linear(32*32, 8*8) for _ in range(2)]
        self.kernels32 = [DynamicKernel(
            self.nns32[i], 32, 32) for i in range(2)]
        self.nn64 = nn.Linear(64*64, 16*16)
        self.kernel64 = DynamicKernel(self.nn64, 64, 64)
        self.output_size = output_size
        self.fc = nn.Linear(64 + 480, output_size)

    def forward(self, batch, lengths):
        predictions = torch.zeros(
            lengths.sum().item(), self.output_size, device=CUDA0)
        p = 0
        for i in range(batch.size(0)):
            x = batch[i]

            self.fms8 = [kernel.apply(x[i:i+8])
                         for i, kernel in enumerate(self.kernels8)]
            self.fms16 = [kernel.apply(x[i:i+16])
                          for i, kernel in enumerate(self.kernels16)]
            self.fms32 = [kernel.apply(x[i:i+32])
                          for i, kernel in enumerate(self.kernels32)]
            self.fm64 = self.kernel64.apply(x)

            self.features = torch.cat(self.fms8.flatten(),
                                      self.fms16.flatten(),
                                      self.fms32.flatten(),
                                      self.fms64.flatten())

            x = x.transpose(0, 1)
            for j in range(lengths[i]):
                predictions[p] = self.fc(
                    torch.cat(x[i].flatten(), self.features))
                p += 1
        return predictions
