
from settings import *
from dataset.timit_dataset import TimitDataset
from dataset.disk_dataset import DiskDataset
from dataset.frame_dataset import FrameDataset
from phonemes import Phoneme
import matplotlib.pyplot as plt
import torch.nn.functional as F
    
# ds = FrameDataset(DiskDataset(TRAIN_PATH), augment=True)
# fbank, labels = ds[0]

# plt.imshow(fbank.transpose(0, 1))
# plt.show()
# exit()

# import torchaudio

# wav_path = 'test.wav'
# ds = DiskDataset(TRAIN_PATH)

# waveform = (waveform*1e4+336).int()
# print(waveform[20000:20020])
# min_val = float('inf')
# max_val = float('-inf')
# for waveform, _ in ds:
#     sample_min = waveform.min().item()
#     if sample_min < min_val:
#         min_val = sample_min
#     sample_max = waveform.max().item()
#     if sample_max > max_val:
#         max_val = sample_max
# print(f" - Max:     {min_val:6.3f}")
# print(f" - Min:     {max_val:6.3f}")
# waveform, _ = ds[400]
# waveform = (waveform + 1) * 100
# print(f" - Max:     {waveform.max().item():6.3f}")
# print(f" - Min:     {waveform.min().item():6.3f}")

import numpy as np
import torch

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
x = torch.tensor([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = torch.tensor([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)

print(x.shape)

from dtw import dtw

manhattan_distance = lambda x, y: np.abs(x - y)

d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

print(path)

# You can also visualise the accumulated cost and the shortest path
import matplotlib.pyplot as plt

plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()