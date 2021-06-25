
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


import torchaudio

wav_path = 'test.wav'
ds = DiskDataset(TRAIN_PATH)

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
waveform, _ = ds[400]
waveform = (waveform + 1) * 100
print(f" - Max:     {waveform.max().item():6.3f}")
print(f" - Min:     {waveform.min().item():6.3f}")


# import torch
# import time

# device = 'cuda:0'
# size = 1000
# n_repeats = 100

# torch.tensor(float('-inf'), device=device).expand(size, size).triu(1)
# torch.triu(torch.full((size, size), float('-inf')), 1)

# time_sum1 = 0
# time_sum2 = 0
# for i in range(n_repeats):
#     # measure time for mask 1
#     start = time.time()
#     torch.tensor(float('-inf'), device=device).expand(size, size).triu(1)
#     time_sum1 += time.time() - start
#     # measure time for mask 2
#     start = time.time()
#     torch.full((size, size), float('-inf'), device=device).triu(1)
#     time_sum2 += time.time() - start

# mask1_time = time_sum1 / n_repeats
# mask2_time = time_sum2 / n_repeats
# speedup = mask1_time / mask2_time
# print(f'mask1 took {mask1_time:f}')
# print(f'mask2 took {mask2_time:f}')
# print(f'speedup: {speedup :f}')
