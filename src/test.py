import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from dataset.data_augmentation import augment_record
from phonemes import Phoneme
import torchaudio


from settings import *
from dataset.disk_dataset import DiskDataset
from dataset.frame_dataset import FrameDataset


# i = 8
# train_ds = DiskDataset(TRAIN_PATH)
# ds0 = FrameDataset(train_ds)
# specgrams0, labels0 = ds0[i]

# ds1 = FrameDataset(train_ds, augment=True)
# specgrams1, labels1 = ds1[i]

# f = plt.figure()
# f.add_subplot(2, 1, 1)
# plt.imshow(specgrams0.transpose(0, 1))
# f.add_subplot(2, 1, 2)
# plt.imshow(specgrams1.transpose(0, 1))

# ds = DiskDataset(TRAIN_PATH)
# record = ds[3]
# waveform0, phonemes0 = record
# waveform1, phonemes1 = augment_record(record)
# torchaudio.save('test.wav', waveform0.view(1, -1), SAMPLE_RATE)
# torchaudio.save('test_aug.wav', waveform1.view(1, -1), SAMPLE_RATE)

symbols = set()
ds = DiskDataset(TRAIN_PATH)
for waveform, labels in ds:
    for label in labels:
        symbols.add(label.symbol)

symbols = sorted(symbols)
print(symbols)