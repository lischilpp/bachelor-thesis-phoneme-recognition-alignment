import torch

from settings import *
from dataset.timit_dataset import TimitDataset
from dataset.disk_dataset import DiskDataset
from dataset.frame_dataset import FrameDataset
from phonemes import Phoneme
import matplotlib.pyplot as plt

    
ds = FrameDataset(DiskDataset(TRAIN_PATH), augment=True)
fbank, labels = ds[0]

plt.imshow(fbank.transpose(0, 1))
plt.show()