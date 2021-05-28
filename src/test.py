import torch

from settings import *
from dataset.disk_dataset import DiskDataset
from dataset.frame_dataset import FrameDataset
from phonemes import Phoneme


def floor2(x):  # floor to 2 decimals
    return x // 0.01 / 100

    
ds = DiskDataset(TRAIN_PATH)
counts = {s:0 for s in Phoneme.folded_phoneme_list}

for waveform, labels in ds:
    for label in labels:
        counts[label.symbol] += 1

total_count = sum([c for c in counts.values()])
avg_count = total_count / len(counts)
weights = []
for symbol, count in counts.items():
    weights.append(floor2(avg_count / count))

print(total_count)
print(counts)
print(weights)
