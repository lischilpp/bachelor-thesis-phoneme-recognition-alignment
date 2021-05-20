import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from phonemes import Phoneme


from settings import *
from dataset.disk_dataset import DiskDataset
from dataset.frame_dataset import FrameDataset


symbols = set()
ds = DiskDataset(TRAIN_PATH)
for _, phonemes in ds:
    for pn in phonemes:
        symbols.add(pn.symbol)

symbols = sorted(symbols)

print(symbols)
print(len(symbols))

folded_symbols = sorted(
    set([Phoneme.symbol_to_folded.get(s, s) for s in symbols]))
print(folded_symbols)
print(len(folded_symbols))
