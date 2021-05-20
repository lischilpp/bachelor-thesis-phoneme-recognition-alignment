# import torch
# import torch.nn as nn
# import torchaudio.transforms as T
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from phonemes import Phoneme


# from settings import *
# from dataset.disk_dataset import DiskDataset
# from dataset.frame_dataset import FrameDataset


# train_ds = DiskDataset(TRAIN_PATH)
# ds0 = FrameDataset(train_ds)
# specgrams0, labels0 = ds0[0]

# ds1 = FrameDataset(train_ds, augment=True)
# specgrams1, labels1 = ds1[0]

# f = plt.figure()
# f.add_subplot(2, 1, 1)
# plt.imshow(specgrams0.transpose(0, 1))
# f.add_subplot(2, 1, 2)
# plt.imshow(specgrams1.transpose(0, 1))
# plt.show()

import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from phonemes import Phoneme


from settings import *
from dataset.timit_dataset import TimitDataset


symbols = set()
train_ds = TimitDataset(TRAIN_PATH)
val_ds = TimitDataset(VAL_PATH)
test_ds = TimitDataset(TEST_PATH)

print(len(train_ds)+len(val_ds))
print(len(test_ds))
# for _, phonemes in ds:
#     for pn in phonemes:
#         symbols.add(pn.symbol)

# symbols = sorted(symbols)

# print(symbols)
# print(len(symbols))

# folded_symbols = sorted(
#     set([Phoneme.symbol_to_folded.get(s, s) for s in symbols]))
# print(folded_symbols)
# print(len(folded_symbols))
