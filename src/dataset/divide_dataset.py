import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import random_split

from settings import *
from dataset.timit_dataset import TimitDataset
from dataset.disk_dataset import DiskDataset


train_val_ds = TimitDataset(test=False)
train_val_count = len(train_val_ds)
val_count = int(train_val_count * VAL_PERCENTAGE)
train_count = train_val_count - val_count

train_ds, val_ds = random_split(train_val_ds, [train_count, val_count])
test_ds = TimitDataset(test=True)

print('processing train dataset...')
TRAIN_RAW_PATH.mkdir(exist_ok=True, parents=True)
DiskDataset.write(train_ds, TRAIN_RAW_PATH)

print('processing val dataset...')
VAL_RAW_PATH.mkdir(exist_ok=True, parents=True)
DiskDataset.write(val_ds, VAL_RAW_PATH)

print('processing test dataset...')
TEST_RAW_PATH.mkdir(exist_ok=True, parents=True)
DiskDataset.write(test_ds, TEST_RAW_PATH)

print('--- DONE ---')