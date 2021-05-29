import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil

from dataset.disk_dataset import DiskDataset
from dataset.timit_dataset import TimitDataset
from settings import *
from torch.utils.data import random_split



train_val_ds = TimitDataset(test=False)
train_val_count = len(train_val_ds)
val_count = int(train_val_count * VAL_PERCENTAGE)
train_count = train_val_count - val_count

train_ds, val_ds = random_split(train_val_ds, [train_count, val_count])
test_ds = TimitDataset(test=True)


print('processing train dataset...')
shutil.rmtree(TRAIN_PATH)
TRAIN_PATH.mkdir(parents=True)
DiskDataset.write(train_ds, TRAIN_PATH, exclude_sa_files=False)

print('processing val dataset...')
shutil.rmtree(VAL_PATH)
VAL_PATH.mkdir()
DiskDataset.write(val_ds, VAL_PATH)

print('processing test dataset...')
shutil.rmtree(TEST_PATH)
TEST_PATH.mkdir()
DiskDataset.write(test_ds, TEST_PATH)

print('--- DONE ---')
