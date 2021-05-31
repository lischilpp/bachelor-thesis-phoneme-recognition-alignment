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
shutil.rmtree(TRAIN_PATH, ignore_errors=True)
TRAIN_PATH.mkdir(parents=True)
DiskDataset.write(train_ds, TRAIN_PATH, exclude_sa_files=False)

print('processing val dataset...')
shutil.rmtree(VAL_PATH, ignore_errors=True)
VAL_PATH.mkdir()
DiskDataset.write(val_ds, VAL_PATH)

print('processing test dataset...')
core_test_ds = []
for entry in test_ds:
    _, _, _, is_core_test = entry
    if is_core_test:
        core_test_ds.append(entry)

shutil.rmtree(CORE_TEST_PATH, ignore_errors=True)
CORE_TEST_PATH.mkdir()
DiskDataset.write(core_test_ds, CORE_TEST_PATH)

shutil.rmtree(FULL_TEST_PATH, ignore_errors=True)
FULL_TEST_PATH.mkdir()
DiskDataset.write(test_ds, FULL_TEST_PATH)

print('--- DONE ---')
