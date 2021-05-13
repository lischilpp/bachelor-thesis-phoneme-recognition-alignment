
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import random_split

from settings import *
from preprocessing.timit_dataset import TimitDataset


def dataset_to_disk(ds, save_path):
    if not save_path.exists():
        save_path.mkdir()
    i = 0
    for entry in ds:
        torch.save(entry, save_path / f'record{i}')
        i += 1


train_val_ds = TimitDataset(test=False)
train_val_count = len(train_val_ds)
val_count = int(train_val_count * VAL_PERCENTAGE)
train_count = train_val_count - val_count

train_ds, val_ds = random_split(train_val_ds, [train_count, val_count])
test_ds = TimitDataset(test=True)

print('processing train dataset...')
TRAIN_RAW_PATH.mkdir(exist_ok=True, parents=True)
dataset_to_disk(train_ds, TRAIN_RAW_PATH)

print('processing val dataset...')
VAL_RAW_PATH.mkdir(exist_ok=True, parents=True)
dataset_to_disk(val_ds, VAL_RAW_PATH)

print('processing test dataset...')
TEST_RAW_PATH.mkdir(exist_ok=True, parents=True)
dataset_to_disk(test_ds, TEST_RAW_PATH)

print('--- DONE ---')