
import torch

from settings import *
from dataset_generator import TimitDatasetGenerator


def dataset_to_disk(ds, save_path):
    if not save_path.exists():
        save_path.mkdir()
    i = 1
    for entry in ds:
        torch.save(entry, save_path / f'record{i}')
        i += 1

if not SAVED_DATASET_PATH.exists():
    SAVED_DATASET_PATH.mkdir()

print('processing train dataset...')
train_ds = TimitDatasetGenerator(root=TIMIT_PATH, train=True, frame_length=FRAME_LENGTH, stride=STRIDE)
dataset_to_disk(train_ds, SAVED_DATASET_PATH / 'train')

print('processing test dataset...')
test_ds = TimitDatasetGenerator(root=TIMIT_PATH, train=False, frame_length=FRAME_LENGTH, stride=STRIDE)
dataset_to_disk(train_ds, SAVED_DATASET_PATH / 'test')

print('--- DONE ---')