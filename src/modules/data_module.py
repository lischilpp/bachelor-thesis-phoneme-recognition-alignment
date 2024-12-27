import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from dataset.disk_dataset import DiskDataset
from dataset.frame_dataset import FrameDataset
from settings import *


def collate_fn(batch):
    lengths = torch.tensor([item[0].size(0) for item in batch])
    fbanks = pad_sequence([item[0] for item in batch], batch_first=True)
    labels = pad_sequence([item[1] for item in batch], batch_first=True)
    sentences = [item[2] for item in batch]
    frame_data = (fbanks, lengths)
    return [frame_data, labels, sentences]

class DataModule(pl.LightningDataModule):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):
        self.train_ds = FrameDataset(
            DiskDataset(TRAIN_PATH), augment=AUGMENT_DATASET)
        self.val_ds = FrameDataset(DiskDataset(VAL_PATH))
        
        if USE_FULL_TEST_SET:
            self.test_ds = FrameDataset(DiskDataset(FULL_TEST_PATH))
        else:
            self.test_ds = FrameDataset(DiskDataset(CORE_TEST_PATH))

        self.loader_args = {'batch_size': self.batch_size,
                            'collate_fn': collate_fn,
                            'num_workers': 4,
                            'pin_memory': True,
                            'persistent_workers': True}

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          shuffle=True,
                          **self.loader_args)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          **self.loader_args)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds,
                          **self.loader_args)