from pathlib import Path
from math import floor
import numpy as np
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from settings import *
from dataset import TimitDataset
from phonemes import Phoneme
from utils import sentence_characters


num_classes = Phoneme.phoneme_count()
num_epochs = 100
batch_size = 64
learning_rate = 0.01
input_size = SPECGRAM_N_MELS


def collate_fn(batch):
    lengths = torch.tensor([item[0].size(0) for item in batch])
    frames = [item[0] for item in batch]
    frames = pad_sequence(frames, batch_first=True)
    labels = torch.cat([item[1] for item in batch])
    frame_data = (frames, lengths)
    return [frame_data, labels]


class TimitDataModule(pl.LightningDataModule):

    def setup(self, stage):
        train_val_data = TimitDataset(train=True)
        train_val_count = len(train_val_data)
        val_percentage = 0.2
        val_count = floor(train_val_count * val_percentage)
        train_count = train_val_count - val_count

        self.train_ds, self.val_ds = random_split(train_val_data,
                                                 [train_count, val_count])
        self.test_ds = TimitDataset(train=False)

        self.ds_args = {'batch_size': batch_size,
                        'collate_fn': collate_fn,
                        'num_workers': 6,
                        'pin_memory': True}

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          shuffle=True,
                          **self.ds_args)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          **self.ds_args)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds,
                          **self.ds_args)
    

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers1 = 2
        self.num_layers2 = 2
        self.hidden_size1 = 128
        self.hidden_size2 = 128
        self.rnn1 = nn.RNN(SPECGRAM_N_MELS, self.hidden_size1, self.num_layers1, batch_first=True, bidirectional=True, dropout=0.5)
        self.rnn2 = nn.GRU(self.hidden_size1*2, self.hidden_size2, self.num_layers1, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size2*2, num_classes)

    def forward(self, x, lengths):
        predictions = torch.zeros(lengths.sum().item(), num_classes).cuda(non_blocking=True)
        p = 0
        for i in range(x.size(0)):
            # feature extraction
            # single frame passed as sequence into BiRNN (many-to-one)
            h01 = torch.zeros(self.num_layers1*2, x.size(1), self.hidden_size1).cuda(non_blocking=True)
            out, _ = self.rnn1(x[i], h01)
            out = out[:, -1, :]
            out2 = out.unsqueeze(0)
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(self.num_layers2*2, 1, self.hidden_size2).cuda(non_blocking=True)
            out2, _ = self.rnn2(out2, h02)
            for j in range(lengths[i]):
                predictions[p] = self.fc(out2[0][j])
                p += 1
        return predictions


class PhonemeClassifier(pl.LightningModule):

    def __init__(self, batch_size, lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.model = ClassificationModel()
        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        (specgrams, lengths), labels = batch

        outputs = self.model(specgrams, lengths)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (specgrams, lengths), labels = batch
        specgrams = specgrams
        labels = labels

        outputs = self.model(specgrams, lengths)
        loss = self.criterion(outputs, labels)
        acc = FM.accuracy(torch.argmax(outputs, dim=1), labels)

        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_loss': metrics['val_loss'], 'test_acc': metrics['val_acc']}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_schedulers = {'scheduler': ReduceLROnPlateau(optimizer, patience=0),
                         'monitor': 'val_loss'}
        return [optimizer], [lr_schedulers]

    # hide v_num in progres bar
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


if __name__ == '__main__':
    dm = TimitDataModule()
    
    model = PhonemeClassifier(batch_size, learning_rate)
    trainer = pl.Trainer(gpus=1, max_epochs=15)#, resume_from_checkpoint='lightning_logs/version_42/checkpoints/epoch=14-step=314.ckpt')

    trainer.fit(model, dm)
    trainer.test(datamodule=dm)