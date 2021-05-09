from pathlib import Path
import numpy as np
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import MultiplicativeLR
import pytorch_lightning as pl

from settings import *
from dataset import TimitDataset
from phonemes import Phoneme
from utils import sentence_characters


num_classes = Phoneme.phoneme_count()
num_epochs = 100
batch_size = 32
learning_rate = 0.0001
input_size = SPECGRAM_N_MELS


def collate_fn(batch):
    # sentences = torch.cat([item[0] for item in batch])
    lengths = torch.tensor([item[0].size(0) for item in batch])
    frames = [item[0] for item in batch]
    frames = pad_sequence(frames, batch_first=True)
    labels = torch.cat([item[1] for item in batch])
    frame_data = (frames, lengths)
    return [frame_data, labels]


class TimitDataModule(pl.LightningDataModule):

    def setup(self, stage):
        self.train_ds = TimitDataset(train=True)
        self.test_ds  = TimitDataset(train=False)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=4)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_ds,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      num_workers=4)
    


class PhonemeClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.num_layers1 = 2
        self.num_layers2 = 2
        self.hidden_size1 = 128
        self.hidden_size2 = 128
        self.rnn1 = nn.RNN(SPECGRAM_N_MELS, self.hidden_size1, self.num_layers1, batch_first=True, bidirectional=True, dropout=0.5)
        self.rnn2 = nn.GRU(self.hidden_size1*2, self.hidden_size2, self.num_layers1, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size2*2, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, lengths):
        predictions = torch.zeros(lengths.sum().item(), num_classes).cuda()
        p = 0
        for i in range(x.size(0)):
            # feature extraction
            # single frame passed as sequence into BiRNN (many-to-one)
            h01 = torch.zeros(self.num_layers1*2, x.size(1), self.hidden_size1).cuda()
            out, _ = self.rnn1(x[i], h01)
            out = out[:, -1, :]
            out2 = out.unsqueeze(0)
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(self.num_layers2*2, 1, self.hidden_size2).cuda()
            out2, _ = self.rnn2(out2, h02)
            for j in range(lengths[i]):
                predictions[p] = self.fc(out2[0][j])
                p += 1

        return predictions

    def training_step(self, train_batch, batch_idx):
        (specgrams, lengths), labels = train_batch

        outputs = self.forward(specgrams, lengths)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        (specgrams, lengths), labels = val_batch
        specgrams = specgrams
        labels = labels

        outputs = self.forward(specgrams, lengths)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


if __name__ == '__main__':
    data_module = TimitDataModule()

    model = PhonemeClassifier()
    trainer = pl.Trainer(gpus=1, max_epochs=100, profiler="simple")

    trainer.fit(model, data_module)