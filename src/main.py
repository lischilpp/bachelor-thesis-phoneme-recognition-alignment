from cnn_model import CNNModel
from dataset.frame_dataset import FrameDataset
from dataset.disk_dataset import DiskDataset
from phonemes import Phoneme
from settings import *
from pytorch_lightning.metrics import functional as FM
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import warnings

# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )


num_epochs = 30
batch_size = 8
initial_lr = 0.001
lr_patience = 0
lr_reduce_factor = 0.1


def collate_fn(batch):
    lengths = torch.tensor([item[0].size(0) for item in batch])
    frames = [item[0] for item in batch]
    frames = pad_sequence(frames, batch_first=True)
    labels = torch.cat([item[1] for item in batch])
    frame_data = (frames, lengths)
    return [frame_data, labels]


class TimitDataModule(pl.LightningDataModule):

    def setup(self, stage):
        self.train_ds = FrameDataset(DiskDataset(TRAIN_PATH), augment=True)
        self.val_ds = FrameDataset(DiskDataset(VAL_PATH))
        self.test_ds = FrameDataset(DiskDataset(TEST_PATH))

        self.ds_args = {'batch_size': batch_size,
                        'collate_fn': collate_fn,
                        'num_workers': 0,
                        'pin_memory': False}

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


class PhonemeClassifier(pl.LightningModule):

    def __init__(self, batch_size, initial_lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = initial_lr
        self.model = CNNModel(output_size=Phoneme.phoneme_count())
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, factor=lr_reduce_factor, patience=lr_patience)

    def on_epoch_end(self):
        self.log('lr', self.optimizer.param_groups[0]['lr'], prog_bar=True)

    def training_step(self, batch, _):
        (specgrams, lengths), labels = batch

        outputs = self.model(specgrams, lengths)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, _):
        (specgrams, lengths), labels = batch
        specgrams = specgrams
        labels = labels
        outputs = self.model(specgrams, lengths)
        # print(outputs.shape)
        # print(labels.shape)
        loss = self.criterion(outputs, labels)
        acc = FM.accuracy(torch.argmax(outputs, dim=1), labels)
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, _):
        (specgrams, lengths), labels = batch
        specgrams = specgrams
        labels = labels
        outputs = self.model(specgrams, lengths)
        loss = self.criterion(outputs, labels)
        acc = FM.accuracy(torch.argmax(outputs, dim=1), labels)
        metrics = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        lr_scheduler = {'scheduler': self.lr_scheduler,
                        'monitor': 'val_loss'}
        return [self.optimizer], [lr_scheduler]

    # hide v_num in progres bar
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


if __name__ == '__main__':
    dm = TimitDataModule()

    model = PhonemeClassifier(batch_size, initial_lr)
    # resume_from_checkpoint='lightning_logs/version_42/checkpoints/epoch=14-step=314.ckpt')
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs,
                         stochastic_weight_avg=True)  # precision=16,

    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
