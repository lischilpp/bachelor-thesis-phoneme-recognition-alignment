from pytorch_lightning.tuner.lr_finder import lr_find
from phonemes import Phoneme
from dataset.disk_dataset import DiskDataset
from dataset.frame_dataset import FrameDataset
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from models.rnn_frequency_model import RNNFrequencyModel
from models.rnn_waveform_model import RNNWaveformModel
from models.encoder_decoder import EncoderDecoderModel
from settings import *
from pytorch_lightning.metrics import functional as FM
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torchmetrics import ConfusionMatrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


num_epochs = 100
batch_size = 8
initial_lr = 0.001
lr_patience = 1
lr_reduce_factor = 0.5
auto_lr_find=False


def collate_fn(batch):
    lengths = torch.tensor([item[0].size(0) for item in batch])
    specgrams = [item[0] for item in batch]
    specgrams = pad_sequence(specgrams, batch_first=True)
    labels = torch.cat([item[1] for item in batch])
    frame_data = (specgrams, lengths)
    return [frame_data, labels]


class TimitDataModule(pl.LightningDataModule):

    def setup(self, stage):
        self.train_ds = FrameDataset(
            DiskDataset(TRAIN_PATH), augment=AUGMENT_DATASET)
        self.val_ds = FrameDataset(DiskDataset(VAL_PATH))
        self.test_ds = FrameDataset(DiskDataset(TEST_PATH))

        self.loader_args = {'batch_size': batch_size,
                            'collate_fn': collate_fn,
                            'num_workers': 12,
                            'pin_memory': True}

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


class PhonemeClassifier(pl.LightningModule):

    def __init__(self, batch_size, lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.last_val_loss = float('inf')
        self.loss_too_high_count = 0
        self.model = RNNModel(output_size=Phoneme.folded_phoneme_count())
        self.criterion = nn.CrossEntropyLoss()#weight=Phoneme.folded_phoneme_weights)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr)
        self.confmatMetric = ConfusionMatrix(num_classes=Phoneme.folded_group_phoneme_count())
        self.init_cyclic_scheduler()

    def init_cyclic_scheduler(self):
        self.lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CyclicLR(
                                self.optimizer,
                                base_lr=self.lr/10,
                                max_lr=self.lr,
                                step_size_up=100,
                                cycle_momentum=False),
            'interval': 'step',
            'frequency': 1
        }

    def training_step(self, batch, _):
        (specgrams, lengths), labels = batch
        outputs = self.model(specgrams, lengths, self.device)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        (specgrams, lengths), labels = batch
        specgrams = specgrams
        labels = labels
        outputs = self.model(specgrams, lengths, self.device)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        preds = self.foldGroupIndices(preds)
        labels = self.foldGroupIndices(labels)
        acc = FM.accuracy(preds, labels)
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def validation_epoch_end(self, val_step_outputs):
        val_loss = sum([output['val_loss'] for output in val_step_outputs])/len(val_step_outputs)
        if val_loss > self.last_val_loss * 0.9999:
            self.loss_too_high_count += 1
        if self.loss_too_high_count > lr_patience:
            self.lr *= lr_reduce_factor
            self.init_cyclic_scheduler()
            self.trainer.lr_schedulers = self.trainer.configure_schedulers(
                [self.lr_scheduler],
                monitor='val_loss',
                is_manual_optimization=False)
            self.loss_too_high_count=0

        self.last_val_loss = val_loss

    def test_step(self, batch, _):
        (specgrams, lengths), labels = batch
        specgrams = specgrams
        outputs = self.model(specgrams, lengths, self.device)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        preds = self.foldGroupIndices(preds)
        labels = self.foldGroupIndices(labels)
        acc = FM.accuracy(preds, labels)
        self.confmatMetric(preds, labels)
        metrics = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]

    def foldGroupIndices(self, indizes):
        for i in range(indizes.size(0)):
            symbol = Phoneme.folded_phoneme_list[indizes[i]]
            indizes[i] = Phoneme.folded_group_phoneme_list.index(Phoneme.symbol_to_folded_group.get(symbol, symbol))
        return indizes

    # hide v_num in progres bar
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


if __name__ == '__main__':
    dm = TimitDataModule()

    model = PhonemeClassifier(batch_size, initial_lr)
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, precision=16,
        auto_lr_find=auto_lr_find,
        gradient_clip_val=0.5)
        # resume_from_checkpoint='lightning_logs/version_164/checkpoints/epoch=79-step=8319.ckpt')

    if auto_lr_find:
        trainer.tune(model, dm)
    else:
        trainer.fit(model, dm)
        trainer.test(datamodule=dm)

        confmat = model.confmatMetric.compute()
        plt.figure(figsize=(15,10))
        class_names = Phoneme.folded_group_phoneme_list
        df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()