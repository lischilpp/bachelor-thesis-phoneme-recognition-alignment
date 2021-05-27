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
batch_size = 32
initial_lr = 0.0001
swa = True
lr_patience = 0
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
                            'num_workers': 10,
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

    def __init__(self, batch_size, initial_lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = initial_lr
        self.model = RNNModel(output_size=61)#Phoneme.phoneme_count())
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        self.confmatMetric = ConfusionMatrix(num_classes=Phoneme.phoneme_count())
        if not swa:
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer, factor=lr_reduce_factor, patience=lr_patience)

    def on_epoch_end(self):
        self.log('lr', self.optimizer.param_groups[0]['lr'], prog_bar=True)

    def training_step(self, batch, _):
        (specgrams, lengths), labels = batch
        outputs = self.model(specgrams, lengths, self.device)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    # def glottal_stops_to_silence(self, preds, labels):
    #     q_idx = Phoneme.symbol_to_index('q')
    #     sil_idx = Phoneme.symbol_to_index('sil')
    #     preds[preds == q_idx] = sil_idx
    #     labels[labels == q_idx] = sil_idx
    #     return preds, labels

    def foldPhonemeIndizes(self, indizes):
        for i in range(indizes.size(0)):
            symbol = Phoneme.phoneme_list[indizes[i]]
            indizes[i] = Phoneme.folded_phoneme_list.index(Phoneme.symbol_to_folded.get(symbol, symbol))
        return indizes

    def validation_step(self, batch, _):
        (specgrams, lengths), labels = batch
        specgrams = specgrams
        labels = labels
        outputs = self.model(specgrams, lengths, self.device)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        # preds, labels = self.glottal_stops_to_silence(preds, labels)
        preds = self.foldPhonemeIndizes(preds)
        labels = self.foldPhonemeIndizes(labels)
        acc = FM.accuracy(preds, labels)
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, _):
        (specgrams, lengths), labels = batch
        specgrams = specgrams
        outputs = self.model(specgrams, lengths, self.device)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        # preds, labels = self.glottal_stops_to_silence(preds, labels)
        preds = self.foldPhonemeIndizes(preds)
        labels = self.foldPhonemeIndizes(labels)
        acc = FM.accuracy(preds, labels)
        self.confmatMetric(torch.argmax(outputs, dim=1), labels)
        metrics = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        if swa:
            return self.optimizer

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
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, precision=16,
        stochastic_weight_avg=swa, auto_lr_find=auto_lr_find) 
        # resume_from_checkpoint='lightning_logs/epoch=55-step=3639.ckpt')

    if auto_lr_find:
        trainer.tune(model, dm)
    else:
        trainer.fit(model, dm)
        trainer.test(datamodule=dm)

        confmat = model.confmatMetric.compute()
        plt.figure(figsize=(15,10))
        class_names = Phoneme.folded_phoneme_list
        df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()