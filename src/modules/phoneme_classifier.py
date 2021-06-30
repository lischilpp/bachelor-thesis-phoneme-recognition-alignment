import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix
import torchmetrics.functional as FM
import pytorch_lightning as pl
from Levenshtein import distance as levenshtein_distance

from settings import *
from phonemes import Phoneme
from models.gru import GRUModel
from models.transformer import TransformerModel
from schedulers.cyclic_plateau_scheduler import CyclicPlateauScheduler


class PhonemeClassifier(pl.LightningModule):

    def __init__(self, batch_size, lr, min_lr, lr_patience, lr_reduce_factor, steps_per_epoch):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = Phoneme.folded_phoneme_count()
        self.model = TransformerModel(self.num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=Phoneme.folded_phoneme_weights)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr)
        self.lr_scheduler = CyclicPlateauScheduler(initial_lr=self.lr,
                                                   min_lr=min_lr,
                                                   lr_patience=lr_patience,
                                                   lr_reduce_factor=lr_reduce_factor,
                                                   lr_reduce_metric='val_PER',
                                                   steps_per_epoch=steps_per_epoch,
                                                   optimizer=self.optimizer)
        self.confmatMetric = ConfusionMatrix(num_classes=Phoneme.folded_group_phoneme_count())

    def training_step(self, batch, step_index):
        self.model.train()
        loss = self.calculate_metrics(batch, mode='train')
        self.lr_scheduler.step(step_index)
        self.log('train_loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'], prog_bar=True)
        return loss

    def remove_silences(self, preds, labels):
        sil_idx = Phoneme.folded_phoneme_list.index('sil')
        non_glottal_indices = torch.nonzero(labels.ne(sil_idx))
        preds = preds[non_glottal_indices]
        labels = labels[non_glottal_indices]
        return preds, labels

    def validation_step(self, batch, step_index):
        self.model.eval()
        loss, acc, per = self.calculate_metrics(batch, mode='val')
        self.lr_scheduler.step(step_index)
        metrics = {'val_loss': loss, 'val_FER': 1-acc, 'val_PER': per}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def validation_epoch_end(self, val_step_outputs): # plateau scheduler
        self.lr_scheduler.validation_epoch_end(val_step_outputs)

    def test_step(self, batch, _):
        self.model.eval()
        loss, acc, per = self.calculate_metrics(batch, mode='test')
        metrics = {'test_loss': loss, 'test_FER': 1-acc, 'test_PER': per}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer

    def calculate_metrics(self, batch, mode):
        (fbank, lengths), labels = batch

        fbank = fbank / 4 + 2

        out = self.model(fbank, lengths, self.device)

        labels = self.remove_padding(labels, lengths)
        out = self.remove_padding(out, lengths)

        loss = self.criterion(torch.cat(out), torch.cat(labels))

        if mode == 'train':
            return loss
        
        preds = [o.softmax(1).argmax(1) for o in out]
        
        preds_folded = self.foldGroupIndices(preds, lengths)
        labels_folded = self.foldGroupIndices(labels, lengths)
        per_value = self.calculate_per(preds_folded, labels_folded, lengths)
        preds_folded = torch.cat(preds_folded)
        labels_folded = torch.cat(labels_folded)
        acc = FM.accuracy(preds_folded, labels_folded)
        
        if mode == 'val':
            return loss, acc, per_value

        self.confmatMetric(preds_folded, labels_folded)
        return loss, acc, per_value
    
    def calculate_per(self, preds, labels, lengths):
        pn_labels_pred = self.get_phoneme_labels(preds, lengths)
        pn_labels_correct = self.get_phoneme_labels(labels, lengths)
        batch_size = lengths.size(0)
        distances = torch.zeros(batch_size)
        for i in range(batch_size):
            distances[i] = levenshtein_distance(
                self.intarray_to_unique_string(pn_labels_pred[i]),
                self.intarray_to_unique_string(pn_labels_correct[i])) / len(pn_labels_correct[i])
        return torch.mean(distances)
    
    def remove_padding(self, tensor, lengths):
        return [tensor[i][:lengths[i]] for i in range(tensor.size(0))]

    def get_phoneme_labels(self, segment_labels, lengths):
        pn_labels = []
        for i in range(lengths.size(0)):
            pn_labels.append([segment_labels[i][0]])
            pn_labels[i].extend(segment_labels[i][j]
                             for j in range(lengths[i])
                             if segment_labels[i][j] != 48 and segment_labels[i][j] != segment_labels[i][j-1])
        return pn_labels
    
    def intarray_to_unique_string(self, intarray):
        return ''.join([chr(65+i) for i in intarray])

    def foldGroupIndices(self, indizes, lengths):
        for i in range(lengths.size(0)):
            for j in range(lengths[i]):
                symbol = Phoneme.folded_phoneme_list[indizes[i][j]]
                indizes[i][j] = Phoneme.folded_group_phoneme_list.index(Phoneme.symbol_to_folded_group.get(symbol, symbol))
        return indizes

    # hide v_num in progres bar
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict