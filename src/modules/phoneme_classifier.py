import string
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import ConfusionMatrix
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from Levenshtein import distance as levenshtein_distance

from settings import *
from phonemes import Phoneme
from models.gru import GRUModel
from models.ligru import LiGRUModel
from models.phoneme_boundary_detector import PhonemeBoundaryDetector


class PhonemeClassifier(pl.LightningModule):

    def __init__(self, batch_size, lr, lr_patience, lr_reduce_factor):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_reduce_factor = lr_reduce_factor
        self.last_val_PER = float('inf')
        self.PER_too_high_count = 0
        self.num_classes = Phoneme.folded_phoneme_count()
        self.model = GRUModel(output_size=self.num_classes)
        self.phoneme_boundary_detector = PhonemeBoundaryDetector()
        self.criterion = nn.CrossEntropyLoss()#weight=Phoneme.folded_phoneme_weights)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, amsgrad=True)
        self.confmatMetric = ConfusionMatrix(num_classes=39)
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
        loss, acc = self.calculate_metrics(batch)
        self.log('train_loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'], prog_bar=True)
        return loss

    def remove_silences(self, preds, labels):
        sil_idx = Phoneme.folded_phoneme_list.index('sil')
        non_glottal_indices = torch.nonzero(labels.ne(sil_idx))
        preds = preds[non_glottal_indices]
        labels = labels[non_glottal_indices]
        return preds, labels

    def validation_step(self, batch, _):
        loss, acc, per = self.calculate_metrics(batch, per=True)
        metrics = {'val_loss': loss, 'val_FER': 1-acc, 'val_PER': per}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def validation_epoch_end(self, val_step_outputs):
        val_PER = sum([output['val_PER'] for output in val_step_outputs])/len(val_step_outputs)
        if val_PER > self.last_val_PER * 0.9999:
            self.PER_too_high_count += 1
        if self.PER_too_high_count > self.lr_patience:
            self.lr *= self.lr_reduce_factor
            self.init_cyclic_scheduler()
            self.trainer.lr_schedulers = self.trainer.configure_schedulers(
                [self.lr_scheduler],
                monitor='val_PER',
                is_manual_optimization=False)
            self.PER_too_high_count=0

        self.last_val_PER = val_PER

    def test_step(self, batch, _):
        loss, acc, per = self.calculate_metrics(batch, per=True, update_conf_mat=True)
        metrics = {'test_loss': loss, 'test_FER': 1-acc, 'test_PER': per}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]

    def calculate_metrics(self, batch, per=False, update_conf_mat=False):
        (fbank, lengths), labels = batch
        labels = self.remove_label_padding(labels, lengths)
        outputs = self.model(fbank, lengths, self.device)
        flattened_outputs = torch.cat(outputs)
        flattened_labels = torch.cat(labels)
        preds = [torch.argmax(output, dim=1) for output in outputs]
        preds_folded = self.foldGroupIndices(preds, lengths)
        labels_folded = self.foldGroupIndices(labels, lengths)

        pn_labels_correct = self.get_phoneme_labels(labels_folded, lengths)
        print(pad_sequence(outputs, batch_first=True).shape)
        boundaries = self.phoneme_boundary_detector(pad_sequence(outputs, batch_first=True), lengths, self.device)
        print(boundaries)
        exit()

        pn_labels_pred = self.get_phoneme_labels(preds_folded, lengths, boundaries.long().cpu())

        loss = self.criterion(flattened_outputs, flattened_labels)
        acc = FM.accuracy(torch.cat(preds_folded), torch.cat(labels_folded))

        if per:
            per_value = levenshtein_distance(
                self.intarray_to_unique_string(pn_labels_pred),
                self.intarray_to_unique_string(pn_labels_correct)) / len(pn_labels_correct)
            return loss, acc, per_value
        if update_conf_mat:
            self.confmatMetric(pn_labels_correct, pn_labels_pred)

        return loss, acc
    
    def remove_label_padding(self, labels, lengths):
        return [labels[i][:lengths[i]] for i in range(labels.size(0))]

    def get_phoneme_boundaries(self, labels, lengths):
        boundaries = []
        for i in range(len(labels)):
            boundaries.append([])
            for j in range(1, lengths[i]):
                 if labels[i][j] != labels[i][j-1]:
                     boundaries[i].append(j)
        return boundaries

    def get_phoneme_labels(self, segment_labels, lengths, boundaries=None):
        if boundaries is None:
            boundaries = self.get_phoneme_boundaries(segment_labels, lengths)
        pn_labels = []
        for i in range(len(segment_labels)):
            segments = torch.tensor_split(segment_labels[i], boundaries[i])
            for segment in segments:
                pn_labels.append(torch.argmax(torch.bincount(segment)).item())
        return pn_labels
    
    def intarray_to_unique_string(self, intarray):
        return ''.join([string.printable[i] for i in intarray])

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