import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix
import torchmetrics.functional as FM
import pytorch_lightning as pl
from Levenshtein import distance as levenshtein_distance
from dtw import dtw
import matplotlib.pyplot as plt

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
        self.model = GRUModel(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
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
        loss, recognition_accuracy, recognition_per = self.calculate_metrics(batch, mode='val')
        self.lr_scheduler.step(step_index)
        metrics = {
            'val_loss': loss,
            'val_FER': 1-recognition_accuracy,
            'val_PER': recognition_per}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def validation_epoch_end(self, val_step_outputs): # plateau scheduler
        self.lr_scheduler.validation_epoch_end(val_step_outputs)

    def test_step(self, batch, _):
        self.model.eval()
        loss, recognition_accuracy, recognition_per, alignment_accuracy = self.calculate_metrics(batch, mode='test')
        metrics = {
            'test_loss': loss,
            'test_FER': 1-recognition_accuracy,
            'test_PER': recognition_per,
            'test_alignment_accuracy': alignment_accuracy}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer

    def fold_probabilities(self, out, batch_size, lengths):
        out_folded = [torch.zeros(lengths[i], 39, device=self.device) for i in range(batch_size)]
        for i in range(48):
            symbol = Phoneme.folded_phoneme_list[i]
            symbol = Phoneme.symbol_to_folded_group.get(symbol, symbol)
            new_idx = Phoneme.folded_group_phoneme_list.index(symbol)

            for j in range(batch_size):
                out_folded[j][:, new_idx] += out[j][:, i]
        return out_folded

    def get_alignments(self, out_folded, sentences, lengths, batch_size):
        preds_folded = [torch.zeros(l, dtype=torch.int32, device=self.device) for l in lengths]
        probability_distance = lambda x, y: 1 - x[y]
        for i in range(batch_size):
            _, _, _, path = dtw(out_folded[i], sentences[i], dist=probability_distance)
            for j in range(lengths[i]):
                preds_folded[i][j] = sentences[i][path[1][j]]
        return preds_folded
            

    def get_phoneme_boundary_indices(self, phoneme_lists, lengths, batch_size):
        boundary_indices = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(1, lengths[i]):
                if phoneme_lists[i][j-1] != phoneme_lists[i][j]:
                    boundary_indices[i].append(j-1)
        return boundary_indices

    def calculate_alignment_accuracy(self, preds_folded, labels_folded, lengths, batch_size):
        predicted_boundary_indices = self.get_phoneme_boundary_indices(preds_folded, lengths, batch_size)
        actual_boundary_indices = self.get_phoneme_boundary_indices(labels_folded, lengths, batch_size)

        n_boundaries = sum([len(b) for b in actual_boundary_indices])
        n_correct = 0
        for i in range(batch_size):
            # print('---')
            # print(preds_folded[i])
            n_correct_i = sum([1 for x, y in zip(predicted_boundary_indices[i], actual_boundary_indices[i]) if abs(x-y) < 2])
            n_correct += n_correct_i
            # print(labels_folded[i])
            # for j in range(len(actual_boundary_indices[i])):
            #     if abs(predicted_boundary_indices[i][j] - actual_boundary_indices[i][j]) > 1:
            #         print(labels_folded[i][actual_boundary_indices[i][j]])
            # print(len(actual_boundary_indices[i]))
            # print(n_correct_i / len(actual_boundary_indices[i]))
        
        return n_correct / n_boundaries

    def calculate_metrics(self, batch, mode):
        (fbank, lengths), labels, sentences = batch

        out = self.model(fbank)

        batch_size = fbank.size(0)
        labels = self.remove_padding(labels, lengths)
        out = self.remove_padding(out, lengths)

        loss = self.criterion(torch.cat(out), torch.cat(labels))

        if mode == 'train':
            return loss

        preds = [o.argmax(1) for o in out]
        preds_folded = self.foldGroupIndices(preds, lengths)
        labels_folded = self.foldGroupIndices(labels, lengths)
        recognition_per = self.calculate_per(preds_folded, labels_folded, lengths)
        recognition_accuracy = FM.accuracy(torch.cat(preds_folded), torch.cat(labels_folded))

        if mode == 'val':
            return loss, recognition_accuracy, recognition_per
        
        out = [x.softmax(0) for x in out]
        out_folded = self.fold_probabilities(out, batch_size, lengths)
        preds_folded = self.get_alignments(out_folded, sentences, lengths, batch_size)
        alignment_accuracy = self.calculate_alignment_accuracy(preds_folded, labels_folded, lengths, batch_size)
        self.confmatMetric(torch.cat(preds_folded), torch.cat(labels_folded))

        return loss, recognition_accuracy, recognition_per, alignment_accuracy

    
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
                             if segment_labels[i][j] != segment_labels[i][j-1])
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
