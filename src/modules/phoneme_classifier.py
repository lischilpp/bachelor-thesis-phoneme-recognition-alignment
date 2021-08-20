import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix, F1
import torchmetrics.functional as FM
import pytorch_lightning as pl
from Levenshtein import distance as levenshtein_distance
from dtw import dtw
import matplotlib.pyplot as plt
import math

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
                                                   min_improve_factor=0.97,
                                                   lr_patience=lr_patience,
                                                   lr_reduce_factor=lr_reduce_factor,
                                                   lr_reduce_metric='val_loss',
                                                   steps_per_epoch=steps_per_epoch,
                                                   optimizer=self.optimizer)
        self.confmat_metric = ConfusionMatrix(num_classes=Phoneme.folded_group_phoneme_count())
        self.f1_metric = F1(num_classes=Phoneme.folded_group_phoneme_count())

    def training_step(self, batch, step_index):
        self.model.train()
        loss = self.calculate_metrics(batch, mode='train')
        self.lr_scheduler.training_step(step_index)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('lr', self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, step_index):
        self.model.eval()
        loss, recognition_accuracy, recognition_per = self.calculate_metrics(batch, mode='val')
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_FER', 1-recognition_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_PER', recognition_per, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_FER': 1-recognition_accuracy, 'val_PER': recognition_per}

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.tensor([o['loss'] for o in outputs]))
        self.logger.experiment.add_scalars('losses', {'train_loss': loss}, global_step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        avg_metrics = {key: torch.mean(torch.tensor([o[key] for o in outputs]))
                       for key in outputs[0].keys()}
        self.lr_scheduler.validation_epoch_end(avg_metrics)
        self.logger.experiment.add_scalars('losses', {'val_loss': avg_metrics['val_loss']}, global_step=self.current_epoch)


    def test_step(self, batch, _):
        self.model.eval()
        loss, recognition_accuracy, recognition_per, alignment_accuracy, f1_score = self.calculate_metrics(batch, mode='test')
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_FER', 1-recognition_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_PER', recognition_per, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', f1_score, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_alignment_accuracy', alignment_accuracy, on_epoch=True, prog_bar=True, logger=True)

    def calculate_metrics(self, batch, mode):
        (fbank, lengths), labels, sentences = batch

        out = self.model(fbank)

        batch_size = fbank.size(0)
        labels = self.remove_padding(labels, lengths)
        out    = self.remove_padding(out, lengths)
        out_flat    = torch.cat(out)
        labels_flat = torch.cat(labels)
        
        loss = self.criterion(out_flat, labels_flat)
        # loss_weights = self.get_phoneme_boundary_loss_weights(labels_flat)
        # loss = self.element_weighted_loss(out_flat, labels_flat, loss_weights)

        if mode == 'train':
            return loss

        preds = [o.argmax(1) for o in out]
        preds_folded = self.foldGroupIndices(preds, lengths)
        labels_folded = self.foldGroupIndices(labels, lengths)
        recognition_per = self.calculate_per(preds_folded, labels_folded, lengths)
        preds_flat = torch.cat(preds_folded)
        labels_flat = torch.cat(labels_folded)
        preds_flat, labels_flat = self.remove_silences(preds_flat, labels_flat)
        recognition_accuracy = FM.accuracy(preds_flat, labels_flat)

        if mode == 'val':
            return loss, recognition_accuracy, recognition_per

        f1_score = self.f1_metric(preds_flat, labels_flat)
        out = [x.softmax(0) for x in out]
        out_folded = self.fold_probabilities(out, batch_size, lengths)
        preds_folded = self.get_alignments(out_folded, sentences, lengths, batch_size)
        alignment_accuracy = self.calculate_alignment_accuracy(preds_folded, labels_folded, lengths, batch_size, sentences)
        self.confmat_metric(preds_flat, labels_flat)

        return loss, recognition_accuracy, recognition_per, alignment_accuracy, f1_score

    def remove_silences(self, preds, labels):
        sil_idx = Phoneme.folded_phoneme_list.index('sil')
        non_glottal_indices = torch.nonzero(labels.ne(sil_idx))
        preds = preds[non_glottal_indices]
        labels = labels[non_glottal_indices]
        return preds, labels

    def configure_optimizers(self):
        return self.optimizer

    def fold_probabilities(self, out, batch_size, lengths):
        out_folded = [torch.zeros(lengths[i], Phoneme.folded_group_phoneme_count(), device=self.device) for i in range(batch_size)]
        for i in range(Phoneme.folded_phoneme_count()):
            symbol = Phoneme.folded_phoneme_list[i]
            symbol = Phoneme.symbol_to_folded_group.get(symbol, symbol)
            new_idx = Phoneme.folded_group_phoneme_list.index(symbol)

            for j in range(batch_size):
                out_folded[j][:, new_idx] += out[j][:, i]
        return out_folded

    def get_alignments(self, out_folded, sentences, lengths, batch_size):
        preds_folded = [torch.zeros(l, dtype=torch.int32, device=self.device) for l in lengths]
        # b = 2
        # k = 2#math.pow(b, -k*x[y])-pow(b,-k)#1 - pow(x[y], 0.01) # math.exp(-x[y])
        for i in range(batch_size):
            probability_distance = lambda x, y: 1 - pow(x[y], 0.001)
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

    def calculate_alignment_accuracy(self, preds_folded, labels_folded, lengths, batch_size, sentences):
        predicted_boundary_indices = self.get_phoneme_boundary_indices(preds_folded, lengths, batch_size)
        actual_boundary_indices = self.get_phoneme_boundary_indices(labels_folded, lengths, batch_size)

        n_boundaries = sum([len(b) for b in actual_boundary_indices])
        n_correct = 0
        for i in range(batch_size):
            # print('---')
            # print(preds_folded[i])
            if len(predicted_boundary_indices[i]) != len(actual_boundary_indices[i]):
                print('--')
                # print(predicted_boundary_indices[i])
                # print(actual_boundary_indices[i])
                print(sentences[i])
                print(labels_folded[i])
            n_correct_i = sum([1 for x, y in zip(predicted_boundary_indices[i], actual_boundary_indices[i]) if abs(x-y) < 2])

            # accuracy_i = n_correct_i / len(actual_boundary_indices[i])
            # if accuracy_i < 0.8:
            #     print('----')
            #     print(sentences[i])
            #     print(preds_folded[i])
            #     print(labels_folded[i])

            n_correct += n_correct_i
            # print(labels_folded[i])
            # for j in range(len(actual_boundary_indices[i])):
            #     if abs(predicted_boundary_indices[i][j] - actual_boundary_indices[i][j]) > 1:
            #         print(labels_folded[i][actual_boundary_indices[i][j]])
            # print(len(actual_boundary_indices[i]))
            # print(n_correct_i / len(actual_boundary_indices[i]))
        
        return n_correct / n_boundaries

    def element_weighted_loss(self, preds, labels, weights):
        m = torch.nn.LogSoftmax(dim=1)
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(m(preds), labels) * weights
        return loss.sum() / weights.sum()

    def get_phoneme_boundary_loss_weights(self, labels_flat):
        loss_weights = torch.ones(len(labels_flat), device=self.device)
        for i in range(2, len(labels_flat)-1):
            if labels_flat[i-1] != labels_flat[i]:
                loss_weights[i-2] = 50
                loss_weights[i-1] = 100
                loss_weights[i]   = 100
                loss_weights[i+1]   = 50
        return loss_weights

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
                             for j in range(1, lengths[i])
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

    # hide v_num in progres bar in terminal window
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
