import torchaudio.transforms as T
from dataset.data_augmentation import augment_record, augment_fbank
from phonemes import Phoneme
from settings import *
import torch
import warnings
import torchaudio.compliance.kaldi as kaldi


class FrameDataset(torch.utils.data.Dataset):

    def __init__(self, root_ds, augment=False):
        super().__init__()
        self.root_ds = root_ds
        self.n_records = len(root_ds)
        self.augment = augment

    def get_frame_labels_and_sentence(self, phonemes, n_samples):
        labels = []
        sentence = [Phoneme.symbol_to_folded_group_index(phonemes[0].symbol)]
        pn_idx = 0
        label_idx = 0
        x = 0
        while x + SAMPLES_PER_FRAME <= n_samples:
            pn = phonemes[pn_idx]
            # > 50% of phoneme in next frame
            pn_idx_updated = False
            while pn_idx < len(phonemes) - 1 and \
               phonemes[pn_idx+1].start - x < 0.5 * SAMPLES_PER_STRIDE:
                # next phoneme
                pn_idx += 1
                pn = phonemes[pn_idx]
                pn_idx_updated = True
            if pn_idx_updated:
                sentence.append(Phoneme.symbol_to_folded_group_index(pn.symbol))
            labels.append(Phoneme.folded_phoneme_list.index(pn.symbol))
            label_idx += 1
            x += SAMPLES_PER_STRIDE
        return torch.tensor(labels), sentence

    def create_fbank(self, waveform):
        fbank = kaldi.fbank(
            waveform,
            frame_length=FRAME_LENGTH,
            frame_shift=STRIDE,
            num_mel_bins=N_MELS)
        return fbank

    def __getitem__(self, i):
        record = self.root_ds[i]
        if self.augment:
            record = augment_record(record)
        waveform, phonemes = record
        fbank = self.create_fbank(waveform.view(1, -1))
        if self.augment:
            fbank = augment_fbank(fbank)
        labels, sentence = self.get_frame_labels_and_sentence(phonemes, len(waveform))
        return fbank, labels, sentence

    def __len__(self):
        return self.n_records
