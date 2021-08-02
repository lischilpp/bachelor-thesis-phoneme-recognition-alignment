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
        self.i = 0

    def get_frame_labels(self, phonemes, n_samples):
        labels = []
        phon_idx = 0
        label_idx = 0
        x = 0
        while x + SAMPLES_PER_FRAME <= n_samples:
            phon = phonemes[phon_idx]
            # > 50% of phoneme in next frame
            if phon_idx < len(phonemes) - 1 and \
               phonemes[phon_idx+1].start - x < 0.5 * SAMPLES_PER_FRAME:
                # next phoneme
                phon_idx += 1
                phon = phonemes[phon_idx]
            labels.append(phon.symbol_idx)
            label_idx += 1
            x += SAMPLES_PER_STRIDE
        return torch.tensor(labels)

    def create_fbank(self, waveform):
        fbank = kaldi.fbank(
            waveform,
            frame_length=FRAME_LENGTH,
            frame_shift=STRIDE,
            num_mel_bins=N_MELS)
        return fbank

    def get_phoneme_sentence_from_list(self, phonemes):
        sentence = torch.zeros(len(phonemes), dtype=torch.int32)
        for i, pn in enumerate(phonemes):
            s = Phoneme.folded_phoneme_list[pn.symbol_idx]
            s = Phoneme.symbol_to_folded_group.get(s, s)
            idx = Phoneme.folded_group_phoneme_list.index(s)
            sentence[i] = idx
        return sentence

    def __getitem__(self, index):
        record = self.root_ds[index]
        if self.augment:
            record = augment_record(record)
        waveform, phonemes, _ = record
        fbank = self.create_fbank(waveform.view(1, -1))
        if self.augment:
            fbank = augment_fbank(fbank)
        sentence = self.get_phoneme_sentence_from_list(phonemes)
        labels = self.get_frame_labels(phonemes, len(waveform))
        return fbank, labels, sentence

    def __len__(self):
        return self.n_records
