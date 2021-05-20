import torchaudio.transforms as T
from dataset.data_augmentation import augment_record, augment_fbank
from phonemes import Phoneme
from settings import *
import torch
import warnings
import torchaudio.compliance.kaldi as kaldi
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )


class FrameDataset(torch.utils.data.Dataset):

    def __init__(self, root_ds, augment=False):
        super().__init__()
        self.root_ds = root_ds
        self.n_records = len(root_ds)
        self.augment = augment

    def get_frame_labels(self, phonemes, n_samples):
        labels = []
        phon_idx = 0
        label_idx = 0
        x = 0
        while x + SAMPLES_PER_FRAME < n_samples:
            phon = phonemes[phon_idx]
            # > 50% of phoneme in next frame
            if phon_idx < len(phonemes) - 1 and \
               phonemes[phon_idx+1].start - x < 0.5 * SAMPLES_PER_FRAME:
                # next phoneme
                phon_idx += 1
                phon = phonemes[phon_idx]
            labels.append(Phoneme.symbol_to_index(phon.symbol))
            label_idx += 1
            x += SAMPLES_PER_STRIDE
        return torch.tensor(labels)

    def create_fbank(self, waveform):
        fbank = kaldi.fbank(
            waveform,
            frame_length=FRAME_LENGTH / FRAME_RESOLUTION,
            frame_shift=STRIDE / FRAME_RESOLUTION,
            num_mel_bins=N_MELS)
        return fbank

    def __getitem__(self, index):
        record = self.root_ds[index]
        if self.augment:
            record = augment_record(record)
        waveform, phonemes = record
        n_samples = len(waveform)
        fbank = self.create_fbank(waveform.view(1, -1))
        if self.augment:
            fbank = augment_fbank(fbank)
        labels = self.get_frame_labels(phonemes, n_samples)
        fbank = fbank[:labels.size(0) * FRAME_RESOLUTION]
        return fbank, labels

    def __len__(self):
        return self.n_records
