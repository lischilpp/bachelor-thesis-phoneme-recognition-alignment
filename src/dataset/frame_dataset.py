import torchaudio.transforms as T
import torchaudio.compliance.kaldi as kaldi
from torchaudio.sox_effects import apply_effects_tensor
from phonemes import Phoneme
from settings import *
import torch
import random
from math import floor
import warnings
import matplotlib.pyplot as plt
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )


class FrameDataset(torch.utils.data.Dataset):

    def __init__(self, root_ds, augment=False):
        super().__init__()
        self.root_ds = root_ds
        self.n_records = len(root_ds)
        self.augment = augment

    def get_frame_labels(self, phonemes, n_frames, n_samples):
        labels = torch.zeros(n_frames, dtype=torch.long)
        phon_idx = 0
        label_idx = 0
        x = 0
        while x + SAMPLES_PER_FRAME < n_samples:
            phon = phonemes[phon_idx]
            # < 50% of phoneme in current frame
            if phon.stop - x < 0.5 * SAMPLES_PER_STRIDE and \
                    phon_idx < len(phonemes) - 1:
                # next phoneme
                phon_idx += 1
                phon = phonemes[phon_idx]
            labels[label_idx] = Phoneme.symbol_to_index(phon.symbol)
            label_idx += 1
            x += SAMPLES_PER_STRIDE
        return labels

    def augment_record(self, record):
        waveform, phonemes = record
        speed_factor = random.uniform(0.85, 1.25)
        pitch_factor = random.uniform(-4, 4)  # 4 semitones
        effects = [
            ['remix', '-'],  # merge all channels
            ['tempo', f'{speed_factor:.2f}'],
            ['pitch', f'{pitch_factor * 100:.2f}'],
            ['rate', f'{SAMPLE_RATE}'],
        ]
        waveform, _ = apply_effects_tensor(
            waveform.reshape(1, -1), SAMPLE_RATE, effects, channels_first=True)
        waveform = waveform[0]
        # update phoneme boundaries
        for pn in phonemes:
            pn.start = floor(pn.start / speed_factor)
            pn.stop = floor(pn.stop / speed_factor)
        return waveform, phonemes

    def augment_specgram(self, specgram):
        # gaussian noise
        noise = torch.randn(specgram.shape)
        specgram += 0.005*noise
        # frequency mask
        if random.random() < 0.2:
            torch.random.manual_seed(4)
            masking = T.FrequencyMasking(freq_mask_param=80)
            specgram = masking(specgram)
        # time mask
        for i in range(0, specgram.size(0) - FBANK_FRAME_LENGTH, FBANK_FRAME_LENGTH):
            if random.random() > 0.2:
                continue
            frame = specgram.narrow(0, i, FBANK_FRAME_LENGTH)
            torch.random.manual_seed(4)
            masking = T.TimeMasking(time_mask_param=2)
            specgram[i:i+FBANK_FRAME_LENGTH, :] = masking(frame)
        return specgram

    def create_fbank(self, waveform):
        fbank = kaldi.fbank(
            waveform,
            frame_length=FBANK_FRAME_LENGTH,
            frame_shift=FBANK_STRIDE,
            num_mel_bins=N_MELS)
        return fbank

    def __getitem__(self, index):
        record = self.root_ds[index]
        if self.augment:
            record = self.augment_record(record)
        waveform, phonemes = record
        n_samples = len(waveform)
        fbank = self.create_fbank(waveform.view(1, -1))
        n_frames = fbank.size(0) // FRAME_RESOLUTION
        fbank = fbank[:n_frames*FRAME_RESOLUTION]
        labels = self.get_frame_labels(phonemes, n_frames, n_samples)
        return fbank, labels

    def __len__(self):
        return self.n_records
