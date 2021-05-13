from math import floor
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import random
import torch
import torchaudio.transforms as T
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

from settings import *
from phonemes import Phoneme


class FrameDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_ds, augment=False):
        super().__init__()
        self.root_ds = root_ds
        self.n_records = len(root_ds)
        self.samples_per_frame = SAMPLE_RATE / 1000 * FRAME_LENGTH     
        self.samples_per_stride = SAMPLE_RATE / 1000 * STRIDE
        self.augment = augment
        if augment:
            self.augment_transform = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=1),
            ])

    def get_frame_labels(self, phonemes, n_samples):
        labels = []
        phon_idx = 0
        x = 0
        while x + self.samples_per_frame < n_samples:
            sample_idx = floor(x)
            phon = phonemes[phon_idx]
            if phon.stop - sample_idx < 0.5 * self.samples_per_frame and \
            phon_idx < len(phonemes) - 1:
                phon_idx += 1
                phon = phonemes[phon_idx]

            labels.append(Phoneme.symbol_to_index(phon.symbol))
            x += self.samples_per_stride
        return torch.tensor(labels)

    def waveform_to_frames(self, waveform, n_samples):
        self.samples_per_frame = floor(self.samples_per_frame)
        frames = []
        x = 0
        i = 0
        while x + self.samples_per_frame < n_samples:
            idx = floor(x)
            frames.append(waveform[idx : idx + self.samples_per_frame])
            x += self.samples_per_stride
            i += 1
        return torch.stack(frames)

    def frames_to_spectrograms(self, frames):
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=SPECGRAM_N_MELS,
            hop_length=SPECGRAM_HOP_LENGTH
        )
        specgrams = T.AmplitudeToDB()(mel_spectrogram_transform(frames)).transpose(1, 2)
        return specgrams

    def random_augment(self, record):
        waveform, phonemes = record
        time_factor = random.uniform(0.8, 1.25)
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=1),
        ])
        waveform = augment(samples=waveform.numpy(), sample_rate=SAMPLE_RATE)
        TimeStretch(min_rate=time_factor, max_rate=time_factor, p=1)(waveform, SAMPLE_RATE)
        waveform = torch.tensor(waveform)
        # update phoneme boundaries
        for pn in phonemes:
            pn.start = floor(pn.start / time_factor)
            pn.stop  = floor(pn.stop  / time_factor)
        return waveform, phonemes

    def __getitem__(self, index):
        record = self.root_ds[index]
        if self.augment:
            record = self.random_augment(record)
        waveform, phonemes = record
        n_samples = len(waveform)
        waveform = waveform.float()
        frames = self.waveform_to_frames(waveform, n_samples)
        specgrams = self.frames_to_spectrograms(frames)
        labels = self.get_frame_labels(phonemes, n_samples)
        return specgrams, labels

    def __len__(self):
        return self.n_records