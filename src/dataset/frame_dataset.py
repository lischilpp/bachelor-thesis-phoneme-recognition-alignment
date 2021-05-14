import torchaudio.transforms as T
from torchaudio.sox_effects import apply_effects_tensor
from phonemes import Phoneme
from settings import *
import torch
import random
from math import floor
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )


class FrameDataset(torch.utils.data.Dataset):

    def __init__(self, root_ds, augment=False):
        super().__init__()
        self.root_ds = root_ds
        self.n_records = len(root_ds)
        self.samples_per_frame = SAMPLE_RATE / 1000 * FRAME_LENGTH
        self.samples_per_stride = SAMPLE_RATE / 1000 * STRIDE
        self.augment = augment

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
            frames.append(waveform[idx: idx + self.samples_per_frame])
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
        speed_factor = random.uniform(0.8, 1.25)
        pitch_factor = random.uniform(-4, 4)  # 4 semitones
        effects = [
            ['remix', '-'],  # merge all the channels
            ['tempo', str(speed_factor)],
            ['pitch', str(pitch_factor * 100)],
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

    def __getitem__(self, index):
        record = self.root_ds[index]
        if self.augment:
            record = self.random_augment(record)
        waveform, phonemes = record
        n_samples = len(waveform)
        frames = self.waveform_to_frames(waveform, n_samples)
        specgrams = self.frames_to_spectrograms(frames)
        labels = self.get_frame_labels(phonemes, n_samples)
        return specgrams, labels

    def __len__(self):
        return self.n_records
