from math import floor
import csv
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torch
import torchaudio.transforms as T

from settings import *
from phonemes import Phoneme


class FrameDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_ds):
        super().__init__()
        self.root_ds = root_ds
        self.n_records = len(root_ds)
        self.samples_per_frame = SAMPLE_RATE / 1000 * FRAME_LENGTH     
        self.samples_per_stride = SAMPLE_RATE / 1000 * STRIDE   

    def get_recording_paths(self, root, train):
        recording_paths = []
        train_test_str = "TRAIN" if train else "TEST"
        with open(root / f'{train_test_str.lower()}_data.csv') as file:
            next(file)
            for row in csv.reader(file, delimiter=','):
                # is train/test data and audiofile
                if row[1] == train_test_str and row[10] == 'TRUE':
                    path = row[5]
                    path_no_ext = path[0:path.index('.')]
                    recording_paths.append(path_no_ext)
        return recording_paths

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

    def __getitem__(self, index):
        waveform, phonemes, _ = self.root_ds[index]
        n_samples = len(waveform)
        frames = self.waveform_to_frames(waveform, n_samples)
        specgrams = self.frames_to_spectrograms(frames)
        labels = self.get_frame_labels(phonemes, n_samples)
        return specgrams, labels

    def __len__(self):
        return self.n_records