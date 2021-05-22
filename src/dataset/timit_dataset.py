from phonemes import Phoneme
from settings import *
import torchaudio
import torch
import csv
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )


class TimitDataset(torch.utils.data.Dataset):

    def __init__(self, test):
        super().__init__()
        self.test = test
        self.recording_paths = self.get_recording_paths()
        self.n_recordings = len(self.recording_paths)

    def get_recording_paths(self):
        recording_paths = []
        train_test_str = "TEST" if self.test else "TRAIN"

        for path in (TIMIT_PATH / train_test_str).rglob('*.WAV'):
            recording_paths.append(str(path.relative_to(TIMIT_PATH))[:-4])

        return recording_paths

    def __getitem__(self, index):
        recording_path = self.recording_paths[index]
        wav_path = TIMIT_PATH / f'{recording_path}.WAV'
        pn_path = TIMIT_PATH / f'{recording_path}.PHN'
        waveform, _ = torchaudio.load(wav_path)
        waveform = waveform[0]
        phonemes = Phoneme.get_phonemes_from_file(pn_path)
        return waveform, phonemes

    def __len__(self):
        return self.n_recordings
