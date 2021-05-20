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
        self.data = TIMIT_PATH / 'data'
        self.test = test
        self.recording_paths = self.get_recording_paths()
        self.n_recordings = len(self.recording_paths)

    def get_recording_paths(self):
        recording_paths = []
        train_test_str = "TEST" if self.test else "TRAIN"

        with open(TIMIT_PATH / f'{train_test_str.lower()}_data.csv') as file:
            for row in csv.DictReader(file, delimiter=','):
                # is train/test data & not spoken dialect & audiofile
                if row['test_or_train'] == train_test_str and \
                        not row['filename'].startswith('SA') and \
                        row['is_converted_audio'] == 'TRUE':
                    path = row['path_from_data_dir']
                    path_no_ext = path[0:path.index('.')]
                    recording_paths.append(path_no_ext)
        return recording_paths

    def __getitem__(self, index):
        recording_path = self.recording_paths[index]
        wav_path = self.data / f'{recording_path}.WAV'
        pn_path = self.data / f'{recording_path}.PHN'
        waveform, _ = torchaudio.load(wav_path)
        waveform = waveform[0]
        phonemes = Phoneme.get_phonemes_from_file(pn_path)
        return waveform, phonemes

    def __len__(self):
        return self.n_recordings
