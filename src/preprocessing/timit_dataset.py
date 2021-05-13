import csv
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import librosa
import torch
import torchaudio

from settings import *
from phonemes import Phoneme


class TimitDataset(torch.utils.data.Dataset):
    
    def __init__(self, test):
        super().__init__()
        self.data = TIMIT_PATH / 'data'
        self.test = test
        self.recording_data = self.get_recording_data()
        self.n_recordings = len(self.recording_data)  

    def get_recording_data(self):
        recording_data = []
        train_test_str = "TEST" if self.test else "TRAIN"

        with open(TIMIT_PATH / f'{train_test_str.lower()}_data.csv') as file:
            next(file)
            for row in csv.reader(file, delimiter=','):
                # is train/test data and audiofile
                if row[1] == train_test_str and row[10] == 'TRUE':
                    path = row[5]
                    path_no_ext = path[0:path.index('.')]
                    speaker_id = row[3]
                    record = (path_no_ext, speaker_id)
                    recording_data.append(record)

        return recording_data

    def resample(self, waveform, sample_rate):
        if sample_rate != SAMPLE_RATE:
            waveform = waveform.detach().cpu().numpy()
            waveform = librosa.resample(waveform, sample_rate, SAMPLE_RATE)
            waveform = torch.tensor(waveform)
        return waveform

    def __getitem__(self, index):
        recording_path, speaker_id = self.recording_data[index]
        wav_path = self.data / f'{recording_path}.WAV'
        pn_path = self.data / f'{recording_path}.PHN'
        
        waveform, old_sample_rate = torchaudio.load(str(wav_path))
        # convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # resample
        waveform = self.resample(waveform, old_sample_rate)
        waveform = waveform[0]

        phonemes = Phoneme.get_phonemes_from_file(pn_path)
    
        return waveform, phonemes, speaker_id

    def __len__(self):
        return self.n_recordings