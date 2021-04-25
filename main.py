import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
from utils import get_recording_paths, get_phonemes_from_file


class TimitDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, train):
        super(TimitDataset, self).__init__()
        self.root = root
        self.data = root / 'data'
        self.train = train
        self.recording_paths = get_recording_paths(root, train)
        self.n_recordings = len(self.recording_paths)
        

    def __getitem__(self, index):
        recording_path = self.recording_paths[index]
        wav_path = self.data / f'{recording_path}.WAV'
        pn_path = self.data / f'{recording_path}.PHN'
        waveform, sample_rate = torchaudio.load(wav_path)
        phonemes = get_phonemes_from_file(pn_path)
        return (waveform, sample_rate), phonemes

    def __len__(self):
        return self.n_recordings


timit_path = Path('../..//ML_DATA/timit')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

frame_length = 100 # in ms

train_dataset = TimitDataset(timit_path, train=True)
(waveform, sample_rate), phonemes = train_dataset[0]
samples_per_frame = sample_rate / 1000 * frame_length
print(samples_per_frame)