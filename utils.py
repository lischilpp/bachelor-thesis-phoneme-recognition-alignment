import csv
from pathlib import Path
import torch
import torchaudio
import librosa
from math import floor, ceil
import numpy as np



class Phoneme():
    all_phonemes = [
        'b', 'd', 'g', 'p', 't', 'k', 'dx', 'q', 'jh', 'ch', 's', 'sh', 'z',
        'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'en', 'eng', 'nx',
        'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'iy', 'ih', 'eh', 'ey', 'ae',
        'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er',
        'ax', 'ix', 'axr', 'ax-h', 'pau', 'epi', 'h#', 'bcl', 'dcl' ,'gcl',
        'pcl', 'tck', 'kcl', 'dcl', 'tcl'
    ]

    def __init__(self, start, stop, symbol):
        self.start = start
        self.stop = stop
        self.symbol = self.strip_digits(symbol)
    
    def strip_digits(self, s):
        length = len(s)
        return s[0:length-1] if s[length - 1].isdigit() else s

    def __str__(self):
        return f"{self.start}-{self.stop}: {self.symbol}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def symbol_to_index(cls, s):
        return cls.all_phonemes.index(s)

    @classmethod
    def index_to_symbol(cls, i):
        return cls.all_phonemes[i]
    
    @classmethod
    def phoneme_count(cls):
        return len(cls.all_phonemes)


def get_phonemes_from_file(path):
    phonemes = []
    with open(path) as pn_file:
        reader = csv.reader(pn_file, delimiter=' ')
        phonemes = [Phoneme(int(row[0]), int(row[1]), row[2])
                    for row in reader]
    return phonemes

def get_labels_from_file(path, samples_per_frame, n_samples):
    phonemes = get_phonemes_from_file(path)
    print(phonemes)
    labels = []
    phon_idx = 0
    sample_idx = 0

    while True:
        if sample_idx + samples_per_frame >= n_samples:
            break
        phon = phonemes[phon_idx]
        if phon.stop - sample_idx < 0.5 * samples_per_frame and \
           phon_idx < len(phonemes) - 1:
            phon_idx += 1
            phon = phonemes[phon_idx]

        labels.append(Phoneme.symbol_to_index(phon.symbol))
        sample_idx += samples_per_frame

    return labels
    


def get_recording_paths(root, train):
    recording_paths = []
    train_test_str = "TRAIN" if train else "TEST"
    with open(root / f'{train_test_str.lower()}_data.csv') as file:
        next(file)
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            # is train/test data and audiofile
            if row[1] == train_test_str and row[10] == 'TRUE':
                path = row[5]
                path_no_ext = path[0:path.index('.')]
                recording_paths.append(path_no_ext)
    return recording_paths


class TimitDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, train, frame_length):
        super(TimitDataset, self).__init__()
        self.root = root
        self.data = root / 'data'
        self.train = train
        self.recording_paths = get_recording_paths(root, train)
        self.n_recordings = len(self.recording_paths)
        self.frame_length = frame_length

    def __getitem__(self, index):
        recording_path = self.recording_paths[index]
        wav_path = self.data / f'{recording_path}.WAV'
        pn_path = self.data / f'{recording_path}.PHN'

        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform[0]
        samples_per_frame = floor(sample_rate / 1000 * self.frame_length)
        n_samples = len(waveform)
        n_frames = ceil(n_samples / samples_per_frame)
        frames = waveform.unfold(0, samples_per_frame, samples_per_frame).reshape(-1, samples_per_frame, 1)

        labels = get_labels_from_file(pn_path, samples_per_frame, n_samples)
        return frames, labels

    def __len__(self):
        return self.n_recordings