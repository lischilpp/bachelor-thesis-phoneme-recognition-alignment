import csv
from pathlib import Path
import torch
import librosa
from math import floor, ceil
import numpy as np
import string
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torchaudio
import torchaudio.transforms as T

sentence_characters = list(string.ascii_lowercase) + \
                 [',', ';', '.', '!', '?', ':', '-', '\'', '\"', ' ']

class Phoneme():
    phoneme_list = [
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
        return cls.phoneme_list.index(s)

    @classmethod
    def index_to_symbol(cls, i):
        return cls.phoneme_list[i]
    
    @classmethod
    def phoneme_count(cls):
        return len(cls.phoneme_list)


def get_phonemes_from_file(path):
    phonemes = []
    with open(path) as pn_file:
        reader = csv.reader(pn_file, delimiter=' ')
        phonemes = [Phoneme(int(row[0]), int(row[1]), row[2])
                    for row in reader]
    return phonemes

def get_labels_from_file(path, samples_per_frame, n_samples):
    phonemes = get_phonemes_from_file(path)
    labels = []
    phon_idx = 0
    sample_idx = 0

    while True:
        if sample_idx + samples_per_frame > n_samples:
            break
        phon = phonemes[phon_idx]
        if phon.stop - sample_idx < 0.5 * samples_per_frame and \
           phon_idx < len(phonemes) - 1:
            phon_idx += 1
            phon = phonemes[phon_idx]

        labels.append(Phoneme.symbol_to_index(phon.symbol))
        sample_idx += samples_per_frame

    return torch.tensor(labels)

def get_sentence_from_file(path):
    sentence = ''
    with open(path) as f:
        line = f.readline().rstrip()
        line2 = line[line.index(' ')+1:]
        sentence = line2[line2.index(' ') + 1:]
    return sentence

def one_hot_encode(e, l):
    return [1 if e == x else 0 for x in l]

def encode_sentence(s):
    s = [one_hot_encode(c, sentence_characters) for c in s.lower()]
    s = torch.FloatTensor(s).flatten()
    return s

def get_encoded_sentence_from_file(path):
    return encode_sentence(get_sentence_from_file(path))
    

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
        first_recording = self.data / f'{self.recording_paths[0]}.WAV'
        _, self.sampling_rate = torchaudio.load(first_recording)
        self.samples_per_frame = floor(self.sampling_rate / 1000 * self.frame_length)
        self.specgram_hop_length = 100

    def resample(self, waveform, sampling_rate):
        if sampling_rate != self.sampling_rate:
            waveform = waveform.detach().cpu().numpy()
            waveform = librosa.resample(waveform, sampling_rate, self.sampling_rate)
            waveform = torch.tensor(waveform)
        return waveform

    def __getitem__(self, index):
        recording_path = self.recording_paths[index]
        wav_path = self.data / f'{recording_path}.WAV'
        pn_path = self.data / f'{recording_path}.PHN'
        sentence_path = self.data / f'{recording_path}.TXT'

        waveform, sampling_rate = torchaudio.load(wav_path)
        waveform = self.resample(waveform[0], self.sampling_rate)
        
        n_samples = len(waveform)
        n_frames = ceil(n_samples / self.samples_per_frame)
        frames = waveform.unfold(0, self.samples_per_frame, self.samples_per_frame)

        # n_fft = 1024
        # win_length = None
        # hop_length = 42
        # n_mels = 128

        # mel_spectrogram = T.MelSpectrogram(
        #     sample_rate=self.sampling_rate,
        #     n_fft=n_fft,
        #     win_length=win_length,
        #     hop_length=hop_length,
        #     center=True,
        #     pad_mode="reflect",
        #     power=2.0,
        #     norm='slaney',
        #     onesided=True,
        #     n_mels=n_mels,
        # )
        # specgram = mel_spectrogram(frames[0])
        # print(specgram.shape)
        # exit()


        sentence = get_encoded_sentence_from_file(sentence_path)
        labels = get_labels_from_file(pn_path, self.samples_per_frame, n_samples)
        
        return sentence, frames, labels

    def __len__(self):
        return self.n_recordings