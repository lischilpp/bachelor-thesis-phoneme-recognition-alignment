from phonemes import Phoneme, get_phonemes_from_file
from utils import encode_sentence, sentence_characters
import csv
import librosa
from math import floor, ceil
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torch

import torchaudio
import torchaudio.transforms as T


class TimitDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, train, frame_length, stride):
        super(TimitDataset, self).__init__()
        self.root = root
        self.data = root / 'data'
        self.train = train
        self.recording_paths = self.get_recording_paths(root, train)
        self.n_recordings = len(self.recording_paths)
        self.frame_length = frame_length
        self.stride = stride
        self.max_sentence_length = 100
        self.sentence_padded_size = self.max_sentence_length * len(sentence_characters)
        first_recording = self.data / f'{self.recording_paths[0]}.WAV'
        _, self.sampling_rate = torchaudio.load(first_recording)
        self.samples_per_frame = self.sampling_rate / 1000 * self.frame_length       
        self.samples_per_stride = self.sampling_rate / 1000 * self.stride    
        self.specgram_hop_length = 100
        self.specgram_n_mels = 64
        self.specgram_height = self.specgram_n_mels
        self.specgram_width = floor(self.samples_per_frame / self.specgram_hop_length) + 1


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


    def get_labels_from_file(self, path, n_samples):
        phonemes = get_phonemes_from_file(path)
        labels = []
        phon_idx = 0
        x = 0
        
        while x + self.samples_per_frame < n_samples:
            sample_idx = round(x)
            phon = phonemes[phon_idx]
            if phon.stop - sample_idx < 0.5 * self.samples_per_frame and \
            phon_idx < len(phonemes) - 1:
                phon_idx += 1
                phon = phonemes[phon_idx]

            labels.append(Phoneme.symbol_to_index(phon.symbol))
            x += self.samples_per_stride

        return torch.tensor(labels)


    def get_encoded_sentence_from_file(self, path):
        sentence = ''

        with open(path) as f:
            line = f.readline().rstrip()
            line2 = line[line.index(' ')+1:]
            sentence = line2[line2.index(' ') + 1:]
        
        if len(sentence) > self.max_sentence_length:
            print(f'sentence too long, check the dataset!, length={len(sentence)}')
            exit()
        
        return encode_sentence(sentence)


    def resample(self, waveform, sampling_rate):
        if sampling_rate != self.sampling_rate:
            waveform = waveform.detach().cpu().numpy()
            waveform = librosa.resample(waveform, sampling_rate, self.sampling_rate)
            waveform = torch.tensor(waveform)

        return waveform


    def waveform_to_specgrams(self, waveform, n_samples):
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_mels=self.specgram_n_mels,
            hop_length=self.specgram_hop_length
        )

        waveform_duration = n_samples / self.sampling_rate * 1000
        n_frames = int(floor((waveform_duration - self.frame_length)/self.stride)) + 1

        full_specgram = mel_spectrogram_transform(waveform).transpose(0, 1)
        full_specgram_width = full_specgram.size(0)
        stride_spec = self.stride / waveform_duration * full_specgram_width
        specgram_width = floor(full_specgram_width / waveform_duration * self.frame_length)

        specgrams = torch.zeros(n_frames, specgram_width, self.specgram_n_mels)
        i = 0
        x = 0
        while x + specgram_width < full_specgram_width:
            start = round(x)
            specgrams[i] = full_specgram[start : start + specgram_width]
            x += stride_spec
            i += 1

        return specgrams


    def __getitem__(self, index):
        recording_path = self.recording_paths[index]
        wav_path = self.data / f'{recording_path}.WAV'
        pn_path = self.data / f'{recording_path}.PHN'
        sentence_path = self.data / f'{recording_path}.TXT'

        # sentence = self.get_encoded_sentence_from_file(sentence_path)
        # sentence = sentence.view(1, -1)
        # sentence_padded = torch.zeros(1, self.sentence_padded_size)
        # sentence_padded[:, :sentence.size(1)] = sentence
        
        waveform, _ = torchaudio.load(wav_path)
        # convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = self.resample(waveform[0], self.sampling_rate)
        n_samples = len(waveform)
        
        specgrams = self.waveform_to_specgrams(waveform, n_samples)
        
        labels = self.get_labels_from_file(pn_path, n_samples)
        
        return specgrams, labels


    def __len__(self):
        return self.n_recordings