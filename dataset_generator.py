from math import floor, ceil
import csv
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import librosa
import torch
import torchaudio
import torchaudio.transforms as T

from phonemes import Phoneme, get_phonemes_from_file


class TimitDatasetGenerator(torch.utils.data.Dataset):
    
    def __init__(self, root, train, frame_length, stride):
        super(TimitDataset, self).__init__()
        self.root = root
        self.data = root / 'data'
        self.train = train
        self.recording_paths = self.get_recording_paths(root, train)
        self.n_recordings = len(self.recording_paths)
        self.frame_length = frame_length
        self.stride = stride
        first_recording_path = self.data / f'{self.recording_paths[0]}.WAV'
        _, self.sampling_rate = torchaudio.load(first_recording_path)
        self.samples_per_frame = self.sampling_rate / 1000 * self.frame_length       
        self.samples_per_stride = self.sampling_rate / 1000 * self.stride    
        self.specgram_width = floor(self.samples_per_frame / SPECGRAM_HOP_LENGTH) + 1


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


    def resample(self, waveform, sampling_rate):
        if sampling_rate != self.sampling_rate:
            waveform = waveform.detach().cpu().numpy()
            waveform = librosa.resample(waveform, sampling_rate, self.sampling_rate)
            waveform = torch.tensor(waveform)

        return waveform


    def waveform_to_frames(self, waveform, n_samples):
        self.samples_per_frame = floor(self.samples_per_frame)
        waveform_duration = n_samples / self.sampling_rate * 1000
        n_frames = int(floor((waveform_duration - self.frame_length) / self.stride)) + 1
        frames = torch.zeros(n_frames, self.samples_per_frame)
        x = 0
        i = 0
        while x + self.samples_per_frame < n_samples:
            idx = floor(x)
            frames[i] = waveform[idx : idx + self.samples_per_frame]
            x += self.samples_per_stride
            i += 1
        return frames


    def frames_to_spectrograms(self, frames):
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_mels=SPECGRAM_N_MELS,
            hop_length=SPECGRAM_HOP_LENGTH
        )

        n_frames = len(frames)
        specgrams = T.AmplitudeToDB()(mel_spectrogram_transform(frames)).transpose(1, 2)

        return specgrams


    def __getitem__(self, index):
        recording_path = self.recording_paths[index]
        wav_path = self.data / f'{recording_path}.WAV'
        pn_path = self.data / f'{recording_path}.PHN'
        
        waveform, _ = torchaudio.load(wav_path)
        # convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = self.resample(waveform[0], self.sampling_rate)
        n_samples = len(waveform)
        frames = self.waveform_to_frames(waveform, n_samples)
        specgrams = self.frames_to_spectrograms(frames)

        labels = self.get_labels_from_file(pn_path, n_samples)
    
        return specgrams, labels


    def __len__(self):
        return self.n_recordings