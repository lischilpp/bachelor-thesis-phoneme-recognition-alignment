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
        self.augment = augment

    def get_frame_labels(self, phonemes, n_frames, n_samples):
        labels = torch.zeros(n_frames, dtype=torch.long)
        phon_idx = 0
        label_idx = 0
        x = 0
        while x + SAMPLES_PER_FRAME < n_samples:
            phon = phonemes[phon_idx]
            # < 50% of phoneme in current frame
            if phon.stop - x < 0.5 * SAMPLES_PER_STRIDE and \
                    phon_idx < len(phonemes) - 1:
                # next phoneme
                phon_idx += 1
                phon = phonemes[phon_idx]
            labels[label_idx] = Phoneme.symbol_to_index(phon.symbol)
            label_idx += 1
            x += SAMPLES_PER_STRIDE
        return labels

    def augment_record(self, record):
        waveform, phonemes = record
        speed_factor = random.uniform(0.85, 1.25)
        pitch_factor = random.uniform(-4, 4)  # 4 semitones
        effects = [
            ['remix', '-'],  # merge all channels
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

    def waveform_to_spectrogram(self, waveform):
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=SPECGRAM_N_MELS,
            hop_length=SPECGRAM_HOP_LENGTH
        )
        n_frames = len(
            waveform) // SPECGRAM_HOP_LENGTH // SPECTROGRAM_FRAME_LENGTH
        specgram = T.AmplitudeToDB()(
            mel_spectrogram_transform(waveform))[:, :n_frames*SPECTROGRAM_FRAME_LENGTH].transpose(0, 1)
        return specgram, n_frames

    def augment_specgram(self, specgram):
        # gaussian noise
        noise = torch.randn(specgram.shape)
        specgram += 0.005*noise
        # frequency mask
        if random.random() < 0.2:
            torch.random.manual_seed(4)
            masking = T.FrequencyMasking(freq_mask_param=80)
            specgram = masking(specgram)
        # time mask
        for i in range(0, specgram.size(0) - SPECTROGRAM_FRAME_LENGTH, SPECTROGRAM_FRAME_LENGTH):
            if random.random() > 0.2:
                continue
            frame = specgram.narrow(0, i, SPECTROGRAM_FRAME_LENGTH)
            torch.random.manual_seed(4)
            masking = T.TimeMasking(time_mask_param=2)
            specgram[i:i+SPECTROGRAM_FRAME_LENGTH, :] = masking(frame)
        return specgram

    def __getitem__(self, index):
        record = self.root_ds[index]
        if self.augment:
            record = self.augment_record(record)
        waveform, phonemes = record
        n_samples = len(waveform)
        specgram, n_frames = self.waveform_to_spectrogram(waveform)
        if self.augment:
            specgram = self.augment_specgram(specgram)
        labels = self.get_frame_labels(phonemes, n_frames, n_samples)
        return specgram, labels

    def __len__(self):
        return self.n_records
