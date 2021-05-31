import random
import torch
import torchaudio.transforms as T
from torchaudio.sox_effects import apply_effects_tensor

from settings import *


gaussian_noise_probability = 0
gaussian_noise_strength = 0.001

pitch_shift_probability = 0
pitch_shift_min = -3
pitch_shift_max = 3  # 4 semitones

time_stretch_probability = 0
time_stretch_min = 0.5
time_stretch_max = 1.5

frequency_mask_probability = 1
frequency_mask_height = 20

time_mask_probability = 1
time_mask_frame_probability = 0.3


def augment_record(record):
    waveform, phonemes = record
    effects = []
    # gaussian noise
    if random.random() < gaussian_noise_probability:
        waveform += gaussian_noise_strength * torch.randn(len(waveform))
    # pitch shift
    if random.random() < pitch_shift_probability:
        pitch_factor = floor2(random.uniform(
            pitch_shift_min, pitch_shift_max))
        effects.append(['pitch', f'{pitch_factor * 100}'])
    # tempo shift
    if random.random() < time_stretch_probability:
        tempo_factor = floor2(random.uniform(time_stretch_min, time_stretch_max))
        effects.append(['tempo', f'{tempo_factor}'])
        # update phoneme boundaries
        for pn in phonemes:
            pn.start = int(round(pn.start / tempo_factor))
            pn.stop = int(round(pn.stop / tempo_factor))
    # apply effects to record
    if len(effects) > 0:
        effects.append(['rate', f'{SAMPLE_RATE}'])
        waveform, _ = apply_effects_tensor(
            waveform.reshape(1, -1), SAMPLE_RATE, effects, channels_first=True)
        waveform = waveform[0]
    return waveform, phonemes


def augment_fbank(fbank):
    fbank = fbank.transpose(0, 1)
    # frequency mask
    if random.random() < frequency_mask_probability:
        masking = T.FrequencyMasking(freq_mask_param=frequency_mask_height)
        fbank = masking(fbank)
    # time mask
    if random.random() < time_mask_probability:
        for i in range(0, fbank.size(1) - 1, 1):
            if random.random() < time_mask_frame_probability:
                fbank[:,i:i+1] = 0
    fbank = fbank.transpose(0, 1)
    return fbank


def floor2(x):  # floor to 2 decimals
    return x // 0.01 / 100
