

from dataset.frame_dataset import FrameDataset
from dataset.disk_dataset import DiskDataset
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor

from settings import *


ds = DiskDataset(TRAIN_PATH)
waveform, labels = ds[0]

# torchaudio.save(filepath=str(DATA_PATH / f'test.wav'),
#                 src=waveform.view(1, -1), sample_rate=SAMPLE_RATE)

# print(torchaudio.sox_effects.effect_names())
# effects = [['remix', '-'],
#            ['rate', f'{SAMPLE_RATE}'], ['speed', f'{2:.5f}']]

effects = [
    ['remix', '-'],  # merge all the channels
    ['tempo', str(1.5)],
    ['pitch', str(-500)],
    ['rate', f'{SAMPLE_RATE}'],
]
waveform, _ = apply_effects_tensor(
    waveform.reshape(1, -1), SAMPLE_RATE, effects, channels_first=True)

torchaudio.save(filepath=str(DATA_PATH / f'test_x.wav'),
                src=waveform, sample_rate=SAMPLE_RATE)
