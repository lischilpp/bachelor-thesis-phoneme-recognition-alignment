import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math import floor
import random
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import pyrubberband as pyrb

from settings import *
from disk_dataset import DiskDataset
from preprocessing.frame_dataset import FrameDataset


def random_augment(record):
    waveform, phonemes = record
    time_factor = random.uniform(0.8, 1.25)
    pitch_factor = random.uniform(-4, 4) # 4 semitones
    #waveform = pyrb.time_stretch(waveform, SAMPLE_RATE, time_factor)
    #waveform = pyrb.pitch_shift(waveform, SAMPLE_RATE, pitch_factor)
    waveform = torch.tensor(waveform)
    inv_time_factor = 1 / time_factor
    # update phoneme boundaries
    for pn in phonemes:
        pn.start = floor(pn.start * inv_time_factor)
        pn.stop  = floor(pn.stop  * inv_time_factor)
    record = (waveform, phonemes)
    return record

TRAIN_AUGMENTED_PATH.mkdir(exist_ok=True)

train_ds = DiskDataset(TRAIN_RAW_PATH)
n_records = len(train_ds)

# n = 0
# for n in range(TRAIN_AUGMENTED_TOTAL_RECORDS):
#     i = n % n_records
#     record = random_augment(train_ds[i])
#     # waveform, phonemes = record
#     # torchaudio.save(filepath=str(TRAIN_AUGMENTED_PATH / f'test{i}.wav'), src=waveform.view(1,-1), sample_rate=SAMPLE_RATE)
#     torch.save(record, TRAIN_AUGMENTED_PATH / f'record{n}')
#     if n % 100 == 0:
#         percent = floor(n / TRAIN_AUGMENTED_TOTAL_RECORDS * 100)
#         print(f'{percent}%')

fds = FrameDataset(train_ds, augment=True)
specgrams, labels = fds[0]
print(specgrams.shape)

