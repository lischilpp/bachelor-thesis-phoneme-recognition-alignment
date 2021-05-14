

from dataset.frame_dataset import FrameDataset
from dataset.disk_dataset import DiskDataset
from settings import TRAIN_RAW_PATH
import librosa
import torchaudio

from settings import *

TRAIN_AUGMENTED_PATH.mkdir(exist_ok=True)

ds = DiskDataset(TRAIN_RAW_PATH)
waveform, labels = ds[0]


torchaudio.save(filepath=str(TRAIN_AUGMENTED_PATH / f'test.wav'), src=waveform.view(1,-1), sample_rate=SAMPLE_RATE)

waveform = librosa.effects.time_stretch(waveform.numpy(), 1.1)
torchaudio.save(filepath=str(TRAIN_AUGMENTED_PATH / f'test_x.wav'), src=torch.from_numpy(waveform).view(1,-1), sample_rate=SAMPLE_RATE)
