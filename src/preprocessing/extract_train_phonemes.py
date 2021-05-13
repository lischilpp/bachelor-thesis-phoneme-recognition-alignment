import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# disable C++ extension warning
import warnings
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torchaudio

from settings import *
from phonemes import Phoneme
from disk_dataset import DiskDataset


def phonemes_to_disk(ds):
    TRAIN_PHONEMES_PATH.mkdir(exist_ok=True, parents=True)
    for pn in Phoneme.phoneme_list:
        (TRAIN_PHONEMES_PATH / pn).mkdir(exist_ok=True)

    pn_indizes = {symbol : 0 for symbol in Phoneme.phoneme_list}
    for (waveform, phonemes) in ds:
        for pn in phonemes:
            pn_waveform = waveform[0][pn.start : pn.stop]
            save_path = TRAIN_PHONEMES_PATH / pn.symbol / f'pn{pn_indizes[pn.symbol]}.wav'
            torchaudio.save(filepath=save_path, src=pn_waveform.view(1,-1), sample_rate=SAMPLE_RATE)
            pn_indizes[pn.symbol] += 1


print('processing train dataset...')
train_ds = DiskDataset(TRAIN_RAW_PATH)
phonemes_to_disk(train_ds)
print('--- DONE ---')