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
    TRAIN_PHONEMES_PATH.mkdir(exist_ok=True)

    pn_indizes = {}
    for (waveform, phonemes, speaker_id) in ds:
        if not speaker_id in pn_indizes:
            pn_indizes[speaker_id] = {symbol : 0 for symbol in Phoneme.phoneme_list}
        speaker_path = TRAIN_PHONEMES_PATH / speaker_id
        speaker_path.mkdir(exist_ok=True)
        for pn in phonemes:
            pn_path = speaker_path / pn.symbol
            pn_path.mkdir(exist_ok=True)
            pn_waveform = waveform[pn.start : pn.stop]
            save_path = str(pn_path / f'pn{pn_indizes[speaker_id][pn.symbol]}.wav')
            torchaudio.save(filepath=save_path, src=pn_waveform.view(1,-1), sample_rate=SAMPLE_RATE)
            pn_indizes[speaker_id][pn.symbol] += 1


print('processing train dataset...')
train_ds = DiskDataset(TRAIN_RAW_PATH)
phonemes_to_disk(train_ds)
print('--- DONE ---')