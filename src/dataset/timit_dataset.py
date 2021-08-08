from pathlib import PurePath
from phonemes import Phoneme
from settings import *
import torchaudio
import torch
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )


class TimitDataset(torch.utils.data.Dataset):

    core_test_set_speakers = [
        'MDAB0', 'MTAS1', 'MJMP0', 'MLLL0',
        'MBPM0', 'MCMJ0', 'MGRT0', 'MJLN0',
        'MWBT0', 'MWEW0', 'MLNT0', 'MTLS0',
        'MKLT0', 'MJDH0', 'MNJM0', 'MPAM0',
        'FELC0', 'FPAS0', 'FPKT0', 'FJLM0',
        'FNLP0', 'FMGD0', 'FDHC0', 'FMLD0'
    ]

    def __init__(self, test):
        super().__init__()
        self.test = test
        self.recording_paths = self.get_recording_paths()
        self.n_records = len(self.recording_paths)

    def get_recording_paths(self):
        recording_paths = []
        train_test_str = "TEST" if self.test else "TRAIN"
        for path in (TIMIT_PATH / train_test_str).rglob('*.WAV.wav'):
            path_entries = PurePath(path).parts
            speaker_id = path_entries[5]
            filename = path_entries[6]
            is_sa_file = filename.startswith('SA')
            if (EXCLUDE_SA_FILES or self.test) and is_sa_file:
                continue
            recording_paths.append({
                'path': str(path.relative_to(TIMIT_PATH))[:-8],
                'is_sa_file': is_sa_file,
                'is_core_test': speaker_id in TimitDataset.core_test_set_speakers})
        return recording_paths

    def __getitem__(self, index):
        recording_path = self.recording_paths[index]
        wav_path = TIMIT_PATH / f'{recording_path["path"]}.WAV'
        pn_path = TIMIT_PATH / f'{recording_path["path"]}.PHN'
        txt_path = TIMIT_PATH / f'{recording_path["path"]}.TXT'
        waveform, _ = torchaudio.load(wav_path)
        waveform = waveform[0]
        phonemes = Phoneme.get_phonemes_from_file(pn_path)
        return waveform, phonemes, recording_path['is_sa_file'], recording_path['is_core_test']

    def __len__(self):
        return self.n_records
