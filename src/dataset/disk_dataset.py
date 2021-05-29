from math import floor
import torch
from settings import *


class DiskDataset(torch.utils.data.Dataset):
    
    def __init__(self, path):
        super().__init__()
        self.record_paths = [x for x in path.glob('*') if x.is_file()]
        self.n_records = len(self.record_paths)

    def __getitem__(self, index):
        item = torch.load(self.record_paths[index])
        return item

    def __len__(self):
        return self.n_records

    @classmethod
    def write(cls, ds, save_path, exclude_sa_files=True):
        n_records = len(ds)
        save_path.mkdir(exist_ok=True)
        for (i, entry) in enumerate(ds):
            waveform, phonemes, is_sa_file = entry
            if exclude_sa_files and is_sa_file:
                continue
            entry = waveform, phonemes
            torch.save(entry, save_path / f'record{i}')
            if i % 100 == 0:
                percent = floor(i / n_records * 100)
                print(f'{percent}%')
    