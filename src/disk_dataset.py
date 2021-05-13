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
    def write(cls, ds, save_path):
        save_path.mkdir(exist_ok=True)
        for (i, entry) in enumerate(ds):
            torch.save(entry, save_path / f'record{i}')
    