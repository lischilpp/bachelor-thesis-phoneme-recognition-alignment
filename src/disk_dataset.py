import torch
from settings import *


class DiskDataset(torch.utils.data.Dataset):
    
    def __init__(self, path):
        super().__init__()
        self.record_paths = [x for x in path.glob('**/*') if x.is_file()]
        self.n_records = len(self.record_paths)

    def __getitem__(self, index):
        item = torch.load(self.record_paths[index])
        return item

    def __len__(self):
        return self.n_records