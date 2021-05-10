import torch
from settings import *


class TimitDataset(torch.utils.data.Dataset):
    
    def __init__(self, train):
        super(TimitDataset, self).__init__()
        train_test_str = 'train' if train else 'test'
        self.dataset_path = SAVED_DATASET_PATH / train_test_str
        self.record_paths = [x for x in self.dataset_path.glob('**/*') if x.is_file()]
        self.n_records = len(self.record_paths)

    def __getitem__(self, index):
        specgrams, labels = torch.load(self.record_paths[index])
        return specgrams, labels

    def __len__(self):
        return self.n_records