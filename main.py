import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
from utils import TimitDataset

timit_path = Path('../..//ML_DATA/timit')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = TimitDataset(root=timit_path, train=True, frame_length=50)
frames, labels = train_dataset[0]
print(len(frames[0]))
print(labels)
