import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
from utils import TimitDataset, Phoneme

timit_path = Path('../..//ML_DATA/timit')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = TimitDataset(root=timit_path, train=True, frame_length=50)
frames, labels = train_dataset[0]
print(frames.shape)
print(len(labels))
print(labels)

exit()


num_classes = Phoneme.phoneme_count()
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = 1
sequence_length = 28
hidden_size = 128
num_layers = 2

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
