from pathlib import Path
import numpy as np
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import MultiplicativeLR

from dataset import TimitDataset
from phonemes import Phoneme
from utils import sentence_characters


timit_path = Path('../../ML_DATA/timit')
checkpoint_path = Path('checkpoint.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = Phoneme.phoneme_count()
num_epochs = 100
batch_size = 64
learning_rate = 0.001

num_layers = 2

def collate_fn(batch):
    # sentences = torch.cat([item[0] for item in batch])
    lengths = torch.tensor([item[0].size(0) for item in batch])
    frames = [item[0] for item in batch]
    frames = pad_sequence(frames, batch_first=True)
    labels = torch.cat([item[1] for item in batch])
    frame_data = (frames, lengths)
    return [frame_data, labels]

train_ds = TimitDataset(root=timit_path, train=True, frame_length=25, stride=10)
test_ds = TimitDataset(root=timit_path, train=False, frame_length=25, stride=10)
input_size = train_ds.specgram_height

train_loader = torch.utils.data.DataLoader(dataset=train_ds, 
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(dataset=test_ds, 
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        collate_fn=collate_fn)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.num_layers1 = 2
        self.num_layers2 = 2
        self.hidden_size1 = 128
        self.hidden_size2 = 128
        self.rnn1 = nn.RNN(train_ds.specgram_height, self.hidden_size1, self.num_layers1, batch_first=True, bidirectional=True, dropout=0.5)
        self.rnn2 = nn.GRU(self.hidden_size1*2, self.hidden_size2, self.num_layers1, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size2*2, num_classes)

    def forward(self, x, lengths):
        predictions = torch.zeros(lengths.sum().item(), num_classes).to(device)
        p = 0
        for i in range(x.size(0)):
            # feature extraction
            # single frame passed as sequence into BiRNN (many-to-one)
            h01 = torch.zeros(self.num_layers1*2, x.size(1), self.hidden_size1).to(device) 
            out, _ = self.rnn1(x[i], h01)
            out = out[:, -1, :]
            out2 = out.unsqueeze(0)
            # frame classification
            # features of all frames of an audiofile passed into BiGRU (many-to-many)
            h02 = torch.zeros(self.num_layers2*2, 1, self.hidden_size2).to(device)
            out2, _ = self.rnn2(out2, h02)
            for j in range(lengths[i]):
                predictions[p] = self.fc(out2[0][j])
                p += 1

        return predictions


class Main():
    def __init__(self):
        self.model = RNN().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) 
        lmbda = lambda epoch: 0.95
        self.scheduler = MultiplicativeLR(self.optimizer, lr_lambda=lmbda)
        self.last_epoch = 0
        if checkpoint_path.exists():
            self.checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self.last_epoch = self.checkpoint['epoch']
            print(f'loaded from checkpoint epoch={self.last_epoch}')

    def train(self):
        n_total_steps = len(train_loader)
        for epoch in range(self.last_epoch, num_epochs):
            for i, ((specgrams, lengths), labels) in enumerate(train_loader): 
                # sentence = sentence.view(1, -1).to(device)
                specgrams = specgrams.to(device)
                labels = labels.to(device)

                outputs = self.model(specgrams, lengths)
                
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            #if epoch % 5 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, checkpoint_path)

    def test(self):
        with torch.no_grad():
            n_correct = 0
            n_frames = 0
            n_class_correct = [0 for i in range(num_classes)]
            n_class_samples = [0 for i in range(num_classes)]
            for i, ((specgrams, lengths), labels) in enumerate(test_loader): 
                specgrams = specgrams.to(device)
                labels = labels.to(device)
                outputs = self.model(specgrams, lengths)
                _, predicted = torch.max(outputs, 1)
                n_frames += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            n_wrong = n_frames - n_correct
            fer = 100.0 * n_wrong / n_frames
            print(f'FER: {fer} %')

if __name__ == '__main__':
    main = Main()
    main.train()
    main.test()