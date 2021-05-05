from dataset import TimitDataset
from phonemes import Phoneme
from utils import sentence_characters
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt

timit_path = Path('../..//ML_DATA/timit')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = TimitDataset(root=timit_path, train=True, frame_length=25)

num_classes = Phoneme.phoneme_count()
num_epochs = 15
batch_size = 4
learning_rate = 0.0001

input_size = train_ds.specgram_height # train_dataset.samples_per_frame
hidden_size = 128
num_layers = 2

def collate_fn(batch):
    sentences = torch.stack([item[0] for item in batch])
    frames = torch.cat([item[1] for item in batch])
    labels = torch.cat([item[2] for item in batch])
    return [sentences, frames, labels]
    

train_loader = torch.utils.data.DataLoader(dataset=train_ds, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           collate_fn=collate_fn)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# train
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (sentence, specgrams, labels) in enumerate(train_loader): 
        # sentence = sentence.view(1, -1).to(device)
        specgrams = specgrams.to(device)
        labels = labels.to(device)

        outputs = model(specgrams)
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
