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

train_ds = TimitDataset(root=timit_path, train=True, frame_length=50)

num_classes = Phoneme.phoneme_count()
num_epochs = 30
batch_size = 1
learning_rate = 0.0001

input_size = train_ds.specgram_height # train_dataset.samples_per_frame
hidden_size = 256
num_layers = 4


train_loader = torch.utils.data.DataLoader(dataset=train_ds, 
                                           batch_size=batch_size, 
                                           shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.gru(x, h0)
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
        specgrams = specgrams[0].to(device)
        labels = labels.flatten().to(device)

        outputs = model(specgrams)
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
