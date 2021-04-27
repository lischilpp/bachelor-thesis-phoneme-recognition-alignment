from utils import TimitDataset, Phoneme
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt

timit_path = Path('../..//ML_DATA/timit')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = TimitDataset(root=timit_path, train=True, frame_length=50)
frames, labels = train_dataset[0]


num_classes = Phoneme.phoneme_count()
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = train_dataset.samples_per_frame
hidden_size = 128
num_layers = 2

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=1, 
                                           shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        frame_count = x.size(1)
        # print(x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        out, _ = self.rnn(x, h0)
        
        predictions = torch.zeros(frame_count, num_classes)

        for i in range(frame_count):
            predictions[i] = self.fc(out[:, i, :])

        return predictions
        

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# train
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (frames, labels) in enumerate(train_loader): 
        frames = frames.to(device)
        labels = labels.flatten().to(device)
        
        outputs = model(frames).to(device)
        # print('#####')
        # print(frames.shape)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
