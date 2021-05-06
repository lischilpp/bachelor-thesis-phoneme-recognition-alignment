from dataset import TimitDataset
from phonemes import Phoneme
from utils import sentence_characters
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt

timit_path = Path('../..//ML_DATA/timit')
checkpoint_path = Path('checkpoint.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = Phoneme.phoneme_count()
num_epochs = 15
batch_size = 4
learning_rate = 0.0001

hidden_size = 128
num_layers = 2

def collate_fn(batch):
    sentences = torch.stack([item[0] for item in batch])
    frames = torch.cat([item[1] for item in batch])
    labels = torch.cat([item[2] for item in batch])
    return [sentences, frames, labels]

train_ds = TimitDataset(root=timit_path, train=True, frame_length=25)
test_ds = TimitDataset(root=timit_path, train=False, frame_length=25)
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
        self.num_layers1 = 1
        self.num_layers2 = 1
        self.hidden_size1 = 128
        self.hidden_size2 = 128
        self.rnn1 = nn.RNN(train_ds.specgram_height, self.hidden_size1, self.num_layers1, batch_first=True)
        self.rnn2 = nn.GRU(self.hidden_size1, self.hidden_size2, self.num_layers1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size2, num_classes)

    def forward(self, x):
        h01 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(device) 
        out, _ = self.rnn1(x, h01)
        out = out[:, -1, :]
        out2 = out.unsqueeze(0)
        h02 = torch.zeros(self.num_layers2, 1, self.hidden_size2).to(device)
        out2, _ = self.rnn2(out2, h02)
        predictions = torch.zeros(x.size(0), num_classes).to(device)
        for i in range(out2.size(1)):
            predictions[i] = self.fc(out2[0][i])
        return predictions
    

model = RNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

last_epoch = 0
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    print(f'loaded from checkpoint epoch={last_epoch}')

# train
n_total_steps = len(train_loader)
for epoch in range(last_epoch, num_epochs):
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

torch.save({'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    for i, (sentence, specgrams, labels) in enumerate(test_loader): 
        specgrams = specgrams.to(device)
        labels = labels.to(device)
        outputs = model(specgrams)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
