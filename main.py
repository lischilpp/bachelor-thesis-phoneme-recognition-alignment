from utils import TimitDataset, Phoneme, sentence_characters
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt

timit_path = Path('../..//ML_DATA/timit')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = TimitDataset(root=timit_path, train=True, frame_length=50)

num_classes = Phoneme.phoneme_count()
num_epochs = 30
batch_size = 1
learning_rate = 0.00001

input_size = train_dataset.specgram_size #train_dataset.samples_per_frame
max_sentence_length = 100
sentence_padded_size = max_sentence_length * len(sentence_characters)
hidden_size = 128
num_layers = 2


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        in_size = input_size + sentence_padded_size + hidden_size
        self.i2h = nn.Linear(in_size, hidden_size)
        self.i2o = nn.Linear(in_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, sentence_padded, hidden_tensor):
        combined = torch.cat((input_tensor, sentence_padded, hidden_tensor), 1)
        
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).to(device)
    

model = RNN(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# train
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (sentence, specgrams, labels) in enumerate(train_loader): 
        loss = torch.tensor(0)
        specgrams = specgrams.to(device)
        labels = labels.flatten().to(device)

        if len(sentence) > max_sentence_length:
            print(f'sentence too long, length={len(sentence)}')
            continue

        hidden = model.init_hidden()
        
        sentence = sentence.view(1, -1)
        sentence_padded = torch.zeros(1, sentence_padded_size).to(device)
        sentence_padded[:, :sentence.size(1)] = sentence

        outputs = torch.zeros(specgrams.size(1), num_classes).to(device)
        for j in range(specgrams.size(1)):
            output, hidden = model(specgrams[0][j].view(1, -1), sentence_padded, hidden)
            outputs[j] = output.view(num_classes)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
