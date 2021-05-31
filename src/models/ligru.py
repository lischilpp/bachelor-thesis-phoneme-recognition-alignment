import torch
import torch.nn as nn
import speechbrain as sb

from settings import *
from pprint import pprint



class LiGRUModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.rnn = sb.nnet.RNN.LiGRU(input_shape=[64, 500, 89],
                           hidden_size=512,
                           num_layers=3,
                           nonlinearity='tanh',
                           normalization='None',
                           bidirectional=True)
        self.fc3 = nn.Linear(self.rnn.hidden_size*2, self.output_size)

    def helper(self, confs):
        batch = confs[0]
        lengths = confs[1]
        out, _ = self.rnn(batch)
        out = self.fc3(out)
        predictions = torch.zeros(
            lengths.sum().item(), self.output_size)
        p = 0
        for i in range(batch.size(0)):
            predictions[p:p+lengths[i], :] = out[i][:lengths[i]]
            p += lengths[i]
        
        return predictions

    def forward(self, *confs):
        return helper(list(confs))
        
