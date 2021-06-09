
# from settings import *
# from dataset.timit_dataset import TimitDataset
# from dataset.disk_dataset import DiskDataset
# from dataset.frame_dataset import FrameDataset
# from phonemes import Phoneme
# import matplotlib.pyplot as plt
# import torch.nn.functional as F


    
# ds = FrameDataset(DiskDataset(TRAIN_PATH), augment=True)
# fbank, labels = ds[0]

# plt.imshow(fbank.transpose(0, 1))
# plt.show()


# import torchaudio

# wav_path = 'test.wav'
# ds = DiskDataset(TRAIN_PATH)
# waveform, phonemes = ds[400]

# waveform, _ = torchaudio.load(wav_path)
# waveform = (waveform[0]*1e4+336).int()
# print(waveform[20000:20020])
# print("Shape:", tuple(waveform.shape))
# print(f" - Max:     {waveform.max().item():6.3f}")
# print(f" - Min:     {waveform.min().item():6.3f}")

import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
counter = Counter()
for line in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter)

def data_process(raw_text_iter):
  data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                       dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
    data, targets = get_batch(train_data, i)
    print(data[0])
    exit()