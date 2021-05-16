import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt


from settings import *
from dataset.disk_dataset import DiskDataset

width = 10
height = 4
kernel_width = 2
kernel_height = 2


def f(x):
    return torch.max(x.flatten())


class Kernel():
    def __init__(self, func, width, height, padding):
        self.func = func
        self.width = width
        self.height = height
        self.padding = nn.ZeroPad2d(padding)

    def apply(self, input):
        rows = self.padding(input).unfold(0, self.height, self.height).unfold(
            1, self.width, self.width)
        out = torch.tensor([[self.func(e) for e in row] for row in rows])
        return out


kernel = Kernel(f, kernel_width, kernel_height, 0)
matrix = torch.arange(width * height).view(height, width)
print(matrix.shape)
print(matrix)
out = kernel.apply(matrix)
print(out.shape)
print(out)

# def waveform_to_spectrogram(waveform):
#     mel_spectrogram_transform = T.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_mels=SPECGRAM_N_MELS,
#         hop_length=SPECGRAM_HOP_LENGTH
#     )
#     specgram = T.AmplitudeToDB()(mel_spectrogram_transform(waveform))
#     return specgram


# def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
#     fig, axs = plt.subplots(1, 1)
#     axs.set_title(title or 'Spectrogram (db)')
#     axs.set_ylabel(ylabel)
#     axs.set_xlabel('frame')
#     im = axs.imshow(spec, origin='lower', aspect=aspect)
#     if xmax:
#         axs.set_xlim((0, xmax))
#     fig.colorbar(im, ax=axs)
#     plt.show()


# ds = DiskDataset(TRAIN_PATH)
# waveform, phonemes = ds[0]
# print(waveform.shape)
# specgram = waveform_to_spectrogram(waveform)
# print(specgram.shape)
# plot_spectrogram(specgram, SAMPLE_RATE)
