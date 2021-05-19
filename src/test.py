import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


from settings import *
from dataset.disk_dataset import DiskDataset
from dataset.frame_dataset import FrameDataset


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


ds = FrameDataset(DiskDataset(TRAIN_PATH))
fbank, labels = ds[0]

plt.imshow(fbank.transpose(0, 1))
plt.show()
