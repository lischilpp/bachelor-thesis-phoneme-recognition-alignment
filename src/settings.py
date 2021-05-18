from pathlib import Path
import torch


DATA_PATH = Path('../data')
TIMIT_PATH = DATA_PATH / 'timit'
TRAIN_PATH = DATA_PATH / 'train'
TEST_PATH = DATA_PATH / 'test'
VAL_PATH = DATA_PATH / 'val'

CUDA0 = torch.device('cuda:0')
VAL_PERCENTAGE = 0.2
AUGMENT_DATASET = True
"""
STRIDE & SPECGRAM_HOP_LENGTH should be set so that
SAMPLES_PER_STRIDE / SPECGRAM_HOP_LENGTH
is an integer (otherwise a spectrogram frame boundaries would overlap)
"""
FRAME_LENGTH = 25  # in ms
STRIDE = 10  # in ms
SPECGRAM_HOP_LENGTH = 40
SPECGRAM_N_MELS = 89
SAMPLE_RATE = 16000

# semi-constants (depend on previous constants)
SAMPLES_PER_FRAME = SAMPLE_RATE / 1000 * FRAME_LENGTH
SAMPLES_PER_STRIDE = SAMPLE_RATE / 1000 * STRIDE
SPECTROGRAM_FRAME_LENGTH = int(SAMPLES_PER_STRIDE / SPECGRAM_HOP_LENGTH)
