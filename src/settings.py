from pathlib import Path
import torch


DATA_PATH = Path('../data')
TIMIT_PATH = DATA_PATH / 'timit'

TRAIN_PATH = DATA_PATH / 'train'
TEST_PATH = DATA_PATH / 'test'
VAL_PATH = DATA_PATH / 'val'

SAMPLE_RATE = 16000
VAL_PERCENTAGE = 0.2
FRAME_LENGTH = 25  # in ms
STRIDE = 10  # in ms
SPECGRAM_HOP_LENGTH = 64
SPECGRAM_N_MELS = 64
CUDA0 = torch.device('cuda:0')
