from pathlib import Path
import torch

TIMIT_PATH = Path('../../ML_DATA/timit')
CHECKPOINT_PATH = Path('checkpoint.pt')
SAVED_DATASET_PATH = Path('dataset')
FRAME_LENGTH = 25 # in ms
STRIDE = 10 # in ms
SPECGRAM_HOP_LENGTH = 128
SPECGRAM_N_MELS = 64
CUDA0 = torch.device('cuda:0')