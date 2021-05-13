import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import *
from preprocessing.frame_dataset import FrameDataset
from disk_dataset import DiskDataset


def dataset_to_frame_dataset(ds_path, fds_path):
    fds = FrameDataset(DiskDataset(ds_path))
    DiskDataset.write(fds, fds_path)

print('processing train dataset...')
dataset_to_frame_dataset(TRAIN_AUGMENTED_PATH, TRAIN_AUGMENTED_FRAMES_PATH)

print('processing val dataset...')
dataset_to_frame_dataset(VAL_RAW_PATH, VAL_RAW_FRAMES_PATH)

print('processing test dataset...')
dataset_to_frame_dataset(TEST_RAW_PATH, TEST_RAW_FRAMES_PATH)

print('--- DONE ---')