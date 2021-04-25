import csv
from pathlib import Path


class Phoneme():
    def __init__(self, start, end, symbol):
        self.start = start
        self.end = end
        self.symbol = symbol

    def __str__(self):
        return f"{self.start}-{self.end}: {self.symbol}"

    def __repr__(self):
        return self.__str__()


def get_phonemes_from_file(path):
    phonemes = []
    with open(path) as pn_file:
        reader = csv.reader(pn_file, delimiter=' ')
        phonemes = [Phoneme(int(row[0]), int(row[1]), row[2])
                    for row in reader][1:-1]
    return phonemes

def get_recording_paths(root, train):
    recording_paths = []
    train_test_str = "TRAIN" if train else "TEST"
    with open(root / f'{train_test_str.lower()}_data.csv') as file:
        next(file)
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            # is train/test data and audiofile
            if row[1] == train_test_str and row[10] == 'TRUE':
                path = row[5]
                path_no_ext = path[0:path.index('.')]
                recording_paths.append(path_no_ext)
    return recording_paths