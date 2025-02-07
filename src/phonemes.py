import csv
import torch

from settings import *


class Phoneme():

    phoneme_list = [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr',
        'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh',
        'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g',
        'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',
        'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p',
        'pau', 'pcl', 'r', 's', 'sh', 't', 'tcl', 'th',
        'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh', 'q'
    ]

    symbol_to_folded = {
        'ux': 'uw',
        'axr': 'er',
        'ax-h': 'ax',
        'em': 'm',
        'nx': 'n',
        'eng': 'ng',
        'hv': 'hh',
        **dict.fromkeys(['pcl', 'tcl', 'kcl', 'qcl'], 'cl'),
        **dict.fromkeys(['bcl', 'dcl', 'gcl'], 'vcl'),
        **dict.fromkeys(['h#', '#h', 'pau'], 'sil')
    }

    folded_phoneme_list = [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch',
        'cl', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er',
        'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k',
        'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
        'sil', 't', 'th', 'uh', 'uw', 'v', 'vcl', 'w',
        'y', 'z', 'zh'
    ]
    
    # CURRENTLY NOT USED: can be used as class weights
    folded_phoneme_weights = torch.tensor([
        1.05, 1.17, 1.02, 1.96, 4.25, 0.94, 1.52, 1.4, 3.83, 0.29,
        1.16, 1.25, 1.49, 0.88, 3.82, 5.77, 4.0, 0.69, 1.33, 1.38,
        2.28, 1.73, 0.52, 0.49, 0.6, 2.87, 0.76, 0.63, 0.83, 0.42,
        2.45, 1.75, 8.17, 1.15, 0.61, 0.47, 2.75, 0.23, 0.75, 4.07,
        6.13, 1.49, 1.51, 0.5, 1.26, 2.67, 0.81, 5.75
    ])

    symbol_to_folded_group = {
        **dict.fromkeys(['cl', 'vcl', 'epi'], 'sil'),
        'el': 'l',
        'en': 'n',
        'sh': 'zh',
        'ao': 'aa',
        'ih': 'ix',
        'ah': 'ax'
    }

    folded_group_phoneme_list = [
        'aa', 'ae', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh',
        'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ix', 'iy',
        'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r',
        's', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
    ]

    word_to_phoneme_dict = {}

    def __init__(self, start, stop, symbol):
        self.start = start
        self.stop = stop
        self.symbol = symbol

    def __str__(self):
        return f'{self.start}-{self.stop}: {self.symbol}'

    def __repr__(self):
        return self.__str__()

    @classmethod
    def strip_digits(cls, s):
        length = len(s)
        return s[0:length-1] if s[length - 1].isdigit() else s

    @classmethod
    def folded_phoneme_count(cls):
        return len(cls.folded_phoneme_list)

    @classmethod
    def folded_group_phoneme_count(cls):
        return len(cls.folded_group_phoneme_list)

    @classmethod
    def get_phonemes_from_file(cls, path):
        phonemes = []
        with open(path) as pn_file:
            reader = csv.reader(pn_file, delimiter=' ')
            phonemes = []
            for row in reader:
                symbol = cls.strip_digits(row[2])
                if symbol == 'q':
                    continue
                symbol = cls.symbol_to_folded.get(symbol, symbol)
                start = int(row[0])
                stop = int(row[1])
                phoneme = Phoneme(start, stop, symbol)
                phonemes.append(phoneme)

        return phonemes

    @classmethod
    def symbol_to_folded_group_index(cls, symbol):
        return Phoneme.folded_group_phoneme_list.index(Phoneme.symbol_to_folded_group.get(symbol, symbol))