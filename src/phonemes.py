import csv


class Phoneme():

    symbol_to_folded = {
        'ao': 'aa',
        'axr': 'er',
        'el': 'l',
        'em': 'm',
        'eng': 'ng',
        'hv': 'hh',
        'ix': 'ih',
        'sh': 'zh',
        'ux': 'uw',
        **dict.fromkeys(['en', 'nx'], 'n'),
        **dict.fromkeys(['ax', 'ax-h'], 'ah'),
        **dict.fromkeys(['pcl', 'tcl', 'kcl',
                         'bcl', 'dcl', 'gcl',
                         'epi', 'h#', 'pau'], 'sil')
    }

    phoneme_list = [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr',
        'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh',
        'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g',
        'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',
        'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p',
        'pau', 'pcl', 'r', 's', 'sh', 't', 'tcl', 'th',
        'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'
    ]

    folded_phoneme_list = [
        'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh',
        'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy',
        'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r',
        's', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
    ]

    def __init__(self, start, stop, symbol):
        self.start = start
        self.stop = stop
        self.symbol = self.strip_digits(symbol)

    def strip_digits(self, s):
        length = len(s)
        return s[0:length-1] if s[length - 1].isdigit() else s

    def __str__(self):
        return f'{self.start}-{self.stop}: {self.symbol}'

    def __repr__(self):
        return self.__str__()

    @classmethod
    def phoneme_count(cls):
        return len(cls.folded_phoneme_list)

    @classmethod
    def get_phonemes_from_file(cls, path):
        phonemes = []
        with open(path) as pn_file:
            reader = csv.reader(pn_file, delimiter=' ')
            phonemes = []
            for row in reader:
                symbol = row[2]
                if symbol == 'q':
                    continue
                symbol = Phoneme.symbol_to_folded.get(symbol, symbol)
                start = int(row[0])
                stop = int(row[1])
                phoneme = Phoneme(start, stop, symbol)
                phonemes.append(phoneme)

        return phonemes