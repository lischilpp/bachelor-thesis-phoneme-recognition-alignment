import csv


class Phoneme():

    symbol_to_folded = {
        'ao': 'aa',
        **dict.fromkeys(['ax', 'ax-h'], 'ah'),
        'axr': 'er',
        'hv': 'hh',
        'ix': 'ih',
        'el': 'l',
        'em': 'm',
        **dict.fromkeys(['en', 'nx'], 'n'),
        'eng': 'ng',
        'zh': 'sh',
        'ux': 'uw',
        **dict.fromkeys(['pcl', 'tcl', 'kcl', 'bcl', 'dcl',
                         'gcl', 'h#', 'pau', 'epi'], 'sil')
    }

    folded_phoneme_list = [
        'iy', 'ih', 'eh', 'ae', 'ix', 'ax', 'ah', 'uw', 'uh',
        'ao', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow', 'l', 'el',
        'r', 'y', 'w', 'er', 'm', 'n', 'en', 'ng', 'ch', 'jh',
        'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k', 'z', 'zh',
        'v', 'f', 'th', 's', 'sh', 'hh', 'cl', 'vcl', 'epi', 'sil'
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
    def symbol_to_index(cls, s):
        return cls.folded_phoneme_list.index(s)

    @classmethod
    def index_to_symbol(cls, i):
        return cls.folded_phoneme_list[i]

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
                if symbol == 'q':  # skip glottal stop
                    continue
                symbol = Phoneme.symbol_to_folded.get(symbol, symbol)
                start = int(row[0])
                stop = int(row[1])
                phoneme = Phoneme(start, stop, symbol)
                phonemes.append(phoneme)

        return phonemes
