import csv

class Phoneme():
    phoneme_list = [
        'b', 'd', 'g', 'p', 't', 'k', 'dx', 'q', 'jh', 'ch', 's', 'sh', 'z',
        'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'en', 'eng', 'nx',
        'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'iy', 'ih', 'eh', 'ey', 'ae',
        'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er',
        'ax', 'ix', 'axr', 'ax-h', 'pau', 'epi', 'h#', 'bcl', 'dcl' ,'gcl',
        'pcl', 'tck', 'kcl', 'dcl', 'tcl'
    ]

    def __init__(self, start, stop, symbol):
        self.start = start
        self.stop = stop
        self.symbol = self.strip_digits(symbol)
    
    def strip_digits(self, s):
        length = len(s)
        return s[0:length-1] if s[length - 1].isdigit() else s

    def __str__(self):
        return f"{self.start}-{self.stop}: {self.symbol}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def symbol_to_index(cls, s):
        return cls.phoneme_list.index(s)

    @classmethod
    def index_to_symbol(cls, i):
        return cls.phoneme_list[i]
    
    @classmethod
    def phoneme_count(cls):
        return len(cls.phoneme_list)


def get_phonemes_from_file(path):
    phonemes = []
    with open(path) as pn_file:
        reader = csv.reader(pn_file, delimiter=' ')
        phonemes = [Phoneme(int(row[0]), int(row[1]), row[2])
                    for row in reader]
    return phonemes