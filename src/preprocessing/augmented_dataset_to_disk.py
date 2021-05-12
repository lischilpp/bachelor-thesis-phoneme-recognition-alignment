import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import string
from random import randrange
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torchaudio

from settings import *


def filter_chars(str, allowed):
    return ''.join([c for c in str if c in allowed])

def get_sentences():
    word_chars = list(string.ascii_lowercase) + ['\'', '-', ' ']
    sentences = []
    with open(TIMIT_PATH / 'PROMPTS.TXT') as file:
        for line in file:
            if line.strip() == '' or line[0] == ';':
                continue
            sentence = line[:line.index('(')].strip()
            sentence = filter_chars(sentence.lower(), word_chars)
            sentences.append(sentence)
    return sentences

def create_word_to_phonemes_dict():
    word_to_phonemes = {}
    phoneme_chars = list(string.ascii_lowercase) + [' ']
    with open(TIMIT_PATH / 'TIMITDIC.TXT') as file:
        for line in file:
            line = line.strip()
            if line == '' or line[0] == ';':
                continue
            ws1 = line.index(' ')
            word = line[:ws1]
            pn_str = line[ws1 + 3:].strip()[:-1]
            phonemes = filter_chars(pn_str, phoneme_chars).split(' ')
            word_to_phonemes[word] = phonemes
    return word_to_phonemes

def sentences_to_phoneme_lists(sentences, word_to_phonemes):
    phon_list_sentences = []
    for sentence in sentences:
        words = sentence.split(' ')
        # timit dictionary is incomplete: exclude untranslatable sentences
        bad_sentence = False
        for word in words:
            if not word in word_to_phonemes:
                bad_sentence = True
                break
        if bad_sentence:
            continue
        phon_list_sentence = sum([word_to_phonemes[word] for word in words], [])
        phon_list_sentences.append(phon_list_sentence)
    return phon_list_sentences

def get_phoneme_file_counts():
    phoneme_file_counts = {}
    for path in TRAIN_PHONEMES_PATH.glob('*'):
        if path.is_dir():
            phoneme_file_counts[path.name] = sum(1 for x in path.glob('*') if x.is_file())
    return phoneme_file_counts

def get_random_phoneme_waveform(pn, phoneme_file_counts):
    i = randrange(phoneme_file_counts[pn])
    wav_path = TRAIN_PHONEMES_PATH / pn / f'pn_{i}.wav'
    waveform, _ = torchaudio.load(wav_path)
    waveform = waveform[0]
    return waveform


TRAIN_AUGMENTED_PATH.mkdir(exist_ok=True)

sentences = get_sentences()
word_to_phonemes = create_word_to_phonemes_dict()
phon_list_sentences = sentences_to_phoneme_lists(sentences, word_to_phonemes)
phoneme_file_counts = get_phoneme_file_counts()

i = 0
for sentence in phon_list_sentences:
    print(sentences[i])
    pn_waveforms = []
    for pn in sentence:
        pn_waveform = get_random_phoneme_waveform(pn, phoneme_file_counts)
        pn_waveforms.append(pn_waveform)
    waveform = torch.cat(pn_waveforms, 0).view(1,-1)
    torchaudio.save(filepath=TRAIN_AUGMENTED_PATH / f'test_{i}.wav', src=waveform, sample_rate=SAMPLE_RATE)
    if i > 10:
        break
    i += 1
