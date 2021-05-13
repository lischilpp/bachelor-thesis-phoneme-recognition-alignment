import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import string
from random import randrange
import warnings
# disable C++ extension warning
warnings.filterwarnings('ignore', 'torchaudio C\+\+', )
import torchaudio
import librosa
import soundfile as sf
import pyrubberband as pyrb
import numpy as np, numpy.random
import matplotlib.pyplot as plt

from settings import *
from phonemes import Phoneme

n_fft = 400
specgram_height = n_fft // 2 + 1
specgram_width = 3

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
    TRAIN_PHONEMES_PATH
    phoneme_file_counts = {}
    for speaker_path in TRAIN_PHONEMES_PATH.glob('*'):
        if not speaker_path.is_dir(): continue
        speaker_id = speaker_path.name
        if not speaker_id in phoneme_file_counts:
            phoneme_file_counts[speaker_id] = {}
        for phoneme_path in speaker_path.glob('*'):
            if not phoneme_path.is_dir(): continue
            phoneme_symbol = phoneme_path.name
            phoneme_file_counts[speaker_id][phoneme_symbol] = sum(1 for x in phoneme_path.glob('*') if x.is_file())
    return phoneme_file_counts

def get_random_sentence(sentences, len_sentences):
    return sentences[randrange(len_sentences)]

def get_random_speaker_id(speaker_ids, speaker_count):
    return speaker_ids[randrange(speaker_count)]

def get_random_phoneme_waveform(speaker_id, pn, phoneme_file_counts):
    i = randrange(phoneme_file_counts[speaker_id][pn])
    wav_path = str(TRAIN_PHONEMES_PATH / speaker_id / pn / f'pn{i}.wav')
    waveform, _ = torchaudio.load(wav_path)
    waveform = waveform[0]
    return waveform

def random_lerp(arr):
    weights = np.random.dirichlet(np.ones(AUGMENT_LERP_N_SAMPLES), size=1)[0]
    arr = np.sum([arr[i] * weights[i] for i in range(AUGMENT_LERP_N_SAMPLES)])
    return arr

def waveform_to_spectrogram(waveform, width):
    hop_length = int(waveform.size(0) / (width-1))
    specgram = torchaudio.transforms.Spectrogram(hop_length=hop_length, n_fft=n_fft)(waveform)
    return specgram

def plot_specgram(specgram):
    plt.imshow(specgram)
    plt.show()

def normalize_duration(waveform):
    duration = waveform.size(0) / SAMPLE_RATE
    waveform = librosa.effects.time_stretch(waveform.numpy(), 1/duration, )
    waveform = torch.tensor(waveform)
    return waveform


TRAIN_AUGMENTED_PATH.mkdir(exist_ok=True)

sentences = get_sentences()
word_to_phonemes = create_word_to_phonemes_dict()
phon_list_sentences = sentences_to_phoneme_lists(sentences, word_to_phonemes)
phoneme_file_counts = get_phoneme_file_counts()
speaker_ids = list(phoneme_file_counts.keys())
speaker_count = len(speaker_ids)

n_sentences = len(phon_list_sentences)
for i in range(AUGMENT_TOTAL_RECORDS):
    speaker_id = get_random_speaker_id(speaker_ids, speaker_count)
    cannot_speak_sentence = True
    while cannot_speak_sentence:
        sentence = get_random_sentence(phon_list_sentences, n_sentences)
        cannot_speak_sentence = False
        for symbol in sentence:
            if not symbol in phoneme_file_counts[speaker_id]:
                cannot_speak_sentence=True
    sentence_phonemes = []
    pn_waveforms = []
    sample_idx = 0
    for symbol in sentence:
        pn_waveform = get_random_phoneme_waveform(speaker_id, symbol, phoneme_file_counts)
        pn_samples = pn_waveform.size(0)
        sentence_phonemes.append(Phoneme(sample_idx, sample_idx + pn_samples, symbol))
        pn_waveforms.append(pn_waveform)
        sample_idx += pn_samples
    waveform = torch.cat(pn_waveforms, 0)
    entry = (waveform, sentence_phonemes, speaker_id)
    torch.save(entry, TRAIN_AUGMENTED_PATH / f'record{i}')