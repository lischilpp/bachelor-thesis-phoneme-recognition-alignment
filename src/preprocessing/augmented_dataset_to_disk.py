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
    phoneme_file_counts = {}
    for path in TRAIN_PHONEMES_PATH.glob('*'):
        if path.is_dir():
            phoneme_file_counts[path.name] = sum(1 for x in path.glob('*') if x.is_file())
    return phoneme_file_counts

def get_random_phoneme_waveform(pn, phoneme_file_counts):
    i = randrange(phoneme_file_counts[pn])
    wav_path = TRAIN_PHONEMES_PATH / pn / f'pn{i}.wav'
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

# sentences = get_sentences()
# word_to_phonemes = create_word_to_phonemes_dict()
# phon_list_sentences = sentences_to_phoneme_lists(sentences, word_to_phonemes)
# phoneme_file_counts = get_phoneme_file_counts()

# # i = 0
# for sentence in phon_list_sentences:
#     merged_specgrams = []
#     for pn in sentence:
#         pn_specgrams = np.zeros((AUGMENT_LERP_N_SAMPLES, specgram_height, specgram_width))
#         for i in range(AUGMENT_LERP_N_SAMPLES):
#             pn_waveform = get_random_phoneme_waveform(pn, phoneme_file_counts)
#             pn_waveform = normalize_duration(pn_waveform)
#             pn_specgrams[i] = waveform_to_spectrogram(pn_waveform, specgram_width)
#         specgram = random_lerp(pn_specgrams)
#         merged_specgrams.append(specgram)
#     full_specgram = np.sum(merged_specgrams)
#     print(full_specgram.shape)
#     exit()
    
    # waveform = torch.cat(pn_waveforms, 0).view(1,-1)
    # torchaudio.save(filepath=TRAIN_AUGMENTED_PATH / f'test_{i}.wav', src=waveform, sample_rate=SAMPLE_RATE)
    # if i > 10:
    #     break
    # i += 1

# waveform, _ = torch.load(TRAIN_RAW_PATH / 'record0')
# n_samples = waveform.size(1)
# torchaudio.save(filepath=TRAIN_AUGMENTED_PATH / f'test1.wav', src=waveform, sample_rate=SAMPLE_RATE)
# waveform = librosa.resample(waveform[0].numpy(), SAMPLE_RATE, n_samples / 10 * 2)
# waveform = torch.from_numpy(waveform)
# torchaudio.save(filepath=TRAIN_AUGMENTED_PATH / f'test2.wav', src=waveform.view(1,-1), sample_rate=SAMPLE_RATE)
y, _ = torch.load(TRAIN_RAW_PATH / 'record0')
y_stretch = pyrb.time_stretch(y[0].numpy(), SAMPLE_RATE, 0.5)
y_stretch = torch.tensor(y_stretch)
torchaudio.save(filepath=str(TRAIN_AUGMENTED_PATH / f'test1.wav'), src=y_stretch.view(1,-1), sample_rate=SAMPLE_RATE)