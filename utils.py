import string
import torch


sentence_characters = list(string.ascii_lowercase) + \
                      [',', ';', '.', '!', '?', ':', '-', '\'', '\"', ' ']

def one_hot_encode(e, l):
    return [1 if e == x else 0 for x in l]

def encode_sentence(s):
    s = [one_hot_encode(c, sentence_characters) for c in s.lower()]
    s = torch.FloatTensor(s).flatten()
    return s
