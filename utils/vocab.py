import os
import re
import json
import random
from collections import Counter

class Vocabulary:
    def __init__(self, freqthreshold=2):
        self.freq_threshold = freqthreshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def buildvocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized]



# #Build vocab from all captions
# all_captions = [cap for _, cap in data_pairs]
# vocab = Vocabulary(freqthreshold=5)
# vocab.buildvocabulary(all_captions)
