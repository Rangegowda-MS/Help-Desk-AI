"""Clean dialogue, split before building vocabulary, and pad minibatches."""
import re
from collections import Counter
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

def clean(text, max_words=15):
    return ' '.join(re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).split()[:max_words])

class Vocabulary:
    def __init__(self, words=None):
        self.words = words or SPECIAL.copy()
        self.ids = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def build(cls, texts, min_frequency=1):
        counts = Counter(w for text in texts for w in text.split())
        return cls(SPECIAL + sorted(w for w, n in counts.items() if n >= min_frequency))

    def encode(self, text):
        return [self.ids.get(w, UNK) for w in text.split()] + [EOS]

    def __len__(self):
        return len(self.words)

def load_pairs(path, max_words=15):
    frame = pd.read_csv(path).dropna(subset=['question', 'answer'])
    for column in ['question', 'answer']:
        frame[column] = frame[column].map(lambda x: clean(x, max_words))
    frame = frame[(frame.question != '') & (frame.answer != '')]
    # Avoid identical normalized prompts crossing splits.
    frame = frame.drop_duplicates('question').reset_index(drop=True)
    if len(frame) < 20:
        raise ValueError('Provide at least 20 distinct, nonempty question/answer pairs.')
    return frame

def split_pairs(frame, seed=42):
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    a, b = int(n * .8), int(n * .9)
    return shuffled.iloc[:a].copy(), shuffled.iloc[a:b].copy(), shuffled.iloc[b:].copy()

class Pairs(Dataset):
    def __init__(self, frame, vocab):
        self.items = [(torch.tensor(vocab.encode(q)), torch.tensor(vocab.encode(a)))
                      for q, a in zip(frame.question, frame.answer)]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

def collate(batch):
    source, target = zip(*batch)
    lengths = torch.tensor([len(s) for s in source])
    return (pad_sequence(source, batch_first=True, padding_value=PAD), lengths,
            pad_sequence(target, batch_first=True, padding_value=PAD))
