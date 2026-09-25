"""Explicit encoder, Bahdanau attention, and autoregressive LSTM decoder."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .data import PAD, SOS

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, source, lengths):
        embedded = self.dropout(self.embedding(source))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, state = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=source.size(1))
        return outputs, state

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, mask):
        # e_ti = v^T tanh(W_h h_i + W_s s_t)
        scores = self.score(torch.tanh(self.query(query).unsqueeze(1) + self.key(keys))).squeeze(-1)
        weights = scores.masked_fill(~mask, float('-inf')).softmax(dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, token, state, encoder_outputs, mask):
        context, weights = self.attention(state[0][-1], encoder_outputs, mask)
        inputs = torch.cat([self.dropout(self.embedding(token)), context], dim=-1).unsqueeze(1)
        result, state = self.lstm(inputs, state)
        logits = self.output(torch.cat([result.squeeze(1), context], dim=-1))
        return logits, state, weights

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, dropout=.15):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, dropout)

    def forward(self, source, lengths, target, teacher_forcing=1.0):
        outputs, state = self.encoder(source, lengths)
        mask = source.ne(PAD)
        token = torch.full((source.size(0),), SOS, device=source.device, dtype=torch.long)
        logits = []
        for t in range(target.size(1)):
            step, state, _ = self.decoder(token, state, outputs, mask)
            logits.append(step)
            use_truth = torch.rand(source.size(0), device=source.device) < teacher_forcing
            token = torch.where(use_truth, target[:, t], step.argmax(-1))
        return torch.stack(logits, dim=1)
