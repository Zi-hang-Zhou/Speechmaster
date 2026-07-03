"""BiLSTM + CTC head over frozen SSL features (continuous or discrete)."""
import torch
import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(self, vocab_size, in_dim=768, hidden=256, layers=2,
                 discrete_vocab=None, dropout=0.1):
        super().__init__()
        self.discrete = discrete_vocab is not None
        if self.discrete:
            self.embed = nn.Embedding(discrete_vocab, in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x, lengths):
        # x: [B,T,D] continuous  or  [B,T] long discrete
        if self.discrete:
            x = self.embed(x)
        x = torch.relu(self.proj(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        return self.out(x)  # [B,T,vocab]
