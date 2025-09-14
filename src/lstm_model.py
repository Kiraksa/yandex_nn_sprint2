import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class MyLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        linear_out = self.fc(out)
        return linear_out
    