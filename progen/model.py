# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_model.ipynb.

# %% auto 0
__all__ = ['PositionalEncoding', 'MultiHeadAttention', 'Block', 'TransformerModel']

# %% ../nbs/03_model.ipynb 4
import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

# %% ../nbs/03_model.ipynb 5
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# %% ../nbs/03_model.ipynb 6
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(output)
        return output

# %% ../nbs/03_model.ipynb 7
class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.self_attn(x, x, x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.ff(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x


# %% ../nbs/03_model.ipynb 8
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, d_model, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)