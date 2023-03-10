# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_model.ipynb.

# %% auto 0
__all__ = ['PositionalEncoding', 'MultiHeadAttention', 'Block', 'ProgenModel']

# %% ../nbs/03_model.ipynb 4
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchtyping import TensorType

# %% ../nbs/03_model.ipynb 5
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(
        self,
        x: TensorType["seq_len", "batch_size", "d_model"]
    ) -> TensorType["seq_len", "batch_size", "d_model"]:
        return x + self.pe[:x.size(0), :]

# %% ../nbs/03_model.ipynb 6
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: TensorType["batch_size", "seq_len", "d_model"],
        k: TensorType["batch_size", "seq_len", "d_model"],
        v: TensorType["batch_size", "seq_len", "d_model"],
        mask: Optional[TensorType["batch_size", "seq_len", "d_model"]]=None
    ) -> TensorType["batch_size", "seq_len", "d_model"]:
        bs = q.size(0)
                
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_head)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_head)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_head)
        
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
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
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float=0.1):
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

    def forward(
        self,
        x: TensorType["batch_size", "seq_len", "d_model"]
    ) -> TensorType["batch_size", "seq_len", "d_model"]:
        x2 = self.self_attn(x, x, x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.ff(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

# %% ../nbs/03_model.ipynb 8
class ProgenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int, n_heads: int, d_model: int, d_ff: int,
        max_seq_len: int, dropout: float=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(
        self,
        x: TensorType["batch_size", "seq_len", "d_model"],
        target: TensorType["batch_size", 1] = None
    ) -> Tuple[
        TensorType["batch_size", "seq_len", "d_model"], # logits
        Optional[TensorType[1]] # loss
    ]:
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.norm(x)
        # logits = self.lm_head(x)
        
        if target is None:
            loss = None
        else:
            logits = rearrange(logits, 'b s d -> (b s) d')
            target = rearrange(target, 'b s -> (b s)')
            
            loss = F.cross_entropy(logits, target)
        
        return logits, loss
