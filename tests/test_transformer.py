import torch
from progen.model import MultiHeadAttention, PositionalEncoding, Block, ProgenModel


def test_multihead_attention():
    x = torch.randn(10, 32, 512)
    attention = MultiHeadAttention(d_model=512, n_heads=8)

    y = attention(x, x, x)

    assert y.shape == (10, 32, 512)

def test_positional_encoding():
    x = torch.randn(10, 32, 512)
    positional_encoding = PositionalEncoding(d_model=512, max_len=32)

    y = positional_encoding(x)

    assert y.shape == (10, 32, 512)

def test_block():
    x = torch.randn(10, 32, 512)
    block = Block(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)

    y = block(x)

    assert y.shape == (10, 32, 512)

def test_progen_model():
    x = torch.randint(0, 100, (10, 32))
    model = ProgenModel(vocab_size=100, n_layers=6, n_heads=8, d_model=512, d_ff=2048, max_seq_len=32)

    y = model(x)

    assert y.shape == (10, 32, 512)


def test_generate_proteins():
    pass