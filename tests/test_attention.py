import torch
from attention.multihead import MultiHeadAttention

def test_output_shape():
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    x = torch.rand(2, 10, 64)
    out = mha(x, x, x)
    assert out.shape == (2, 10, 64)
