import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T, D = query.size()
        def transform(x, linear):
            x = linear(x)
            return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        Q = transform(query, self.q_linear)
        K = transform(key, self.k_linear)
        V = transform(value, self.v_linear)
        out, attn = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k)
        return self.out_proj(out)
