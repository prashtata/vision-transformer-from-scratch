import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MultiHeadedAttention_ParameterLooped(nn.Module):
    def __init__(self, embed_dim = 32, num_heads = 4):
        assert embed_dim % num_heads == 0 
        super().__init__()
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k)) # dim_0 is num_heads so we can iterate the projections over the different heads
        self.W_k = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k))
        self.W_v = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k))  # The parameters can also be written as Linear layers


        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, X):
        bs, seq_len = X.shape[:2]

        heads = []
        for i in range(self.num_heads):
            Q = X @ self.W_q[i]
            K = X @ self.W_k[i]
            V = X @ self.W_v[i]

            attention = nn.Softmax(dim=2)(Q @ K.transpose(1,2) / np.sqrt(self.d_k)) @ V
            heads.append(attention)

        multi_head_attn = torch.cat(heads, dim=-1)
        multi_head_output = self.W_o(multi_head_attn)

        return multi_head_output



class MultiHeadedAttention_ParameterVec(nn.Module):
    def __init__(self, embed_dim = 32, num_heads = 4):
        assert embed_dim % num_heads == 0
        super().__init__()
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k)) # dim_0 is num_heads so we can iterate the projections over the different heads
        self.W_k = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k))
        self.W_v = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k))  # The parameters can also be written as Linear layers

        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, X):
        bs, seq_len = X.shape[:2]

        X = torch.unflatten(X, 0, (-1, 1))

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        attention = nn.Softmax(dim=3)(Q @ K.transpose(2,3) / np.sqrt(self.d_k)) @ V

        multi_head_attn = torch.flatten(attention.transpose(1,2), 2)
        multi_head_output = self.W_o(multi_head_attn)

        return multi_head_output


class MultiHeadedAttention_ParamBias(nn.Module):
    def __init__(self, embed_dim = 32, num_heads = 4):
        assert embed_dim % num_heads == 0
        super().__init__()
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k)) # dim_0 is num_heads so we can iterate the projections over the different heads
        self.W_k = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k))
        self.W_v = nn.Parameter(torch.randn(num_heads, self.d_model, self.d_k))  # The parameters can also be written as Linear layers

        # Biases
        self.b_q = nn.Parameter(torch.zeros(num_heads, 1, self.d_k))  # broadcast over sequence
        self.b_k = nn.Parameter(torch.zeros(num_heads, 1, self.d_k))
        self.b_v = nn.Parameter(torch.zeros(num_heads, 1, self.d_k))


        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, X):
        bs, seq_len = X.shape[:2]

        X = torch.unflatten(X, 0, (-1, 1))

        Q = X @ self.W_q + self.b_q
        K = X @ self.W_k + self.b_k
        V = X @ self.W_v + self.b_v

        attention = nn.Softmax(dim=3)(Q @ K.transpose(2,3) / np.sqrt(self.d_k)) @ V

        multi_head_attn = torch.flatten(attention.transpose(1,2), 2)
        multi_head_output = self.W_o(multi_head_attn)

        return multi_head_output


class MultiHeadedAttention_Linear(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        assert embed_dim % num_heads == 0
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.d_model = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        # Use linear layers instead of per-head parameters
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        # X: [batch_size, seq_len, embed_dim]
        bs, seq_len, _ = X.size()

        Q = self.W_q(X)  # [bs, seq_len, embed_dim]
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.view(bs, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(bs, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(bs, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)  # [bs, num_heads, seq_len, seq_len]
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V  # [bs, num_heads, seq_len, d_k]

        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)

        output = self.W_o(attn_output)

        return output


class TorchMultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x)[0]
