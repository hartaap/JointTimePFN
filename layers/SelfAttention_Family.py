import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask

#code implementation from iTransformer


# Code implementation from SODA group


# Code implementation from https://github.com/thuml/Flowformer


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# Code implementation from https://github.com/zhouhaoyi/Informer2020


class RotaryEmbedding(nn.Module):
    # Rotary positional embedding (RoPE) with a cached cos/sin table.
    # position = token % n_patches, i.e. the patch (time) index

    def __init__(self, dim, n_patches, base=10000.0):
        # default base 10000; the reported runs use 10 (--rope_base)
        super().__init__()
        inv_freq = 1.0/ (base ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(n_patches).float(), inv_freq)  
        emb = torch.cat([freqs, freqs], dim=-1)                        
        self.register_buffer("cos_table", emb.cos(), persistent=False)
        self.register_buffer("sin_table", emb.sin(), persistent=False)
        self.n_patches = n_patches

    def forward(self, length, device):
        idx = torch.arange(length, device=device) % self.n_patches       
        return self.cos_table[idx], self.sin_table[idx]                 


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, L, H, E]; cos/sin: [L, E] -> rotate the head dim by position
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, n_patches=None, rope_base=10000.0):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        # RoPE on the time (patch) axis. Enabled iff n_patches is given.
        self.n_patches = n_patches
        if n_patches is not None:
            assert d_keys % 2 == 0, "RoPE needs an even head dim"
            self.rope = RotaryEmbedding(d_keys, n_patches, base=rope_base)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if self.n_patches is not None:                       # RoPE on the patch (time) axis
            qcos, qsin = self.rope(L, queries.device)        
            kcos, ksin = self.rope(S, keys.device)
            queries = apply_rope(queries, qcos, qsin)
            keys = apply_rope(keys, kcos, ksin)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

