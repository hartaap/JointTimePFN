# Joint model: the marginal model (TimePFN_PPD_unified_fold) plus an R-dimensional loading w_i
# per cell, so the joint over any set of masked cells is Sigma = diag(d) + W W^T
# (layers/joint_head.py). Sigma is positive definite by construction, and any subset's Sigma is the
# exact marginal of the larger one. The reported model uses R = 32.


import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_FeaturePatchingMasked


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.T = configs.seq_len + configs.pred_len
        self.N = configs.enc_in
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding_FeaturePatchingMasked(
            self.T, configs.patch_size,
            embed_dim=configs.embed_dim, embed_type='fixed', freq='h', dropout=0.1,
        )

        # RoPE on the patch axis, off unless configs.use_rope.
        rope_n_patches = (self.enc_embedding.n_of_patches
                          if getattr(configs, 'use_rope', False) else None)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.embed_dim, configs.n_heads,
                        n_patches=rope_n_patches,
                        rope_base=getattr(configs, 'rope_base', 10.0),
                    ),
                    configs.embed_dim,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.embed_dim),
        )

        flat_dim = ((self.T - configs.patch_size)//(configs.patch_size // 2) + 1) * configs.embed_dim

        self.projector_mean_1   = nn.Linear(flat_dim, configs.d_model, bias=True)
        self.projector_mean_2   = nn.Linear(configs.d_model, self.T, bias=True)
        self.projector_logvar_1 = nn.Linear(flat_dim, configs.d_model, bias=True)
        self.projector_logvar_2 = nn.Linear(configs.d_model, self.T, bias=True)

        # joint head
        self.R = int(getattr(configs, 'joint_rank', 32))
        self.projector_factor_1 = nn.Linear(flat_dim, configs.d_model, bias=True)
        self.projector_factor_2 = nn.Linear(configs.d_model, self.T * self.R, bias=True)
        # small random init: dNLL/dW vanishes identically at W = 0
        nn.init.normal_(self.projector_factor_2.weight, std=1e-2)
        nn.init.zeros_(self.projector_factor_2.bias)

    @staticmethod
    def _masked_stats(values, observed, eps=1e-4):
        n = observed.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu = (values*observed).sum(dim=1, keepdim=True)/n
        var = ((values - mu)**2 * observed).sum(dim=1, keepdim=True)/n
        sigma = torch.sqrt(var + eps)
        return mu, sigma

    def forward(self, values, mask, full_mark=None):

        # values    [B, T, N]   masked cells are already zero in the input
        # mask      [B, T, N]   1 = predict, 0 = observed
        # full_mark [B, T, 4]   optional time marks (or None)

        B, T, N = values.shape
        assert T == self.T, f"T mismatch: model expects {self.T}, got {T}"
        observed = 1.0 - mask

        if self.use_norm:
            mu, sigma = self._masked_stats(values, observed)
            # masked cells stay 0 after normalisation
            values_n = (values - mu) * observed / sigma
        else:
            mu = torch.zeros(B, 1, N, device=values.device)
            sigma = torch.ones(B, 1, N, device=values.device)
            values_n = values

        enc_out = self.enc_embedding(values_n, mask, full_mark)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        n_chan = N + (full_mark.shape[2] if full_mark is not None else 0)
        enc_out = enc_out.reshape(B, n_chan, -1)  # [B, n_chan, flat_dim]

        mean   = self.projector_mean_2(F.gelu(self.projector_mean_1(enc_out)))
        logvar = self.projector_logvar_2(F.gelu(self.projector_logvar_1(enc_out)))
        factor = self.projector_factor_2(F.gelu(self.projector_factor_1(enc_out)))

        # keep the N value channels
        mean   = mean.permute(0, 2, 1)[:, :, :N]
        logvar = logvar.permute(0, 2, 1)[:, :, :N]
        factor = factor.reshape(B, n_chan, self.T, self.R).permute(0, 2, 1, 3)[:, :, :N, :]

        if self.use_norm:
            mean = mean * sigma + mu
            logvar = logvar + 2.0 * torch.log(sigma + 1e-8)
            # denormalise: d -> sigma^2 d (logvar above), w -> sigma * w
            factor = factor * sigma.unsqueeze(-1)

        return mean, logvar, factor
