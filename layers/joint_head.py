#!/usr/bin/env python3
# Low-rank joint Gaussian head: Sigma = diag(d) + W W^T

import math
import torch


def lrd_logdet_quad(r, d, W, jitter=1e-6):
    # calculate log determinant and quadratic term using Woodbury and add jitter to d
    d = d + jitter
    rank = W.shape[-1]
    inv_d = 1.0 / d

    wt_dinv = W.transpose(-1, -2)*inv_d.unsqueeze(-2)
    matrix = torch.eye(rank, device=W.device, dtype=W.dtype) + wt_dinv @ W
    chol, info = torch.linalg.cholesky_ex(matrix)

    if bool(info.any()):
        # retry in float64 if float32 rounding causes cholesky to fail
        if W.dtype != torch.float64:
            logdet, quad = lrd_logdet_quad(r.double(), d.double(), W.double(), jitter=0.0)
            return logdet.to(W.dtype), quad.to(W.dtype)
        chol = torch.linalg.cholesky(matrix)

    a = wt_dinv @ r.unsqueeze(-1)
    solution = torch.cholesky_solve(a, chol)
    quad = (r*r*inv_d).sum(-1) - (a * solution).sum((-2, -1))
    logdet = torch.log(d).sum(-1) + 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return logdet, quad


def lrd_joint_nll(y, mu, d, W, k_eff=None, jitter=1e-6):
    # Joint Gaussian negative log-likelihood
    # k_eff excludes padded slots from the constant term
    r = y - mu
    logdet, quad = lrd_logdet_quad(r, d, W, jitter=jitter)
    if k_eff is None:
        k_eff = torch.full_like(logdet, r.shape[-1])
    return (0.5)*(quad + logdet + k_eff * math.log(2.0 * math.pi))


def lrd_sigma(d, W):
    # Full covariance matrix, for evaluation
    return torch.diag_embed(d) + W @ W.transpose(-1, -2)


def lrd_marginal_var(d, W):
    # Diagonal of Sigma: d_i + sum_j W_ij^2
    return d + (W * W).sum(-1)
