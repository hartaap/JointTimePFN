#!/usr/bin/env python3
# Exact LMC posterior predictive, built densely in numpy from arbitrary kernels
import numpy as np


def kernel_grams(kernels, x):
    # [Q, T, T] stack of gram matrices
    x = np.asarray(x, float).reshape(-1, 1)

    return np.stack([np.asarray(k(x), float) for k in kernels])


def _block(Kq, W, ci, ti, cj, tj):
    # Sigma[(ci,ti),(cj,tj)] = sum_q W[ci,q] W[cj,q] Kq[q,ti,tj]
    Kb = Kq[:, ti[:, None], tj[None, :]]  # [Q, ni, nj]
    out = np.einsum('iq,jq,qij->ij', W[ci], W[cj], Kb, optimize=True)

    return out


def cells_channel_major(mask):
    # (t_idx, c_idx) of True cells in channel-major order
    ts, cs = [], []
    for c in range(mask.shape[1]):
        w = np.where(mask[:, c])[0]
        if w.size:
            ts.append(w); cs.append(np.full(w.size, c))
    if not ts:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(ts), np.concatenate(cs)


def lmc_predict(Y, observed, scored, kernels, W, noise_var, x=None, return_cov=False):
    # basic, exact posterior predictive at scored conditioning on observed
    jitter = 1e-10
    T, N = Y.shape
    x = np.linspace(0.0, 1.0, T) if x is None else np.asarray(x, float)
    Kq = kernel_grams(kernels, x)
    W = np.asarray(W, float)

    ot, oc = cells_channel_major(observed)
    st, sc = cells_channel_major(scored)

    if st.size == 0:
        return (np.zeros(0), np.zeros((0, 0)) if return_cov else np.zeros(0))

    Soo = _block(Kq, W, oc, ot, oc, ot) + (noise_var + jitter) * np.eye(len(ot))
    Sso = _block(Kq, W, sc, st, oc, ot)
    yo = Y[ot, oc]

    L = np.linalg.cholesky(Soo)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yo))
    mu = Sso @ alpha
    V = np.linalg.solve(L, Sso.T)  # [n_obs, n_scored]

    if return_cov:
        Sss = _block(Kq, W, sc, st, sc, st) + noise_var * np.eye(len(st))
        cov = Sss - V.T @ V
        return mu, 0.5 * (cov + cov.T)

    kss = (np.einsum('iq,iq,qi->i', W[sc], W[sc], Kq[:, st, st]) + noise_var)

    return mu, np.maximum(kss - (V ** 2).sum(0), 1e-12)
