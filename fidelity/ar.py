#!/usr/bin/env python3
# Teacher-forced version of autoregressive joint density for the unified mask-conditional PFN

import numpy as np
import torch

D_FLOOR = 1e-4  # experiments/exp_unified_joint.D_FLOOR


def _forward(exp, vals, m, marks):

    out = exp.model(vals * (1.0 - m), m, marks)
    if len(out) == 3:
        mean, lv, fac = out
        var = lv.clamp(-10.0, 10.0).exp().clamp_min(D_FLOOR) + (fac ** 2).sum(-1)
        return mean, var.clamp_min(1e-8).log()
    return out


def make_order(scored):
    # (K, 2) (t, c) pairs of the scored cells, time-major
    ts, cs = np.where(scored)
    idx = np.stack([ts, cs], axis=1)

    return idx[np.lexsort((idx[:, 1], idx[:, 0]))]


def _init_state(full, mask0, batch, device):
    # Zeroed values and float mask, repeated over the batch
    vals0 = (full * (~mask0)).astype('float32')
    vals = torch.tensor(np.repeat(vals0[None], batch, axis=0), device=device)
    m = torch.tensor(np.repeat(mask0.astype('float32')[None], batch, axis=0), device=device)

    return vals, m


def _expand_marks(marks, batch):
    # Repeat [1, T, 4] time marks over the batch.
    if marks is None:
        return None
    if marks.shape[0] == batch:
        return marks

    return marks.repeat(batch, 1, 1)


@torch.no_grad()
def ar_logdensity(exp, full, mask0, orders, marks=None):
    # Teacher-forced AR log-density of the true values, one per order
    dev = exp.device
    P, K = len(orders), len(orders[0])
    assert all(len(o) == K for o in orders), 'all orders must cover the same cells'
    vals, m = _init_state(full, mask0, P, dev)
    marks = _expand_marks(marks, P)
    truth = torch.tensor(full.astype('float32'), device=dev)
    logp = torch.zeros(P, device=dev)
    const = float(np.log(2.0 * np.pi))

    for k in range(K):
        mean, lv = _forward(exp, vals, m, marks)
        for p in range(P):
            t, c = int(orders[p][k, 0]), int(orders[p][k, 1])
            mu = mean[p, t, c]
            lvv = lv[p, t, c].clamp(-10.0, 10.0)
            y = truth[t, c]
            logp[p] = logp[p] - 0.5 * (lvv + (y - mu) ** 2 / lvv.exp() + const)
            vals[p, t, c] = y
            m[p, t, c] = 0.0

    return logp.cpu().numpy()


