#!/usr/bin/env python3
# Pyro NUTS for composed-kernel LMC hyperparameters.

import math
import os
import time
import numpy as np
import torch
from fidelity.kernel_tree import spec_from_sklearn, composed_gram, active_mask, _UNIT
from synthetic_data_generation.LMC_Synth_uusi import PRIOR_BANDS, CANONICAL_PERIODS, LENGTH
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.ops import stats as _st


def cells_channel_major(mask):
    # Return time and channel indices in channel-major order
    T, N = mask.shape
    tt, cc = [], []

    for c in range(N):
        t = np.nonzero(mask[:, c])[0]
        tt.append(t)
        cc.append(np.full(t.size, c, dtype=int))

    return ( np.concatenate(tt) if tt else np.zeros(0, int), np.concatenate(cc) if cc else np.zeros(0, int))


# Use the pretraining prior band, regarldess of the data, i.e in vs off prior
_SLOT_BAND = {'ls': 'ls', 'period': 'es_period', 'noise': 'noise', 'steep': 'cp_steep', 'base_var': 'cp_basevar'}
_FIXED_SLOT_BAND = {'sigma_0': (0.1, 1.0), 'alpha': (0.1, 10.0)}
EDGE_TOL = 1e-3


def slot_bounds(spec, band):
    # Bounds and transforms for each parameter slot
    b = PRIOR_BANDS[band] if isinstance(band, str) else band
    out = []

    def walk(s):
        for name, tr, v in s.slots:
            if s.kind == 'expsine' and name == 'ls':
                lo, hi = b['es_ls']

            elif s.kind == 'expsine' and name == 'period':
                lo, hi = b['es_period']
                if b['use_canonical']:
                    lo = min(lo, min(CANONICAL_PERIODS) / LENGTH)
                    hi = max(hi, max(CANONICAL_PERIODS) / LENGTH)

            elif name == 'cp':
                lo, hi = 0.3, 0.7

            elif name in _FIXED_SLOT_BAND:
                lo, hi = _FIXED_SLOT_BAND[name]

            else:
                lo, hi = b[_SLOT_BAND[name]]

            # Allow a little slack for values just outside a band edge
            if v < lo and v >= lo * (1.0 - EDGE_TOL):
                lo = v * 0.999
            if v > hi and v <= hi * (1.0 + EDGE_TOL):
                hi = v * 1.001

            out.append((lo, hi, tr, not (lo <= v <= hi)))

        for c in s.children:
            walk(c)

    walk(spec)
    assert len(out) == spec.n_params()
    return out


def _lmc_blocks(specs, thetas, x, ta, ca, tb, cb, W):
    # Sum the kernel covariance blocks weighted by the LMC loadings.
    K = None
    for q, (sp, th) in enumerate(zip(specs, thetas)):
        Kq = composed_gram(sp, th, x)
        blk = Kq[ta][:, tb] * (W[ca, q].unsqueeze(-1) * W[cb, q].unsqueeze(-2))
        K = blk if K is None else K + blk

    return K


def _predict_given_theta(specs, thetas, x, W, sig, ot, oc, yo, st, sc, return_cov=False):
    # Predict scored cells given one sampled set of parameters.
    jitter = 1e-10

    Koo = _lmc_blocks(specs, thetas, x, ot, oc, ot, oc, W)
    Koo = Koo + (sig ** 2 + jitter) * torch.eye(len(ot), dtype=Koo.dtype, device=Koo.device)
    Kso = _lmc_blocks(specs, thetas, x, st, sc, ot, oc, W)

    L = torch.linalg.cholesky(Koo)
    mu = (Kso @ torch.cholesky_solve(yo.unsqueeze(-1), L)).squeeze(-1)
    V = torch.linalg.solve_triangular(L, Kso.transpose(-1, -2), upper=False)
    Kss = _lmc_blocks(specs, thetas, x, st, sc, st, sc, W)

    if return_cov:
        cov = Kss + sig ** 2 * torch.eye(len(st), dtype=Kss.dtype) - V.transpose(-1, -2) @ V
        return mu, 0.5 * (cov + cov.transpose(-1, -2))

    var = torch.diagonal(Kss) + sig ** 2 - (V ** 2).sum(0)
    return mu, var.clamp_min(1e-8)


# Match the generator's Dirichlet range when setting the loading scale.
_TRAIN_DIRICHLET = (0.1, 10.0)


def _train_w_scale(Q):
    # Gaussian loading scale matched to the generator's expected row norm.
    n = 20001
    a = np.linspace(_TRAIN_DIRICHLET[0], _TRAIN_DIRICHLET[1], n)
    y = (a + 1.0) / (Q * a + 1.0)
    integ = float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(a)))
    return float(np.sqrt(integ / (_TRAIN_DIRICHLET[1] - _TRAIN_DIRICHLET[0]) / Q))


def nuts_composed_predict(Y, observed, scored, kernels, x, warmup=1000, samples=1000, num_chains=4, seed=0, max_tree_depth=7, want_cov=False, verbose=True):
    # Return predictive mean, variance and diagnostics at the scored cells.
    noise_band = (0.01, 0.5)
    device = os.environ.get('PFN_NUTS_DEVICE', '').strip()

    if device:
        torch.set_default_device(device)
    try:
        return _nuts_composed_impl(Y, observed, scored, kernels, x, warmup, samples, num_chains, seed, max_tree_depth, noise_band, want_cov, verbose)
    finally:
        if device:
            torch.set_default_device('cpu')


def _nuts_composed_impl(Y, observed, scored, kernels, x, warmup, samples, num_chains, seed,
                        max_tree_depth, noise_band, want_cov, verbose):
    # Always use the matched imd prior band
    band = 'imd'
    T, N = Y.shape
    specs = [spec_from_sklearn(k) for k in kernels]
    Qtot = len(specs)
    xt = torch.tensor(np.asarray(x, float), dtype=torch.float64)
    masks = [active_mask(sp, xt) for sp in specs]
    bounds = [slot_bounds(sp, band) for sp in specs]
    truths = [sp.true_vector() for sp in specs]
    n_act = sum(sum(m) for m in masks)
    n_tot = sum(sp.n_params() for sp in specs)

    ot, oc = cells_channel_major(observed)
    st, sc = cells_channel_major(scored)
    to = torch.tensor(ot, dtype=torch.long)
    co = torch.tensor(oc, dtype=torch.long)
    ts = torch.tensor(st, dtype=torch.long)
    cs = torch.tensor(sc, dtype=torch.long)
    yo = torch.tensor(Y[ot, oc], dtype=torch.float64)

    # Free parameters are sampled & inactive ones stay at their true values
    log_lo, log_hi, cp_lo, cp_hi, act_idx, log_pos, cp_pos = [], [], [], [], [], [], []
    off = 0
    for sp, m, bd in zip(specs, masks, bounds):
        for j, (a, (lo, hi, tr, outside)) in enumerate(zip(m, bd)):
            if a:
                if tr == _UNIT:
                    cp_pos.append(len(act_idx))
                    cp_lo.append(lo)
                    cp_hi.append(hi)
                else:
                    log_pos.append(len(act_idx))
                    log_lo.append(math.log(lo))
                    log_hi.append(math.log(hi))
                act_idx.append(off + j)

        off += sp.n_params()

    log_lo_t = torch.tensor(log_lo, dtype=torch.float64)
    log_hi_t = torch.tensor(log_hi, dtype=torch.float64)
    cp_lo_t = torch.tensor(cp_lo, dtype=torch.float64)
    cp_hi_t = torch.tensor(cp_hi, dtype=torch.float64)
    log_pos_t = torch.tensor(log_pos, dtype=torch.long)
    cp_pos_t = torch.tensor(cp_pos, dtype=torch.long)
    truth_flat = torch.tensor([v for tv in truths for v in tv], dtype=torch.float64)
    idx_t = torch.tensor(act_idx, dtype=torch.long)

    def assemble(free_log, cp):
        # Log-transformed slots and changepoints in sampler space
        v = torch.zeros(len(act_idx), dtype=torch.float64)
        if free_log is not None:
            v = v.index_copy(0, log_pos_t, free_log)
        if cp is not None:
            v = v.index_copy(0, cp_pos_t, torch.logit(cp))
        return v

    def draw(s, j):
        return assemble(s['free'][j] if 'free' in s else None, s['cp'][j] if 'cp' in s else None)

    def unpack(free):
        full = truth_flat.clone()
        if idx_t.numel():
            full = full.index_copy(0, idx_t, free)
        out, o = [], 0
        for sp in specs:
            out.append(full[o:o + sp.n_params()])
            o += sp.n_params()
        return out

    def model():
        free_log = (pyro.sample('free', dist.Uniform(log_lo_t, log_hi_t).to_event(1)) if log_pos else None)
        cp = pyro.sample('cp', dist.Uniform(cp_lo_t, cp_hi_t).to_event(1)) if cp_pos else None
        free = assemble(free_log, cp)

        w_scale = _train_w_scale(Qtot)
        Wm = pyro.sample('W', dist.Normal(torch.zeros(N, Qtot, dtype=torch.float64), w_scale).to_event(2))

        # Log-uniform observation noise standard deviation.
        nlo, nhi = float(noise_band[0]), float(noise_band[1])
        sigma_u = pyro.sample('sigma_u', dist.Uniform(torch.tensor(0.0, dtype=torch.float64),torch.tensor(1.0, dtype=torch.float64)))
        sig = nlo * (nhi / nlo) ** sigma_u
        K = _lmc_blocks(specs, unpack(free), xt, to, co, to, co, Wm)
        K = K + (sig ** 2 + 1e-8) * torch.eye(len(ot), dtype=K.dtype)

        pyro.sample('y', dist.MultivariateNormal(torch.zeros(len(ot), dtype=torch.float64), covariance_matrix=K), obs=yo)

    chains, n_div, t0 = [], [], time.time()

    for ch in range(num_chains):
        pyro.set_rng_seed(seed * 100 + ch)
        pyro.clear_param_store()
        mcmc = MCMC(NUTS(model, max_tree_depth=max_tree_depth, jit_compile=False), num_samples=samples, warmup_steps=warmup, disable_progbar=not verbose)
        mcmc.run()
        samp = {k: v.detach() for k, v in mcmc.get_samples().items()}

        nlo, nhi = float(noise_band[0]), float(noise_band[1])
        samp['sigma'] = nlo * (nhi / nlo) ** samp['sigma_u']
        chains.append(samp)
        try:
            n_div.append(sum(len(v) for v in (mcmc.diagnostics().get('divergences') or {}).values()))
        except Exception:
            pass
        if verbose:
            print(f'  [nuts] chain {ch} done {time.time() - t0:.0f}s')

    # Collect the predictive moments for all posterior draws.
    MU, VAR, SIG = [], [], []
    for s in chains:
        for j in range(samples):
            free = draw(s, j)
            if want_cov:
                m, C = _predict_given_theta(specs, unpack(free), xt, s['W'][j], s['sigma'][j], to, co, yo, ts, cs, return_cov=True)
                SIG.append(C.cpu().numpy())
                v = torch.diagonal(C).clamp_min(1e-8)
            else:
                m, v = _predict_given_theta(specs, unpack(free), xt, s['W'][j], s['sigma'][j], to, co, yo, ts, cs)
            MU.append(m.cpu().numpy())
            VAR.append(v.cpu().numpy())

    MU, VAR = np.asarray(MU), np.asarray(VAR)
    mu = MU.mean(0)
    # total variance of the finite mixture (ddof=0)
    var = VAR.mean(0) + MU.var(0, ddof=0)

    # Between-chain predictive spread, used for logging
    per = [
        np.stack([_predict_given_theta(specs, unpack(draw(s, j)), xt, s['W'][j], s['sigma'][j], to, co, yo, ts, cs)[0].cpu().numpy() for j in range(samples)]).mean(0) for s in chains
    ]
    spread = float(np.abs(np.stack(per) - np.mean(per, 0)).max()) if num_chains > 1 else float('nan')

    # number of active parameters whose true values lie outside the prior band
    n_out = sum(1 for mk, bd in zip(masks, bounds) for a_, sl in zip(mk, bd) if a_ and sl[3])

    # rhat splitted across chains, ignore coordinates that never change
    rhat_max = float('nan')
    rhat_pred, rhat_wwT = float('nan'), float('nan')
    ess_min, ess_pred, ess_lpd, rhat_lpd = float('nan'), float('nan'), float('nan'), float('nan')

    if num_chains > 1:
        try:
            rhats, esss = [], []
            for site in ('free', 'cp', 'W', 'sigma_u'):
                if site not in chains[0]:
                    continue
                stacked = torch.stack([c[site] for c in chains])
                flat = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)
                keep = flat.reshape(-1, flat.shape[-1]).std(0) > 1e-12
                if not bool(keep.any()):
                    continue
                sub = flat[:, :, keep]
                rhats.append(float(_st.split_gelman_rubin(sub).max()))
                esss.append(float(_st.effective_sample_size(sub).min()))

            if rhats:
                rhat_max, ess_min = max(rhats), min(esss)

            # sign invariant diagnostics: predictive mean and upper triangle of W W^T.
            def live_coords(a):
                a = a.reshape(a.shape[0], a.shape[1], -1)
                return a[:, :, a.reshape(-1, a.shape[-1]).std(0) > 1e-12]

            def rhat_of(a):
                a = live_coords(a)
                return float(_st.split_gelman_rubin(a).max()) if a.shape[-1] else float('nan')

            def ess_of(a):
                a = live_coords(a)
                return float(_st.effective_sample_size(a).min()) if a.shape[-1] else float('nan')

            MU_chains = torch.as_tensor(MU).reshape(num_chains, samples, -1)
            rhat_pred = rhat_of(MU_chains)
            W_chains = torch.stack([c['W'] for c in chains])
            iu = torch.triu_indices(W_chains.shape[2], W_chains.shape[2])
            rhat_wwT = rhat_of((W_chains @ W_chains.transpose(-1, -2))[:, :, iu[0], iu[1]])
            ess_pred = ess_of(MU_chains)
            with torch.device('cpu'):
                ys = torch.as_tensor(Y[st, sc], dtype=torch.float64)
                lp = []
                for ci, s in enumerate(chains):
                    for j in range(samples):
                        if want_cov:
                            m, C = MU[ci * samples + j], SIG[ci * samples + j]
                        else:
                            m, C = _predict_given_theta(specs, unpack(draw(s, j)), xt, s['W'][j], s['sigma'][j], to, co, yo, ts, cs, return_cov=True)
                        m = torch.as_tensor(m, dtype=torch.float64).cpu()
                        C = torch.as_tensor(C, dtype=torch.float64).cpu()
                        C = 0.5 * (C + C.T)
                        L, bad = torch.linalg.cholesky_ex(C)
                        if bad:
                            L = torch.linalg.cholesky(C + 1e-8 * C.diagonal().mean() * torch.eye(len(C), dtype=C.dtype))
                        lp.append(torch.distributions.MultivariateNormal(m, scale_tril=L).log_prob(ys))
                LP = torch.stack(lp).reshape(num_chains, samples, 1)
                ess_lpd, rhat_lpd = ess_of(LP), rhat_of(LP)
        except Exception:
            pass

    info = dict(n_active=n_act, n_pinned=n_tot - n_act, chain_spread=spread, n_slots_outside_band=n_out,
        rhat_max=rhat_max, rhat_pred=rhat_pred, rhat_wwT=rhat_wwT,
        ess_min=ess_min, ess_pred=ess_pred, ess_lpd=ess_lpd, rhat_lpd=rhat_lpd, divergences=int(sum(n_div)),
    )

    if verbose:
        print(f'  [nuts] {n_act}/{n_tot} slots sampled ({n_tot - n_act} pinned as blind), n_obs={len(ot)}, across-chain predictive spread {spread:.4f}')

    if want_cov:
        info['MU'], info['VAR'] = MU, VAR
        info['SIG'] = np.asarray(SIG)

    return mu, var, info
