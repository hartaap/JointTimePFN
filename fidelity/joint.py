#!/usr/bin/env python3
# Joint fidelity on the composed LMC suite: oracle, joint head, diagonal, AR and NUTS

import json
import os
import pickle
import sys

import numpy as np
import torch
from harness import build_args, build_grids, csv_window_batches
from benchmark.models import D_FLOOR
from experiments.exp_unified_masked import Exp_UnifiedMasked
from fidelity.ar import make_order, ar_logdensity
from utils.metrics import gauss_nll, kl_gauss, score_joint_ppd, JITTER_USED, COND_SEEN
from layers.joint_head import lrd_sigma
from fidelity.lmc_dense import lmc_predict
from fidelity.nuts import nuts_composed_predict


def poparg(flag, default, cast=str):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = cast(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return v
    return default


COND = poparg('--condition', 'A')
ORACLE_PARAMS = poparg('--oracle_params', '')
MAXW = poparg('--max_windows', 10, int)
MAXC = poparg('--max_cells', 60, int)  # K
TARGETS = poparg('--targets', 'all')
RATE = poparg('--rate', 0.3, float)

# NUTS settings
DO_NUTS = '--nuts' in sys.argv
if DO_NUTS:
    sys.argv.remove('--nuts')

# Same default settings as the reported runs
NUTS_MAXW = poparg('--nuts_max_windows', 5, int)
NUTS_MAXT = poparg('--nuts_max_targets', 1, int)
NUTS_WARMUP = poparg('--nuts_warmup', 1000, int)
NUTS_SAMPLES = poparg('--nuts_samples', 1000, int)
NUTS_CHAINS = poparg('--nuts_chains', 4, int)
# Monte Carlo samples for H(q*)
NUTS_MC = 4000
OUT = poparg('--out', '')
if '--ppd_ckpt' not in sys.argv:
    sys.argv += ['--ppd_ckpt', 'UNUSED']


def cells_channel_major(scored):
    # (t, c) pairs in channel-major order, as lmc_predict and NUTS use
    ts, cs = [], []
    for c in range(scored.shape[1]):
        w = np.where(scored[:, c])[0]
        if w.size:
            ts.append(w)
            cs.append(np.full(w.size, c))
    return np.concatenate(ts), np.concatenate(cs)


def subsample(scored, max_cells, rng):
    ts, cs = np.where(scored)
    if len(ts) <= max_cells:
        return scored.copy()
    keep = rng.choice(len(ts), size=max_cells, replace=False)
    out = np.zeros_like(scored)
    out[ts[keep], cs[keep]] = True
    return out


def main():
    args = build_args()
    args.use_gpu = torch.cuda.is_available()
    exp = Exp_UnifiedMasked(args)
    exp.model.load_state_dict(torch.load(args.unified_ckpt, map_location='cpu'))
    exp.model.eval()
    dev = exp.device
    has_head = hasattr(exp.model, 'projector_factor_2')

    d = np.load(ORACLE_PARAMS, allow_pickle=True)
    kern_type = str(d['kernel']) if 'kernel' in d else 'matern'
    if kern_type != 'composed':
        raise ValueError(f"the fidelity suite is composed-kernel only, got {kern_type!r}")
    with open(os.path.join(os.path.dirname(ORACLE_PARAMS), 'kernels.pkl'), 'rb') as fh:
        composed_kernels = pickle.load(fh)

    oracle = dict(W=d['W'], noise_var=float(d['noise_var']))
    # Time axis length used by the data generator
    gen_len = int(d['length']) if 'length' in d else None
    T = args.seq_len + args.pred_len
    t_star = args.seq_len
    stride = T
    rng = np.random.default_rng(1000)
    sub_rng = np.random.default_rng(7000)
    # Each window ID is one bootstrap cluster.
    acc = {k: [] for k in ('chain_spread', 'slots_outside_band', 'rhat_max', 'n_pinned', 'n_active', 'rhat_pred', 'rhat_wwT', 'ess_min', 'ess_pred', 'ess_lpd', 'rhat_lpd', 'divergences', 'nll_o', 'nll_h', 'nll_diag', 'wid')}
    nw = 0

    for wi, (full, fmk) in enumerate(csv_window_batches(args.csv_path, T, stride, 1, standardize=args.standardize)):
        if wi >= MAXW:
            break
        Wg = full[0].numpy()
        N = Wg.shape[1]
        marks = exp._maybe_marks(fmk)
        tgts = range(N) if TARGETS.strip().lower() == 'all' else [int(t) for t in TARGETS.split(',')]

        for ti, tgt in enumerate(tgts):
            observed, scored = build_grids(COND, T, N, t_star, tgt, RATE, rng)
            scored = subsample(scored, MAXC, sub_rng)
            if scored.sum() < 4:
                continue
            model_mask = ~observed | scored
            ts, cs = cells_channel_major(scored)
            y = Wg[ts, cs]

            # Exact oracle prediction
            x_win = np.linspace(0.0, 1.0, gen_len or T)[wi * stride: wi * stride + T]
            mu_o, S_o = lmc_predict(Wg, observed, scored, composed_kernels, oracle['W'], oracle['noise_var'], x=x_win, return_cov=True)
            acc['nll_o'].append(gauss_nll(y, mu_o, S_o))
            # Number of scored cells may be smaller than max_cells
            acc.setdefault('K_cells', []).append(int(len(y)))
            acc['wid'].append(int(wi))

            fm = torch.tensor(model_mask[None].astype('float32'), device=dev)
            with torch.no_grad():
                out = exp.model(full.float().to(dev) * (1 - fm), fm, marks)
            mean_g = out[0][0].cpu().numpy()
            mu_m = mean_g[ts, cs]

            # Joint Gaussian head
            if has_head:
                fac = out[2][0].cpu().numpy()
                dvar = np.maximum(np.exp(np.clip(out[1][0].cpu().numpy(), -10, 10)), D_FLOOR)[ts, cs]
                Wl = fac[ts, cs]
                S_h = lrd_sigma(torch.tensor(dvar)[None], torch.tensor(Wl)[None])[0].numpy()
                acc['nll_h'].append(gauss_nll(y, mu_m, S_h))

                # NUTS reference on the selected windows and targets
                if DO_NUTS and wi < NUTS_MAXW and ti < NUTS_MAXT:
                    _, _, info = nuts_composed_predict(
                        Wg, observed, scored, composed_kernels, x_win,
                        warmup=NUTS_WARMUP, samples=NUTS_SAMPLES, num_chains=NUTS_CHAINS,
                        seed=int(wi), want_cov=True, verbose=False)

                    # MU and SIG use channel-major cell order
                    acc['chain_spread'].append(float(info['chain_spread']))
                    acc['slots_outside_band'].append(float(info['n_slots_outside_band']))
                    for stat in ('rhat_max', 'n_pinned', 'n_active', 'rhat_pred', 'rhat_wwT', 'ess_min', 'ess_pred', 'ess_lpd', 'rhat_lpd', 'divergences'):
                        acc[stat].append(float(info.get(stat, float('nan'))))

                    g = score_joint_ppd(info['MU'], info['SIG'], mu_m, S_h,
                                        n_mc=NUTS_MC, seed=int(wi))
                    for k, v in g.items():
                        acc.setdefault(f'ppd_{k}', []).append(float(v))
                    try:
                        acc.setdefault('ppd_kl_orc_vs_h', []).append(
                            float(kl_gauss(mu_o, S_o, mu_m, S_h)) / max(len(y), 1))
                    except Exception:
                        acc.setdefault('ppd_kl_orc_vs_h', []).append(float('nan'))

                    # moment-matched Gaussian approximation of the NUTS mixture
                    try:
                        mu_N = np.asarray(info['MU'], float).mean(axis=0)
                        S_N = (np.asarray(info['SIG'], float).mean(axis=0) + np.cov(np.asarray(info['MU'], float).T, bias=True))
                        acc.setdefault('ppd_kl_orc_vs_nuts', []).append(float(kl_gauss(mu_o, S_o, mu_N, S_N)) / max(len(y), 1))
                    except Exception:
                        acc.setdefault('ppd_kl_orc_vs_nuts', []).append(float('nan'))

                # Diagonal baseline: remove off-diagonal covariance.
                S_d = np.diag(np.diag(S_h))
                acc['nll_diag'].append(gauss_nll(y, mu_m, S_d))

            # Autoregressive score in raster order.
            order = make_order(scored)
            acc.setdefault('nll_ar_tf', []).append(
                -float(ar_logdensity(exp, Wg, model_mask, [order], marks=marks)[0]))
        nw += 1

    def mean_or_none(v):
        vals = [x for x in v if x is not None]
        return float(np.mean(vals)) if vals else None

    m = {k: mean_or_none(v) for k, v in acc.items()}
    K_mean = mean_or_none(acc.get('K_cells', [])) or float('nan')
    fmt = lambda k, sc=1.0: f"{m[k] / sc:.4f}" if m.get(k) is not None else '--'
    line = (f"{COND}: {len(set(acc['wid']))} windows, {len(acc['wid'])} rows  |  NLL/cell "
            f"oracle {fmt('nll_o', K_mean)}  head {fmt('nll_h', K_mean)}  AR {fmt('nll_ar_tf', K_mean)}  "
            f"diag {fmt('nll_diag', K_mean)}")
    if m.get('ppd_kl_amort') is not None:
        line += (f"  |  KL(orc|N) {fmt('ppd_kl_orc_vs_nuts')}  KL(N|Ours) {fmt('ppd_kl_amort')}"
                 f"  diag null {fmt('ppd_kl_G_vs_diag')}  spread {fmt('chain_spread')}"
                 f"  rhat_pred {fmt('rhat_pred')}")
    print(line)

    if COND_SEEN:
        worst = min(COND_SEEN)
        mxj = max(JITTER_USED) if JITTER_USED else 0.0
        if worst < 1e-6 or mxj > 1e-4:
            print(f'  warning: Sigma_oracle nearly singular (min_eig/mean_diag {worst:.1e}, jitter {mxj:.0e}): the joint NLL columns are unreliable on this dataset.')
    if OUT:
        # save averages and per row results
        json.dump({'mean': {k: v for k, v in m.items() if k != 'wid'},
                   'per_row': acc,
                   'config': dict(condition=COND, csv=args.csv_path, ckpt=args.unified_ckpt,
                                  stride=stride, T=T, max_cells=MAXC,
                                  targets=TARGETS, rate=RATE, has_head=bool(has_head),
                                  nuts=dict(on=bool(DO_NUTS), warmup=NUTS_WARMUP,
                                            samples=NUTS_SAMPLES, chains=NUTS_CHAINS))},
                  open(OUT, 'w'), indent=2)
        print(f"\nsaved {OUT}")


if __name__ == '__main__':
    main()
