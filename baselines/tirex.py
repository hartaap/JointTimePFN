#!/usr/bin/env python3
# TiRex-2 on the cells the other models were scored on
import glob
import os
import sys
import time
from collections import Counter

import numpy as np

# torch.compile needs a C++ toolchain, so set this before torch is imported
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import torch
from tirex2 import TimeseriesType, load_model

cfg = dict(in_dir='tabpfn_io', device='cuda', conditions='A,F,C,DRH',model='NX-AI/TiRex-2', tag='tirex2', multivariate=1, batch_size=64)
for _k in list(cfg):
    _f = '--' + _k
    if _f in sys.argv:
        _i = sys.argv.index(_f)
        cfg[_k] = type(cfg[_k])(sys.argv[_i + 1]);del sys.argv[_i:_i + 2]

CONDS = [c.strip() for c in cfg['conditions'].split(',') if c.strip()]

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_NAN_OK = [None]  # does this build accept NaN in the context?
_NAN_COV_OK = [None]  # and in a future covariate?
_N_PAST = [0]  # groups that passed past-only covariates
_ragged = [0]


def _obj_array(lst):
    a = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst):
        a[i] = v
    return a


def probe_nan(model, TimeseriesType, torch):
    # test once whether NaN is accepted in the context and in a future covariate
    x = torch.randn(1, 64)
    x[0, 10] = float('nan')
    try:
        o = model.forecast([TimeseriesType(target=x, past_covariates=None,future_covariates=None)],prediction_length=8, output_type='numpy')[0]
        _NAN_OK[0] = bool(np.isfinite(np.asarray(o)).all())
        print(f'[tirex] NaN in context: accepted, output finite={_NAN_OK[0]}', flush=True)
    except Exception as e:
        _NAN_OK[0] = False
        print(f'[tirex] NaN in context REJECTED ({type(e).__name__}: {str(e)[:80]}); 'f'groups with gappy context will be recorded, not guessed at', flush=True)

    xt = torch.randn(1, 64)
    cv = torch.randn(1, 72)
    cv[0, 70] = float('nan') # in the future span (64 context + 8)
    try:
        o = model.forecast([TimeseriesType(target=xt, past_covariates=None, future_covariates=cv)],prediction_length=8, output_type='numpy')[0]
        _NAN_COV_OK[0] = bool(np.isfinite(np.asarray(o)).all())
        print(f'[tirex] NaN in future covariate: accepted, output finite={_NAN_COV_OK[0]}', flush=True)
    except Exception as e:
        _NAN_COV_OK[0] = False
        print(f'[tirex] NaN in future covariate REJECTED ({type(e).__name__}: {str(e)[:80]}); ' f'only fully observed channels will be passed as covariates', flush=True)
    return _NAN_OK[0]


def build_ts(Wg, observed, scored, chans, TimeseriesType, torch, forced_t0=None):
    T, N = Wg.shape
    t_fut_sets = [np.where(scored[:, c])[0] for c in chans]
    
    if any(t.size == 0 for t in t_fut_sets):
        return None, 'no_scored_cells'
    
    # dense horizon from a shared origin, then select
    t0 = int(forced_t0) if forced_t0 is not None else int(min(int(t[0]) for t in t_fut_sets))
    t_last = int(max(int(t[-1]) for t in t_fut_sets))

    if forced_t0 is not None and any(int(t[0]) < t0 for t in t_fut_sets):
        return None, 'scored_before_t_star'
    
    H = t_last - t0 + 1
    # queried channels must be unobserved across the span
    if np.asarray(observed[t0:t_last + 1, chans], dtype=bool).any():
        return None, 'not_horizon_shaped'

    ctx = np.asarray(Wg[:t0, chans], dtype='float64').T.copy()  # [V_t, t0]
    gap = ~np.asarray(observed[:t0, chans], dtype=bool).T
    if gap.any():
        if not _NAN_OK[0]:
            return None, 'gappy_context_unsupported'
        ctx[gap] = np.nan

    # known-future covariate
    query_set = set(chans)
    if _NAN_COV_OK[0]:
        known = [c for c in range(N)
                 if c not in query_set and observed[t0:t0 + H, c].any() and observed[:t0, c].any()]
    else:
        known = [c for c in range(N)
                 if c not in query_set and observed[:t0 + H, c].all()]
    fut = None

    if known:
        cov = np.asarray(Wg[:t0 + H, known], dtype='float64').copy()
        cov[~observed[:t0 + H, known]] = np.nan
        fut = torch.as_tensor(cov.T.copy(), dtype=torch.float32)
    # other channels with an observed past are past covariates; NaN cells are masked by the model
    known_set = set(known)

    if _NAN_OK[0]:
        past_ch = [c for c in range(N) if c not in query_set and c not in known_set and observed[:t0, c].any()]
    else:
        past_ch = [c for c in range(N) if c not in query_set and c not in known_set and observed[:t0, c].all()]
    past = None

    if past_ch:
        pc = np.asarray(Wg[:t0, past_ch], dtype='float64').copy()
        pc[~observed[:t0, past_ch]] = np.nan
        past = torch.as_tensor(pc.T.copy(), dtype=torch.float32) # [V_p, t0]
        _N_PAST[0] += 1

    return (TimeseriesType(target=torch.as_tensor(ctx, dtype=torch.float32), past_covariates=past, future_covariates=fut), H, t0), bool(known)


def main():
    model = load_model(cfg['model'], device=cfg['device'])
    print(f"[tirex] model={cfg['model']}  device={cfg['device']}  conditions={CONDS}  " f"multivariate={cfg['multivariate']}  levels={QUANTILES}  tag={cfg['tag']}", flush=True)
    probe_nan(model, TimeseriesType, torch)

    files = sorted(glob.glob(f"{cfg['in_dir']}/prep_*.npz"))

    if not files:
        raise SystemExit(f"no prep_*.npz in {cfg['in_dir']}, run benchmark/prep.py first")

    for f in files:
        d = np.load(f, allow_pickle=True)
        n = len(d['y'])
        D_Wg, D_obs, D_sc, D_cond = d['Wg'], d['observed'], d['scored'], d['cond']
        # shared forecast origin, written by benchmark.prep
        t_star = int(np.asarray(d['t_star']).reshape(-1)[0]) if 't_star' in d.files else None
        mus, vrs, qs, status = [], [], [], []
        t_start = time.time()

        for i in range(n):
            cond = str(D_cond[i])
            # object-dtype elements; cast for torch
            Wg = np.asarray(D_Wg[i], dtype=float)
            observed = np.asarray(D_obs[i], dtype=bool)
            scored = np.asarray(D_sc[i], dtype=bool)
            K = int(scored.sum())

            def skip(tag):
                nan = np.full(K, np.nan, 'float32')
                mus.append(nan); vrs.append(nan.copy())
                qs.append(np.full((len(QUANTILES), K), np.nan, 'float32'))
                status.append(tag)

            if cond == 'DRH':
                skip('unsupported_interior_imputation'); continue

            if cond not in CONDS:
                skip('cond_skipped'); continue
            chans = [c for c in range(Wg.shape[1]) if scored[:, c].any()]
            if not chans:
                skip('no_channels'); continue

            # one prediction_length per call, so different horizons are split
            if cond == 'F' and int(cfg['multivariate']):
                if t_star is None:
                    skip('no_t_star_in_prep'); continue
                groups = [chans]
            elif cond == 'C':
                groups = [[c] for c in chans]
                if len(chans) > 1:
                    _ragged[0] += 1
            elif int(cfg['multivariate']):
                groups = [chans]
            else:
                groups = [[c] for c in chans]

            parts, bad, used_known = [], None, False
            
            for g in groups:
                built, has_cov = build_ts(Wg, observed, scored, g, TimeseriesType, torch, forced_t0=t_star if cond == 'F' else None)
                if built is None:
                    bad = has_cov if isinstance(has_cov, str) else 'build_failed'; break
                ts, H, t0 = built
                try:
                    out = np.asarray(model.forecast([ts], prediction_length=H, output_type='numpy', batch_size=int(cfg['batch_size']))[0])
                except Exception as e:
                    bad = f'error:{type(e).__name__}:{e}'[:300]; break

                if out.shape != (len(g), len(QUANTILES), H):
                    bad = f'shape:{out.shape}!={(len(g), len(QUANTILES), H)}'; break
                used_known = used_known or bool(has_cov)
                for j, c in enumerate(g):  # channel-major, ascending

                    sel = np.where(scored[:, c])[0] - t0
                    if sel.size == 0 or sel.min() < 0 or sel.max() >= H:
                        bad = f'selection_outside_horizon:ch{c}'; break
                    q = out[j][:, sel]  # [Q, Kc]
                    parts.append((q[len(QUANTILES) // 2], np.full(sel.size, np.nan), q))
                if bad is not None:
                    break

            if bad is not None or not parts:
                skip(bad or 'no_parts'); continue
            cat = lambda j: np.concatenate([p[j] for p in parts]).astype('float32')  # noqa: E731
            mu = cat(0)

            if mu.size != K:
                skip(f'len_mismatch:{mu.size}!={K}'); continue
            
            mus.append(mu); vrs.append(cat(1))
            qs.append(np.concatenate([p[2] for p in parts], axis=1).astype('float32'))
            status.append('ok_A' if used_known else 'ok_B')

            if (i+1) % 25 == 0:
                elapsed = time.time() - t_start
                print(f'  {os.path.basename(f)}: {i + 1}/{n}  {elapsed:.0f}s elapsed, {elapsed / (i + 1):.2f}s/group, eta {elapsed / (i + 1) * (n - i - 1) / 60:.1f} min', flush=True)

        tg = cfg['tag']
        out_f = f.replace('prep_', tg + '_')
        np.savez_compressed(out_f, **{f'mu_{tg}': _obj_array(mus), f'var_{tg}': _obj_array(vrs), f'q_{tg}': _obj_array(qs),f'qlevels_{tg}': np.asarray(QUANTILES, 'float64'), 'status': np.array(status)})
        ok = sum(str(s).startswith('ok') for s in status)
        print(f"wrote {out_f}   ok {ok}/{n}   (with known futures: {sum(s == 'ok_A' for s in status)}; per-channel fallback on {_ragged[0]} ragged groups; past covariates passed: {_N_PAST[0]} builds so far)", flush=True)
        for k, v in Counter(s for s in status if not str(s).startswith('ok')).most_common():
            print(f'  non-ok x{v}: {k}')


if __name__ == '__main__':
    main()
