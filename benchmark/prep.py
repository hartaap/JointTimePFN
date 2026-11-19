#!/usr/bin/env python3
# Build MIMIC-IV benchmark groups and save model predictions to NPZ

import os
import sys
from collections import defaultdict
import numpy as np
import torch
from harness import build_args, build_grids
from benchmark.mimic_io import load_mimic
from benchmark.models import MODELS, build, predict


cfg = dict(conditions='A,F,C,DRH', max_windows=409, max_targets=10, rate=0.3, out_dir='bench_io', models='', mimic_npz='')
casts = dict(conditions=str, max_windows=int, max_targets=int, rate=float, out_dir=str, models=str, mimic_npz=str)

for key in cfg:
    flag = '--' + key
    if flag in sys.argv:
        i = sys.argv.index(flag)
        cfg[key] = casts[key](sys.argv[i + 1])
        del sys.argv[i:i + 2]

CONDS = [c.strip() for c in cfg['conditions'].split(',') if c.strip()]
for flag, value in (('--csv_path', 'UNUSED'), ('--ppd_ckpt', 'UNUSED'),
                    ('--unified_ckpt', 'UNUSED')):
    if flag not in sys.argv:
        sys.argv += [flag, value]


def gather(grid, scored):
    # Read scored cells channel by channel, and in the same order for all models
    return np.concatenate([grid[np.where(scored[:, c])[0], c]
                           for c in range(scored.shape[1]) if scored[:, c].any()])


def main():
    args = build_args()
    args.use_gpu = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu else 'cpu')
    T = args.seq_len + args.pred_len
    t_star = args.seq_len
    os.makedirs(cfg['out_dir'], exist_ok=True)

    nets = {}
    model_names = [m.strip() for m in cfg['models'].split(',') if m.strip()]
    for name in model_names:
        if name not in MODELS:
            print(f'[skip] unknown model {name}; have {list(MODELS)}')
            continue

        net, n_params, error = build(MODELS[name], args, device)
        if error:
            print(f'[skip] {name}: {error}')
            continue

        nets[name] = (net, MODELS[name], n_params)
        conditions = MODELS[name].get('conditions', CONDS)
        print(f'[ok]   {name}: {n_params / 1000000.0:.2f}M params  conditions={conditions}')

    if not cfg['mimic_npz'] or not os.path.exists(cfg['mimic_npz']):
        raise SystemExit(f"--mimic_npz {cfg['mimic_npz']!r} missing")

    stays, never_measured, meta = load_mimic(cfg['mimic_npz'])
    if stays.shape[1] != T:
        raise SystemExit(f'mimic window {stays.shape[1]} != seq_len+pred_len {T}')

    print(f"mimic_iv: {stays.shape[0]} stays x {T} x {stays.shape[2]}ch, "
          f"{meta['missing_frac']:.4f} never measured")
    marks = torch.zeros(1, T, 4, device=device)

    rng = np.random.default_rng(1000)
    store = defaultdict(list)
    n_windows = 0

    for stay_idx in range(stays.shape[0]):
        if n_windows >= cfg['max_windows']:
            break

        grid = stays[stay_idx]
        missing = never_measured[stay_idx]
        n_channels = grid.shape[1]
        n_targets = min(cfg['max_targets'], n_channels) if cfg['max_targets'] > 0 else n_channels

        for cond in CONDS:
            # F has no target channel, so process it once per window.
            targets_to_try = 1 if cond == 'F' else n_targets
            for target in range(targets_to_try):
                observed, scored = build_grids(cond, T, n_channels, t_star,
                                               target, cfg['rate'], rng)
                observed = observed & ~missing
                scored = scored & ~missing
                if not scored.any():
                    continue

                y = gather(grid, scored)
                store['window'].append(n_windows)
                store['cond'].append(cond)
                store['target'].append(target)
                store['Wg'].append(grid.astype('float32'))
                store['observed'].append(observed)
                store['scored'].append(scored)
                store['y'].append(y.astype('float32'))

                for name, (net, spec, _) in nets.items():
                    if cond not in spec.get('conditions', CONDS):
                        # Use NaNs for conditions a model does not support.
                        store[f'mu_{name}'].append(np.full(len(y), np.nan, 'float32'))
                        store[f'var_{name}'].append(np.full(len(y), np.nan, 'float32'))
                        continue

                    pred_mu, pred_var, factor = predict(net, spec, grid, observed,
                                                        marks, device, t_star,
                                                        want_factor=True)
                    if factor is not None:
                        # Factor rows follow the same channel-major cell order.
                        channels, times = np.where(scored.T)
                        store[f'fac_{name}'].append(factor[times, channels].astype('float32'))

                    store[f'mu_{name}'].append(gather(pred_mu, scored).astype('float32'))
                    store[f'var_{name}'].append(
                        np.full(len(y), np.nan, 'float32') if pred_var is None
                        else gather(pred_var, scored).astype('float32'))

        n_windows += 1

    out = f"{cfg['out_dir']}/prep_mimic_iv_s0.npz"
    np.savez_compressed(out, n_windows=n_windows, T=T, t_star=t_star, window=np.array(store['window']), cond=np.array(store['cond']), target=np.array(store['target']), model_names=np.array(sorted(nets), dtype=object), model_params=np.array([nets[name][2] for name in sorted(nets)]), **{key: np.array(store[key], dtype=object) for key in ('Wg', 'observed', 'scored', 'y')}, **{key: np.array(values, dtype=object) for key, values in store.items() if key.startswith(('mu_', 'var_', 'fac_'))})
    print(f'wrote {out}  ({len(store["y"])} groups, {n_windows} windows)')


if __name__ == '__main__':
    main()
