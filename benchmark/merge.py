#!/usr/bin/env python3
# Score every model's MIMIC-IV predictions on the same cells and bootstrap over windows.

import glob
import importlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
from utils.metrics import gaussian_nll, covered95, crps, joint_scores, DECILES, Z_DECILES


# Some prediction files were saved with a newer NumPy version.
if not hasattr(np, '_core'):
    for module in ('', '.multiarray', '.numeric', '.umath', '._multiarray_umath', '.overrides'):
        try:
            sys.modules['numpy._core' + module] = importlib.import_module('numpy.core' + module)
        except ImportError:
            pass


cfg = dict(in_dir='tabpfn_io', out='bench_all.json', n_boot=4000, drop_wins='')
for key in list(cfg):
    flag = '--' + key
    if flag in sys.argv:
        i = sys.argv.index(flag)
        cfg[key] = type(cfg[key])(sys.argv[i + 1])
        del sys.argv[i:i + 2]


# Baselines saved in their own files: (file prefix, key in that file, name in the tables)
BASELINES = [('tabpfnts_', 'tabpfnts', 'tabpfn-ts'),
             ('tabimpute_', 'tabimpute', 'tabimpute'),
             ('chronos2_', 'chronos2', 'chronos-2 (120M)'),
             ('chronos2small_', 'chronos2small', 'chronos-2 (28M)'),
             ('tirex2_', 'tirex2', 'tirex-2')]

METRICS = ('mse', 'crps_d', 'nll', 'cov95', 'jnll', 'jnll_diag', 'jgain')
# CRPS uses the nine deciles for every model, since every model can give them


def bootstrap(values, n_boot):
    values = np.asarray([v for v in values if np.isfinite(v)], float)
    if values.size == 0:
        return None
    if values.size == 1:
        return dict(mean=float(values[0]), lo=float(values[0]), hi=float(values[0]), n=1)
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(1)
    return dict(mean=float(values.mean()), lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)), n=int(values.size))


def score(mu, var, y, q, levels, nll_native):
    s = {'mse': (mu - y) ** 2}
    if var is not None:
        s['nll'] = gaussian_nll(mu, var, y)
        s['cov95'] = covered95(mu, var, y).astype(float)
    if nll_native is not None:
        s['nll'] = nll_native                                      # the model's own density
    if q is not None:
        # native quantiles: deciles for CRPS, 0.025 and 0.975 for coverage when given
        if all(x in levels for x in DECILES):
            s['crps_d'] = crps(q[[levels.index(x) for x in DECILES]], y)
        if 0.025 in levels and 0.975 in levels:
            s['cov95'] = ((y >= q[levels.index(0.025)]) & (y <= q[levels.index(0.975)])).astype(float)
    elif var is not None:
        sd = np.sqrt(np.maximum(var, 1e-12))
        s['crps_d'] = crps(mu[None, :] + sd[None, :] * Z_DECILES[:, None], y)
    return s


def load(prep_file):
    prep = np.load(prep_file, allow_pickle=True)
    names = [str(n) for n in prep['model_names']]
    params = dict(zip(names, map(int, prep['model_params']))) if 'model_params' in prep else {}
    # name -> the arrays of that model, one entry per group
    models = {n: dict(mu=prep[f'mu_{n}'], var=prep.get(f'var_{n}'), fac=prep.get(f'fac_{n}')) for n in names}
    for prefix, key, name in BASELINES:
        path = prep_file.replace('prep_', prefix)
        if os.path.exists(path):
            f = np.load(path, allow_pickle=True)
            models[name] = dict(mu=f[f'mu_{key}'], var=f[f'var_{key}'], q=f.get(f'q_{key}'),
                                levels=f.get(f'qlevels_{key}'), nll=f.get(f'nll_{key}'))
    return prep, models, params


def dropped_windows():
    spec = cfg['drop_wins'].strip()
    if not spec:
        return set()
    text = open(spec).read() if os.path.exists(spec) else spec.replace(',', ' ')
    return {int(x) for line in text.splitlines() for x in line.split('#', 1)[0].split()}


def finite(a, size):
    a = np.asarray(a, float)
    return a if a.size == size and np.all(np.isfinite(a)) else None


def main():
    files = sorted(glob.glob(f"{cfg['in_dir']}/prep_*.npz"))
    if not files:
        raise SystemExit(f"no prep_*.npz in {cfg['in_dir']}, run benchmark/prep.py --models first")

    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # (dataset, cond) -> model -> metric -> window means
    params = {}
    drop = dropped_windows()
    for prep_file in files:
        dataset = os.path.basename(prep_file).split('prep_')[1].rsplit('_s', 1)[0]
        prep, models, p = load(prep_file)
        params.update(p)
        if drop:
            ids = {int(w) for w in prep['window']}
            print(f'  [drop_wins] excluding {len(drop & ids)} of {len(ids)} window ids from every arm, '
                  f'leaving {len(ids - drop)}', file=sys.stderr)

        cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # (window, cond) -> model -> metric -> cells
        for i, y in enumerate(prep['y']):
            window, cond = int(prep['window'][i]), str(prep['cond'][i])
            if window in drop:
                continue
            y = np.asarray(y, float)
            for name, m in models.items():
                mu = finite(m['mu'][i], y.size)
                if mu is None:
                    continue                                       # a dash in the tables
                var = finite(m['var'][i], y.size) if m['var'] is not None else None
                q, levels = None, None
                if m.get('q') is not None and m.get('levels') is not None:
                    q = np.asarray(m['q'][i], float)
                    q = q if q.ndim == 2 and q.shape[1] == y.size and np.all(np.isfinite(q)) else None
                    levels = [round(float(x), 6) for x in np.ravel(m['levels'])]
                nll = finite(m['nll'][i], y.size) if m.get('nll') is not None else None
                s = score(mu, var, y, q, levels, nll)
                if m.get('fac') is not None and var is not None:
                    js = joint_scores(mu, var, np.asarray(m['fac'][i], float), y)
                    if js is not None:
                        for key, value in zip(('jnll', 'jnll_diag', 'jgain'), js):
                            s[key] = np.full(y.size, value)
                for metric, v in s.items():
                    cells[(window, cond)][name][metric].append(v)

        # the mean over the cells of a window, then the windows are bootstrapped
        for (window, cond), by_model in cells.items():
            for name, by_metric in by_model.items():
                for metric, v in by_metric.items():
                    scores[(dataset, cond)][name][metric].append(float(np.concatenate(v).mean()))

    results = {}
    print(f"\n{'dataset':<12}{'cond':<8}{'model':<13}" + ''.join(f'{m:>13}' for m in METRICS))
    for (dataset, cond), by_model in sorted(scores.items()):
        for name in sorted(by_model):
            stats = {metric: bootstrap(v, cfg['n_boot']) for metric, v in by_model[name].items()}
            results[f'{dataset}|{cond}|{name}'] = {k: v for k, v in stats.items() if v}
            print(f'{dataset:<12}{cond:<8}{name:<13}' + ''.join(
                f"{stats[m]['mean']:>13.4f}" if stats.get(m) else f"{'--':>13}" for m in METRICS))
        print()
    results['_params'] = params
    with open(cfg['out'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {cfg['out']}")


if __name__ == '__main__':
    main()
