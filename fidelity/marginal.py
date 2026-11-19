#!/usr/bin/env python3
# Compare PFN and NUTS predictions with the exact oracle on composed LMC datasets.

import glob as _glob
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from harness import build_args, build_grids, csv_window_batches
from benchmark.models import D_FLOOR
from experiments.exp_unified_masked import Exp_UnifiedMasked
from fidelity.lmc_dense import lmc_predict
from fidelity.nuts import nuts_composed_predict
from utils.metrics import kl_terms, cell_scores

# Experiment settings, command-line options can override these defaults
cfg = dict(
    conditions='A,F,C,DRH', data_seeds='0', max_windows=15, max_targets=5,
    rate=0.3, n_boot=2000,
    nuts_warmup=1000, nuts_samples=1000, nuts_max_windows=5, nuts_max_targets=1,
    nuts_chains=4, nuts_tree_depth=7,
    ckpts='', out='marginal.json',
    composed_glob=''  # One directory per composed dataset configuration
)
casts = dict(
    conditions=str, data_seeds=str, max_windows=int, max_targets=int, rate=float,
    n_boot=int, nuts_warmup=int, nuts_samples=int, nuts_max_windows=int,
    nuts_max_targets=int, nuts_chains=int, nuts_tree_depth=int,
    ckpts=str, out=str, composed_glob=str
)


for _k in list(cfg):
    _f = '--' + _k
    if _f in sys.argv:
        _i = sys.argv.index(_f)
        cfg[_k] = casts[_k](sys.argv[_i + 1])
        del sys.argv[_i:_i + 2]


def popflag(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


DO_NUTS = popflag('--nuts')

CONDS = [c.strip() for c in cfg['conditions'].split(',') if c.strip()]
DATA_SEEDS = [int(s) for s in cfg['data_seeds'].split(',') if s.strip()]

CKPTS = []

for _c in cfg['ckpts'].split(','):
    _c = _c.strip()
    if not _c:
        continue
    CKPTS += sorted(_glob.glob(_c)) if any(ch in _c for ch in '*?[') else [_c]


if '--csv_path' not in sys.argv:
    sys.argv += ['--csv_path', 'UNUSED']
if '--ppd_ckpt' not in sys.argv:
    sys.argv += ['--ppd_ckpt', 'UNUSED']
if '--unified_ckpt' not in sys.argv:
    sys.argv += ['--unified_ckpt', CKPTS[0] if CKPTS else 'UNUSED']


def build_grid():
    # One configuration for each directory with params.npz
    out = []
    for d in sorted(_glob.glob(cfg['composed_glob'])) if cfg['composed_glob'] else []:
        if not os.path.exists(f'{d}/params.npz'):
            continue
        params = np.load(f'{d}/params.npz', allow_pickle=True)
        out.append(('composed', os.path.basename(d).split('_')[-1], {'kernel': 'composed', 'N': int(params['W'].shape[0]),
             'Q': int(params['n_latents']), 'composed_dir': d}
        ))
    return out


#  Metrics 

def gather(grid_TN, scored):
    # Flatten selected cells channel by channel, matching the oracle order
    return np.concatenate([grid_TN[np.where(scored[:, c])[0], c] for c in range(scored.shape[1]) if scored[:, c].any()])


def cluster_boot(vals, n_boot):
    # Bootstrap over window level clusters
    seed = 0
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return dict(mean=float('nan'), lo=float('nan'), hi=float('nan'), se=float('nan'), n=0)
    if a.size == 1:
        return dict(mean=float(a[0]), lo=float(a[0]), hi=float(a[0]), se=float('nan'), n=1)

    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(1)

    return dict(mean=float(a.mean()), lo=float(np.percentile(bs, 2.5)),hi=float(np.percentile(bs, 97.5)), se=float(bs.std(ddof=1)), n=int(a.size))


# Evaluate one composed dataset using one checkpoint
def eval_dataset(exp, dev, ddir, seed, T, t_star, conds, max_windows, max_targets, rate):

    d = np.load(f'{ddir}/params.npz', allow_pickle=True)
    kern = str(d['kernel'])

    if kern != 'composed':
        raise ValueError(f"the fidelity suite is composed-kernel only, got {kern!r}")
    with open(f'{ddir}/kernels.pkl', 'rb') as fh:
        composed_kernels = pickle.load(fh)
    oracle = dict(W=d['W'], noise_var=float(d['noise_var']), composed_kernels=composed_kernels)

    rec = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rng = np.random.default_rng(1000 + seed)  # mask RNG
    nw = 0

    for full, fmk in csv_window_batches(f'{ddir}/toy_oracle.csv', T, T, 1, standardize=False):
        if nw >= max_windows:
            break
        Wg = full[0].numpy()
        N = Wg.shape[1]
        marks = exp._maybe_marks(fmk)
        n_tgt = min(max_targets, N) if max_targets > 0 else N
        cell = defaultdict(lambda: defaultdict(list))

        for cond in conds:
            pfn_cache = {}

            for tgt in range(n_tgt):
                observed, scored = build_grids(cond, T, N, t_star, tgt, rate, rng)
                if not scored.any():
                    continue
                y = gather(Wg, scored)

                # reuse the PFN output if the observation mask is the same here
                key = observed.tobytes()
                if key not in pfn_cache:
                    fm = torch.tensor((~observed)[None].astype('float32'), device=dev)
                    with torch.no_grad():
                        out = exp.model(full.float().to(dev) * (1 - fm), fm, marks)

                    # Joint model: include the factor contribution in marginal variance
                    if len(out) == 3:
                        m, lv, fac = out
                        var = (np.maximum(np.exp(np.clip(lv[0].cpu().numpy(), -10, 10)), D_FLOOR) + (fac[0].cpu().numpy() ** 2).sum(-1))
                    else:
                        m, lv = out
                        var = np.exp(np.clip(lv[0].cpu().numpy(), -10, 10))
                    pfn_cache[key] = (m[0].cpu().numpy(), var)
                mg, vg = pfn_cache[key]
                mup, varp = gather(mg, scored), gather(vg, scored)

                for k, v in cell_scores(mup, varp, y).items():
                    cell[(cond, 'pfn')][k].append(v)

                # Exact oracle predictions for the same scored cells 
                gen_len = int(d['length']) if 'length' in d else T
                x_win = np.linspace(0.0, 1.0, gen_len)[nw * T: nw * T + T]
                muo, varo = lmc_predict(Wg, observed, scored, oracle['composed_kernels'], oracle['W'], oracle['noise_var'], x=x_win)

                for k, v in cell_scores(muo, varo, y).items():
                    cell[(cond, 'oracle')][k].append(v)

                cell[(cond, 'pfn')]['kl'].append(kl_terms(mup, varp, muo, varo)[0])
                # Reverse KL
                cell[(cond, 'pfn')]['kl_orc_vs_pfn'].append(kl_terms(muo, varo, mup, varp)[0])

                # Run NUTS only on the selected subset of windows and targets
                if DO_NUTS and nw < cfg['nuts_max_windows'] and tgt < cfg['nuts_max_targets']:
                    mun, varn, nuts_info = nuts_composed_predict(
                        Wg, observed, scored, oracle['composed_kernels'], x_win,
                        warmup=cfg['nuts_warmup'], samples=cfg['nuts_samples'],
                        num_chains=cfg['nuts_chains'], seed=seed,
                        max_tree_depth=cfg['nuts_tree_depth'], verbose=False)

                    # Keep the NUTS diagnostics in the output
                    cell[(cond, 'nuts')]['chain_spread'].append(float(nuts_info['chain_spread']))
                    cell[(cond, 'nuts')]['slots_outside_band'].append(float(nuts_info['n_slots_outside_band']))

                    for stat in ('rhat_max', 'n_pinned', 'n_active', 'rhat_pred', 'rhat_wwT', 'ess_min', 'ess_pred', 'ess_lpd', 'rhat_lpd', 'divergences'):
                        cell[(cond, 'nuts')][stat].append(float(nuts_info.get(stat, float('nan'))))

                    for k, v in cell_scores(mun, varn, y).items():
                        cell[(cond, 'nuts')][k].append(v)

                    cell[(cond, 'nuts')]['kl'].append(kl_terms(mun, varn, muo, varo)[0])

                    # KL(orc || N)
                    cell[(cond, 'nuts')]['kl_orc_vs_nuts'].append(kl_terms(muo, varo, mun, varn)[0])

                    # Matched PFN and oracle results on the same cells as NUTS.
                    for k, v in cell_scores(muo, varo, y).items():
                        cell[(cond, 'oracle_m')][k].append(v)
                    for k, v in cell_scores(mup, varp, y).items():
                        cell[(cond, 'pfn_m')][k].append(v)
                    cell[(cond, 'pfn_m')]['kl'].append(kl_terms(mup, varp, muo, varo)[0])
                    cell[(cond, 'pfn_m')]['kl_orc_vs_pfn'].append(
                        kl_terms(muo, varo, mup, varp)[0])

                    # KL(N ||Ours)
                    cell[(cond, 'pfn')]['kl_nuts_vs_pfn'].append(kl_terms(mun, varn, mup, varp)[0])

        # Average over cells and targets within each window
        for (cond, meth), md in cell.items():
            for k, arrs in md.items():
                flat = np.concatenate([np.atleast_1d(np.asarray(a, float)) for a in arrs])
                flat = flat[np.isfinite(flat)]
                if flat.size == 0:
                    continue
                rec[cond][meth][k].append(float(flat.mean()))
        nw += 1
    return rec, kern, nw


def main():
    args = build_args()
    args.use_gpu = torch.cuda.is_available()
    T = args.seq_len + args.pred_len
    t_star = args.seq_len
    ckpts = CKPTS or [args.unified_ckpt]
    out_dir = os.path.dirname(cfg['out'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    grid = build_grid()

    print(f"grid: {len(grid)} configs x {len(DATA_SEEDS)} data seeds x {len(ckpts)} model seeds")
    print(f"conditions {CONDS} | <= {cfg['max_windows']} disjoint windows " f"x {cfg['max_targets']} targets | nuts={('on' if DO_NUTS else 'off')}")

    if not grid:
        sys.exit(f"--composed_glob {cfg['composed_glob']!r} matched no dataset directory")

    exp = Exp_UnifiedMasked(args)
    dev = exp.device
    results = {
        'meta': dict(
            data_seeds=DATA_SEEDS, ckpts=ckpts, conditions=CONDS,
            max_windows=cfg['max_windows'], max_targets=cfg['max_targets'],
            T=T, t_star=t_star, n_boot=cfg['n_boot'], stride=T,
            cluster='(data_seed, window)',
            nuts=dict(
                on=bool(DO_NUTS), warmup=cfg['nuts_warmup'],
                samples=cfg['nuts_samples'], chains=cfg['nuts_chains'],
                max_windows=cfg['nuts_max_windows'],
                max_targets=cfg['nuts_max_targets'],
                tree_depth=cfg['nuts_tree_depth']
            )
        ),
        'configs': []
    }

    for gi, (axis, label, c) in enumerate(grid, 1):
        print(f"[{gi}/{len(grid)}] {os.path.basename(c['composed_dir'])}  N={c['N']} Q={c['Q']}", flush=True)

        per_model = []
        kern_seen, nwin = None, 0
        for ci, ck in enumerate(ckpts):
            exp.model.load_state_dict(torch.load(ck, map_location='cpu'))
            exp.model.eval()
            merged = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            # Each composed directory is one prior draw, so do not repeat data seeds.
            for s in ([0] if 'composed_dir' in c else DATA_SEEDS):
                dd = c['composed_dir']
                rec, kern, nw = eval_dataset(exp, dev, dd, s, T, t_star, CONDS, cfg['max_windows'], cfg['max_targets'], cfg['rate'])
                kern_seen = kern

                if ci == 0:
                    nwin += nw

                for cond, methods in rec.items():
                    for meth, metrics in methods.items():
                        for metric, values in metrics.items():
                            merged[cond][meth][metric].extend(values)
            per_model.append(merged)

        entry = dict(axis=axis, label=label, config=dict(c), kernel=kern_seen,
                     n_clusters=nwin, conditions={})

        for cond in CONDS:
            cd = {}
            for meth in ('oracle', 'nuts', 'oracle_m', 'pfn_m', 'pfn'):
                if meth not in per_model[0][cond]:
                    continue
                md = {}
                # the relevant metrics in the json file
                for k in ('mse', 'cov', 'z2', 'kl', 'kl_orc_vs_pfn',
                          'kl_orc_vs_nuts', 'kl_nuts_vs_pfn',
                          'chain_spread', 'slots_outside_band',
                          'rhat_max', 'n_pinned', 'n_active', 'rhat_pred', 'rhat_wwT', 'ess_min', 'ess_pred', 'ess_lpd', 'rhat_lpd', 'divergences'):
                    pooled = [v for pm in per_model for v in pm[cond][meth].get(k, [])]
                    if pooled:
                        md[k] = cluster_boot(pooled, cfg['n_boot'])
                cd[meth] = md

            entry['conditions'][cond] = cd

            mean_of = lambda meth, k: cd.get(meth, {}).get(k, {}).get('mean', float('nan'))
            line = f"  {cond:4} KL(orc|Ours) {mean_of('pfn_m' if 'pfn_m' in cd else 'pfn', 'kl_orc_vs_pfn'):.4f}"
            if 'nuts' in cd: 
                line += (f"  KL(orc|N) {mean_of('nuts', 'kl_orc_vs_nuts'):.4f}" f"  KL(N|Ours) {mean_of('pfn', 'kl_nuts_vs_pfn'):.4f}" f"  spread {mean_of('nuts', 'chain_spread'):.4f}  rhat_pred {mean_of('nuts', 'rhat_pred'):.3f}")
            print(line)

        results['configs'].append(entry)
        with open(cfg['out'], 'w') as f:  # Save after every configuration.
            json.dump(results, f, indent=2)

    with open(cfg['out'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {cfg['out']}")


if __name__ == '__main__':
    main()
