# The shared evaluation harness: the CLI contract, the window loader and the mask conditions

import argparse
import csv

import numpy as np
import torch


def build_args():
    p = argparse.ArgumentParser()
    for name, kw in [
        ('--joint_rank', dict(type=int, default=32)),
        ('--is_training', dict(type=int, default=0)), ('--model_id', dict(default='xm')),
        ('--model', dict(required=True)), ('--data', dict(default='ETTh1')),
        ('--root_path', dict(default='./dataset/ETT-small/')), ('--data_path', dict(default='ETTh1.csv')),
        ('--features', dict(default='M')), ('--target', dict(default='OT')), ('--freq', dict(default='h')),
        ('--checkpoints', dict(default='./checkpoints/')),
        ('--seq_len', dict(type=int, default=96)), ('--label_len', dict(type=int, default=48)),
        ('--pred_len', dict(type=int, default=96)), ('--enc_in', dict(type=int, default=7)),
        ('--d_model', dict(type=int, default=512)), ('--n_heads', dict(type=int, default=8)),
        ('--e_layers', dict(type=int, default=2)), ('--d_ff', dict(type=int, default=2048)), ('--factor', dict(type=int, default=1)), ('--dropout', dict(type=float, default=0.1)),
        ('--embed', dict(default='timeF')), ('--activation', dict(default='gelu')),
        ('--optimizer', dict(default='adam')), ('--num_workers', dict(type=int, default=4)),
        ('--itr', dict(type=int, default=1)), ('--train_epochs', dict(type=int, default=1)),
        ('--batch_size', dict(type=int, default=16)), ('--patience', dict(type=int, default=20)),
        ('--learning_rate', dict(type=float, default=5e-4)), ('--des', dict(default='Exp')),
        ('--lradj', dict(default='synthetic')),
        ('--gpu', dict(type=int, default=0)), ('--devices', dict(default='0,1,2,3')),
        ('--exp_name', dict(default='unified_masked')), ('--class_strategy', dict(default='projection')),
        ('--patch_size', dict(type=int, default=16)), ('--embed_dim', dict(type=int, default=128)),
        ('--pct_start', dict(type=float, default=0.3)), ('--synthetic_data_path', dict(default='')), ('--synthetic_root_path', dict(default='./dataset/synthetics/')),
        ('--synthetic_length', dict(type=int, default=1024)), ('--stride', dict(type=int, default=8)),
        ('--data_amount', dict(type=int, default=-1)), ('--max_channel', dict(type=int, default=-1)),
        ('--total_synthetic_series', dict(type=int, default=15000)),
    ]:
        p.add_argument(name, **kw)
    for flag in ['--output_attention', '--use_multi_gpu', '--inverse',
                 '--combine_Gaussian_datasets', '--ordered_data', '--load']:
        p.add_argument(flag, action='store_true', default=False)
    p.add_argument('--use_norm', type=int, default=True)
    p.add_argument('--use_gpu', type=bool, default=True)
    p.add_argument('--unified_ckpt', type=str, required=True)
    p.add_argument('--ppd_ckpt', type=str, default='')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--csv_path', type=str, default='', help='load windows from this CSV')
    p.add_argument('--standardize', action='store_true', help='per-channel z-score of the --csv_path data')
    p.add_argument('--use_rope', action='store_true', help='RoPE, which must match the checkpoint')
    p.add_argument('--rope_base', type=float, default=10.0, help='RoPE base, which must match the checkpoint, and note that it does not store it')
    return p.parse_args()


def _is_float(v):
    try:
        float(v); return True
    except ValueError:
        return False


def csv_window_batches(csv_path, T, stride, batch_size, standardize=False):
    """Yield (full, full_mark) T-window batches from a CSV."""
    rows = []
    with open(csv_path) as f:
        rd = csv.reader(f); next(rd)
        first = next(rd)
        num_idx = [j for j, v in enumerate(first) if _is_float(v)]
        rows.append([float(first[j]) for j in num_idx])
        for row in rd:
            rows.append([float(row[j]) for j in num_idx])
    arr = np.asarray(rows, dtype=np.float32)  # [Ttot, N]
    if standardize:
        mu = arr.mean(axis=0, keepdims=True)
        sd = arr.std(axis=0, keepdims=True) + 1e-8
        arr = (arr - mu) / sd
        print("standardized per-channel (z-score, whole-series stats)")
    Ttot, N = arr.shape
    starts = list(range(0, Ttot - T + 1, stride))
    wins = np.stack([arr[s:s + T] for s in starts])  # [W, T, N]
    print(f"CSV: {Ttot} steps x {N} channels, {len(wins)} windows of {T} (stride {stride})")
    for i in range(0, len(wins), batch_size):
        full = torch.from_numpy(wins[i:i + batch_size]).float()
        yield full, torch.zeros(full.shape[0], T, 4)


def build_grids(condition, T, N, t_star, target, rate, rng):
    observed = np.zeros((T, N), bool); scored = np.zeros((T, N), bool)
    if condition == 'A':
        scored[t_star:, target] = True
        observed[:] = True; observed[t_star:, target] = False
    elif condition == 'F':  # B in paper
        observed[:t_star, :] = True
        scored[t_star:, :] = True
    elif condition == 'C':
        observed[:] = True
        for c in range(N):
            cut = int(rng.integers(t_star, T))
            observed[cut:, c] = False; scored[cut:, c] = True
    elif condition == 'DRH':  # D in paper
        for c in range(N):
            r_c = float(rng.uniform(0.4, 0.9))
            m_c = rng.random(T) < r_c
            scored[m_c, c] = True; observed[~m_c, c] = True
    else:
        raise ValueError(condition)
    return observed, scored
