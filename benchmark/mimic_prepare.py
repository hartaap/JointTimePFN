#!/usr/bin/env python3
# Turn the MIMIC-IV BigQuery vitals export into model-ready windows
import argparse
import glob
import os

import numpy as np
import pandas as pd

from benchmark.mimic_io import save_mimic


CORE = ['heart_rate', 'resp_rate', 'spo2', 'sbp', 'dbp', 'mbp']

EHR = CORE + ['temperature', 'glucose', 'fio2', 'peep']

# physiological bounds, applied before standardizing
VALID = {'heart_rate': (20, 300), 'sbp': (40, 300), 'dbp': (20, 200), 'mbp': (20, 250),
    'resp_rate': (2, 80), 'spo2': (50, 100), 'temperature': (30, 45),
    'glucose': (20, 1000), 'fio2': (21, 100), 'peep': (0, 30) }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='mimic_iv_data/*.csv')
    p.add_argument('--out', default='dataset/mimic_iv')
    a = p.parse_args()

    src = sorted(glob.glob(a.src))
    assert src, f'no file matched {a.src}'
    # a split export would lose all but the first file
    assert len(src) == 1, f'{len(src)} files matched {a.src}; expected the single export'
    # the paper's presented cohort, pinned
    chans = EHR
    W = 192  # = seq_len + pred_len
    B = 3  # hours per cell
    WH = W*B  # contiguous hours per stay

    df = pd.read_csv(src[0])
    print(f'loaded {len(df):,} stay-hours, {df.stay_id.nunique():,} stays')

    # range filter
    for c in chans:
        if c in VALID and c in df.columns:
            lo, hi = VALID[c]
            bad = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
            if bad.any():
                print(f'  range-filtered {c}: {bad.sum():,} values outside [{lo},{hi}] set to NaN')
            df.loc[bad, c] = np.nan

    # contiguity
    df = df.sort_values(['stay_id', 'hours_since_admission'], kind='mergesort')
    head = df.groupby('stay_id', sort=False).head(WH)
    n = head.groupby('stay_id', sort=False)['hours_since_admission'].agg(['size', 'min', 'max'])

    ok = (n['size'] == WH) & (n['max'] - n['min'] == WH - 1)
    keep = n.index[ok]

    print(f'stays with {WH} contiguous hours ({WH / 24:.0f} d): {len(keep):,} / {len(n):,}' + ('' if ok.all() else '   (rest dropped: gaps or too short)'))
    head = head[head.stay_id.isin(keep)]

    # bin: a cell is observed when any of its B hours was measured
    head = head.sort_values(['stay_id', 'hours_since_admission'], kind='mergesort')
    head['_cell'] = head.groupby('stay_id', sort=False).cumcount() // B
    head = (head.groupby(['stay_id', '_cell'], sort=False, as_index=False)[chans].mean().sort_values(['stay_id', '_cell'], kind='mergesort').reset_index(drop=True))

    per = head.groupby('stay_id', sort=False).size()
    assert (per == W).all(), f'binning did not yield {W} cells/stay: {per.value_counts()}'
    print(f'binned {B} h per cell, {W} cells covering {WH} h per stay')

    # standardise on all eligible stays (the evaluation set), and from observed cells only
    parts = {'test': set(keep)}
    col = head['stay_id']
    mu = head[chans].mean(skipna=True).values.astype('float64')
    sd = head[chans].std(skipna=True).values.astype('float64')
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0

    print(f'\nper-channel coverage at {B} h per cell / mean / sd:')
    for i, c in enumerate(chans):
        print(f'   {c:14s} {100 * head[c].notna().mean():5.1f}%   {mu[i]:9.2f}   {sd[i]:8.2f}')

    os.makedirs(a.out, exist_ok=True)
    
    for name, ids in parts.items():
        sub = head[col.isin(ids)]
        stays = sub.stay_id.drop_duplicates().values
        V = sub[chans].values.astype('float64').reshape(-1, W, len(chans))
        m = ~np.isfinite(V)  # True = not measured anywhere
        V = (V - mu) / sd 
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        # values keep their NaNs, so load_mimic returns the mask with them
        save_mimic(os.path.join(d, 'mimic.npz'), V, stays, chans, W)
        print(f'  {name:5s}: {len(stays):,} stays x {W} cells x {len(chans)} ch   missing {100 * m.mean():.1f}%')

    np.savez(os.path.join(a.out, 'scaler.npz'), mean=mu, std=sd, channels=np.array(chans))
    print(f'\nwrote {a.out}/')


if __name__ == '__main__':
    main()
