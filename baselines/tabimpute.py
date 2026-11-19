#!/usr/bin/env python3
# TabImpute 0.0.1 on the shared prep_*.npz cells, scored on its own predictive distribution
import argparse
import glob
import os
import time

import numpy as np
import torch
from tabimpute.interface import (ImputePFN, FullSupportBarDistribution, create_train_test_sets)

# must match benchmark.merge.CRPS_LEVELS
QLEVELS = np.array([0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975])


def gather(grid, scored):
    #Scored cells in channel-major order
    return np.concatenate([grid[np.where(scored[:, c])[0], c]
                           for c in range(scored.shape[1]) if scored[:, c].any()])


def obj(xs):
    a = np.empty(len(xs), dtype=object)
    a[:] = xs
    return a


def normalize(X):
    # Column standardisation from observed entries, as impute() does
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where(np.isnan(sd), 1.0, sd)

    return (X - mu) / (sd + 1e-16), mu, sd


def logits_for(imputer, Xn):
   # The logits the installed get_imputation() computes 
    t = torch.from_numpy(np.ascontiguousarray(Xn)).to(imputer.device)
    train_X, train_y, test_X, test_y = create_train_test_sets(t, t)
    input_y = torch.cat((train_y, torch.full_like(test_y, float('nan'), device=imputer.device)), dim=0)
    X_input = torch.cat((train_X.to(imputer.device), test_X.to(imputer.device)),dim=0).unsqueeze(0).float()

    with torch.no_grad():
        preds = imputer.model(X_input, input_y.to(imputer.device).float().unsqueeze(0))

    return preds.squeeze(0)[train_y.shape[0]:], np.where(np.isnan(Xn))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', default='tabpfn_io')
    p.add_argument('--device', default='cuda')
    p.add_argument('--tag', default='tabimpute')
    p.add_argument('--conditions', default='A,F,C,DRH')
    a = p.parse_args()
    conds = {c.strip() for c in a.conditions.split(',') if c.strip()}

    imputer = ImputePFN(device=a.device)

    if imputer.preprocessors is not None:
        raise SystemExit('This evaluator assumes preprocessors=None')
    
    bar = FullSupportBarDistribution(borders=imputer.borders.to(imputer.device))

    if getattr(imputer, 'max_num_rows', None) is not None:
        raise SystemExit('This evaluator requires ImputePFN(max_num_rows=None)')

    files = sorted(glob.glob(os.path.join(a.in_dir, 'prep_*.npz')))

    if not files:
        raise SystemExit(f'no prep_*.npz in {a.in_dir}')

    for path in files:
        d = np.load(path, allow_pickle=True)
        Wg, Obs, Sc, Cond = d['Wg'],d['observed'],d['scored'],d['cond']
        Y_all = d['y'] if 'y' in d.files else None
        n_all = len(Wg)
        mus, meds, vrs, nlls, qs, los, his, status = [], [], [], [], [], [], [], []

        t0 = time.time()
        for i in range(n_all):
            if i and i % 100 == 0:
                print(f'  {i}/{n_all} groups, {(time.time() - t0) / i:.1f} s/group', flush=True)
            W = np.asarray(Wg[i], float)
            observed = np.asarray(Obs[i], bool)
            scored = np.asarray(Sc[i], bool)
            # truth in the channel-major order every prediction uses
            y = gather(W, scored)
            K = int(scored.sum())
            nan = np.full(K, np.nan, 'float32')
            qnan = np.full((len(QLEVELS), K), np.nan, 'float32')

            def blank(tag):
                mus.append(nan); meds.append(nan.copy()); vrs.append(nan.copy()); nlls.append(nan.copy())
                qs.append(qnan);los.append(nan.copy()); his.append(nan.copy())
                status.append(tag)

            if str(Cond[i]) not in conds:
                blank('cond_skipped'); continue

            if Y_all is not None:
                y_prep = np.asarray(Y_all[i], float)
                assert y_prep.shape == y.shape and np.allclose(y_prep, y, equal_nan=True), i

            try:
                # model and input: observed cells visible, the rest NaN
                X = W.copy()
                X[~observed] = np.nan

                Xn, mu_c, sd_c = normalize(X)
                logits, missing_idx = logits_for(imputer, Xn)

                # logits back on the grid, then gathered channel-major
                idx = tuple(t.cpu().numpy() if torch.is_tensor(t) else t for t in missing_idx) \
                    if isinstance(missing_idx, tuple) else missing_idx
                
                rows = np.full(W.shape, -1, dtype=np.int64)
                rows[idx] = np.arange(logits.shape[0])
                sel = gather(rows, scored)

                if (sel < 0).any():
                    raise RuntimeError('a scored cell was not among the missing cells')
                
                sub = logits[torch.as_tensor(sel, device=logits.device)]

                sd_cell = gather(np.broadcast_to(sd_c, W.shape).copy(), scored)
                ycell = torch.as_tensor((y - gather(np.broadcast_to(mu_c, W.shape).copy(), scored))/(sd_cell + 1e-16), dtype=sub.dtype, device=sub.device)

                # native Nll plus the jacobian: -log q_Y(y) = -log q_Z(z) + log s
                with torch.no_grad():
                    nll_z = bar(sub, ycell).cpu().numpy().astype(float)  # its training loss

                nll_y = nll_z + np.log(sd_cell + 1e-16)

                # point forecast: the mean of the same bar distribution, denormalised
                with torch.no_grad():
                    mz = bar.mean(sub.float()).cpu().numpy().astype(float)
                mu = mz * (sd_cell + 1e-16) + gather(np.broadcast_to(mu_c, W.shape).copy(), scored)

                # the median impute() returns: bar.median of the same logits, denormalised
                with torch.no_grad():
                    mdz = bar.median(logits=sub.float()).cpu().numpy().astype(float)

                med = mdz * (sd_cell + 1e-16) + gather(np.broadcast_to(mu_c, W.shape).copy(), scored)

                # native quantiles, denormalised
                with torch.no_grad():
                    qz = np.stack([bar.icdf(sub, float(t)).cpu().numpy() for t in QLEVELS])

                qy = qz * (sd_cell + 1e-16) + gather(np.broadcast_to(mu_c, W.shape).copy(), scored)
                crossed = bool((np.diff(qy, axis=0) < 0).any())

                if crossed:
                    qy = np.full_like(qy, np.nan)

                if mu.shape != (K,) or qy.shape != (len(QLEVELS), K) or nll_y.shape != (K,):
                    raise RuntimeError(f'shape {mu.shape}/{qy.shape}/{nll_y.shape}, K={K}')

                mus.append(mu.astype('float32'))
                meds.append(med.astype('float32'))
                # var stays NaN, the density is not Gaussian, and merge derives NLL/Cov95 from var
                vrs.append(nan.copy())
                nlls.append(nll_y.astype('float32'))
                qs.append(qy.astype('float32'))
                los.append(qy[0].astype('float32'))
                his.append(qy[-1].astype('float32'))
                status.append('ok_crossed' if crossed else 'ok')
                
            except Exception as e:
                blank(f'error:{type(e).__name__}:{e}'[:300])

        tg = a.tag
        out = path.replace('prep_', f'{tg}_')
        np.savez_compressed(out, **{f'mu_{tg}': obj(mus), f'med_{tg}': obj(meds), f'var_{tg}': obj(vrs), f'nll_{tg}': obj(nlls), f'q_{tg}': obj(qs), f'qlevels_{tg}': QLEVELS, f'lo_{tg}': obj(los), f'hi_{tg}': obj(his), 'status': np.asarray(status)})
        u, c = np.unique([s.split(':')[0] for s in status], return_counts=True)
        print(f'{os.path.basename(out)}: {dict(zip(u.tolist(), c.tolist()))}', flush=True)


if __name__ == '__main__':
    main()
