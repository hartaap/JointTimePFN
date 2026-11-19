#!/usr/bin/env python3
# Native Chronos-2 on the benchmark conditions A, F, C (paper A, B, C)

import argparse
import glob
import os
import time

import numpy as np
from chronos import Chronos2Pipeline

QLEVELS = np.array([0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975], dtype=np.float64)
INTERIOR = {"DRH"}

def obj(xs):
    a = np.empty(len(xs), dtype=object)
    a[:] = xs
    return a

def masked_values(W, observed, rows, c):
    # channel c on rows, NaN where that cell is not observed
    rows = np.asarray(rows, dtype=int)
    out = np.full(rows.size, np.nan, dtype=np.float32)
    m = observed[rows, c]
    out[m] = W[rows[m], c].astype(np.float32, copy=False)
    return out


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _validate_output(point, q, H, n_variates):

    point = _to_numpy(point)
    q = _to_numpy(q)

    if q.shape != (n_variates, H, len(QLEVELS)):
        raise RuntimeError(f"Chronos quantile shape {q.shape}, expected {(n_variates, H, len(QLEVELS))}")
    if not np.all(np.isfinite(point)):
        raise RuntimeError("Chronos returned non-finite point forecasts")
    if not np.all(np.isfinite(q)):
        raise RuntimeError("Chronos returned non-finite quantiles")

    return point.astype(np.float64, copy=False), q.astype(np.float64, copy=False)


def predict_f_multivariate(pipeline, W, observed, scored, t_star):

    # cond F: all channels share one origin and are one native multivariate target
    T, N = W.shape
    scored_ch = [c for c in range(N) if scored[:, c].any()]
    if not scored_ch:
        return None

    cut = int(t_star)
    # every scored cell must be at or after the shared origin
    if scored[:cut, :].any():
        raise RuntimeError("F has scored cells before t_star, not a shared-origin forecast")

    last = max(int(np.flatnonzero(scored[:, c])[-1]) for c in scored_ch)
    H = last - cut+1

    # all channels are variates
    target = np.stack([masked_values(W, observed, np.arange(cut), c) for c in range(N)],axis=0)  # [N, cut]

    q_list, p_list = pipeline.predict_quantiles([target],prediction_length=H,quantile_levels=QLEVELS.tolist(),)

    if len(q_list) != 1 or len(p_list) != 1:
        raise RuntimeError("Chronos returned unexpected number of F tasks")

    point, q = _validate_output(p_list[0], q_list[0], H, N)

    ps, qs = [], []

    for c in range(N):
        ts = np.flatnonzero(scored[:, c])

        if ts.size == 0:
            continue
        
        sel = ts - cut
        ps.append(point[c, sel])
        qs.append(q[c, sel, :].T)  # [Q, Kc]

    return np.concatenate(ps), np.concatenate(qs, axis=1)


def predict_one_channel_with_covariates(pipeline, W, observed, scored, c):
    # native univariate target plus covariates
    T, N = W.shape
    ts = np.flatnonzero(scored[:, c])
    if ts.size == 0:
        return None

    cut = int(ts[0])
    last = int(ts[-1])
    H = last - cut + 1
    if cut <= 0:
        raise RuntimeError(f"empty context for ch{c} (cut={cut})")

    task = {"target": masked_values(W, observed, np.arange(cut), c)}

    past_covariates = {}
    future_covariates = {}
    fut_rows = np.arange(cut, last + 1)

    for j in range(N):
        if j == c:
            continue

        past = masked_values(W, observed, np.arange(cut), j)

        if np.isfinite(past).any():
            key = f"ch{j}"
            past_covariates[key] = past

            future = masked_values(W, observed, fut_rows, j)
            
            # known future only if at least one value is known
            if np.isfinite(future).any():
                future_covariates[key] = future

    if past_covariates:
        task["past_covariates"] = past_covariates
        # past and future
    if future_covariates:
        task["future_covariates"] = future_covariates

    q_list, p_list = pipeline.predict_quantiles([task], prediction_length=H,quantile_levels=QLEVELS.tolist())

    if len(q_list) != 1 or len(p_list) != 1:
        raise RuntimeError("Chronos returned unexpected number of tasks")

    point, q = _validate_output(p_list[0], q_list[0], H, 1)

    sel = ts - cut
    return point[0, sel], q[0, sel, :].T


def predict_a_or_c(pipeline, W, observed, scored):
    # channel-major, as benchmark.prep.gather
    ps, qs = [], []

    for c in range(W.shape[1]):

        if not scored[:, c].any():
            continue
        
        out = predict_one_channel_with_covariates(pipeline, W, observed, scored, c)

        if out is not None:
            ps.append(out[0]); qs.append(out[1])

    if not ps:
        return None
    
    return np.concatenate(ps), np.concatenate(qs, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="tabpfn_io")
    p.add_argument("--model", default="amazon/chronos-2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="chronos2")
    p.add_argument("--conditions", default="A,F,C,DRH")
    args = p.parse_args()

    conds = {x.strip() for x in args.conditions.split(",") if x.strip()}

    print(f'[chronos2] model={args.model} device={args.device} conditions={sorted(conds)}', flush=True)

    pipeline = Chronos2Pipeline.from_pretrained(args.model,device_map=args.device)

    files = sorted(glob.glob(os.path.join(args.in_dir, "prep_*.npz")))

    if not files:
        raise SystemExit(f"no prep_*.npz in {args.in_dir}")

    for path in files:
        d = np.load(path, allow_pickle=True)
        Wg = d["Wg"]
        Obs = d["observed"]
        Sc = d["scored"]
        Cond = d["cond"]
        # shared forecast origin, written by benchmark.prep
        t_star = int(np.asarray(d["t_star"]).reshape(-1)[0]) if "t_star" in d.files else None
        n_all = len(Wg)

        mus, vrs, qs, los, his, status = [], [], [], [], [], []
        t0 = time.time()
        n_ok = 0

        for i in range(n_all):
            W = np.asarray(Wg[i], dtype=float)
            observed = np.asarray(Obs[i], dtype=bool)
            scored = np.asarray(Sc[i], dtype=bool)
            cond = str(Cond[i])
            K = int(scored.sum())

            nan = np.full(K, np.nan, dtype=np.float32)
            qnan = np.full((len(QLEVELS), K), np.nan, dtype=np.float32)

            def blank(tag):
                mus.append(nan.copy())
                vrs.append(nan.copy())
                qs.append(qnan.copy())
                los.append(nan.copy())
                his.append(nan.copy())
                status.append(tag)

            if cond not in conds:
                blank("cond_skipped")
                continue
            if cond in INTERIOR:
                blank("unsupported_interior_imputation")
                continue

            try:
                if cond == "F":
                    out = predict_f_multivariate(pipeline, W, observed, scored, t_star)
                    ok_tag = "ok_F_multivariate"
                elif cond in {"A", "C"}:
                    out = predict_a_or_c(pipeline, W, observed, scored)
                    ok_tag = (
                        "ok_A_future_covariates"
                        if cond == "A"
                        else "ok_C_ragged_covariates"
                    )
                else:
                    blank(f"unsupported_condition:{cond}")
                    continue

                if out is None:
                    blank("no_prediction")
                    continue

                mu, qv = out
                mu = np.asarray(mu, dtype=np.float64)
                qv = np.asarray(qv, dtype=np.float64)

                mus.append(mu.astype(np.float32))
                vrs.append(nan.copy())
                qs.append(qv.astype(np.float32))
                los.append(qv[0].astype(np.float32))
                his.append(qv[-1].astype(np.float32))
                status.append(ok_tag)
                n_ok += 1

            except Exception as e:
                blank(f"error:{type(e).__name__}:{e}"[:300])

            if n_ok and n_ok % 25 == 0:
                elapsed = time.time() - t0
                print(f'  {os.path.basename(path)} ok={n_ok} elapsed={elapsed:.0f}s', flush=True)

        tg = args.tag
        out_path = path.replace("prep_", f"{tg}_")
        np.savez_compressed(out_path, **{f"mu_{tg}": obj(mus), f"var_{tg}": obj(vrs), f"q_{tg}": obj(qs), f"qlevels_{tg}": QLEVELS, f"lo_{tg}": obj(los), f"hi_{tg}": obj(his), f"model_{tg}": np.array(args.model), "status": np.asarray(status)})
        keys, counts = np.unique([str(s).split(":")[0] for s in status], return_counts=True)
        print(f'{out_path}: {dict(zip(keys.tolist(), counts.tolist()))}', flush=True)


if __name__ == "__main__":
    main()
