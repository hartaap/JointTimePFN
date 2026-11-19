#!/usr/bin/env python3
import argparse, glob, os, time
import numpy as np
import pandas as pd
from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode
from tabpfn_time_series.features import RunningIndexFeature, AutoSeasonalFeature

Q = np.array([0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975])


# not a forecasting task:
INTERIOR = {"DRH"}


class NotHorizon(Exception):
    pass


def obj(xs):
    a = np.empty(len(xs), dtype=object)
    a[:] = xs
    return a

def qcols(df):
    lut = {}
    for c in df.columns:
        s = str(c)
        for p in ("quantile_", "q_", "q"):
            if s.startswith(p):
                s = s[len(p):]
                break
        try:
            lut[round(float(s), 6)] = c
        except ValueError:
            pass
    return [lut[round(float(q), 6)] for q in Q]

def forecast_channel(pipe, W, observed, scored, c, stamps):
    test = np.flatnonzero(scored[:, c])
    train = np.flatnonzero(observed[:, c])

    if len(test) == 0:
        return None
    # dense horizon, then select
    if train.max(initial=-1) >= test.min():
        raise NotHorizon("interior query: not native TabPFN-TS forecasting")

    dense = np.arange(int(test.min()), int(test.max()) + 1)
    if observed[dense, c].any():
        raise NotHorizon("target is observed inside its own horizon: interior imputation")

    ctx = {"timestamp": stamps[train], "target": W[train, c]}
    fut = {"timestamp": stamps[dense]}

    # native known-future covariates
    for j in range(W.shape[1]):
        if j == c or not observed[dense, j].any():
            continue

        x_ctx = np.full(len(train), np.nan)
        m_ctx = observed[train, j]
        x_ctx[m_ctx] = W[train[m_ctx], j]

        x_fut = np.full(len(dense), np.nan)
        m_fut = observed[dense, j]
        x_fut[m_fut] = W[dense[m_fut], j]

        ctx[f"ch{j}"] = x_ctx
        fut[f"ch{j}"] = x_fut

    out = pipe.predict_df(pd.DataFrame(ctx), future_df=pd.DataFrame(fut), quantiles=Q.tolist())

    if "timestamp" not in out.columns:
        out = out.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.set_index("timestamp").loc[pd.DatetimeIndex(stamps[test])]

    point = out["target"].to_numpy(float)
    quant = np.stack([out[x].to_numpy(float) for x in qcols(out)], axis=0)
    return point, quant

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="tabpfn_io")
    p.add_argument("--freq", default="h")  # MIMIC: --freq 3h in our version
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--tag", default="tabpfnts")
    p.add_argument("--conditions", default="A,F,C,DRH")
    p.add_argument("--point", choices=["mean", "median", "mode"], default="mean")
    p.add_argument("--temporal_features", choices=["noncalendar", "default"], default="noncalendar", help="noncalendar: running index + auto-seasonal only (timestamps are synthetic)")
    args = p.parse_args()

    conds = set(args.conditions.split(","))

    feat_mode = args.temporal_features
    kw = dict(tabpfn_mode=TabPFNMode.LOCAL, tabpfn_output_selection=args.point)
    if feat_mode == "noncalendar":
        kw["temporal_features"] = [RunningIndexFeature(), AutoSeasonalFeature()]
    pipe = TabPFNTSPipeline(**kw)
    print(f'[tabpfn-ts] temporal_features={feat_mode}  point={args.point}  freq={args.freq}', flush=True)

    for path in sorted(glob.glob(os.path.join(args.in_dir, "prep_*.npz"))):
        d = np.load(path, allow_pickle=True)
        Wg, Obs, Sc, Cond = d["Wg"], d["observed"], d["scored"], d["cond"]

        n_all = len(Wg)
        mus, vars_, qs, los, his, status = [], [], [], [], [], []
        t_start = time.time(); n_done = 0

        for i in range(n_all):
            W = np.asarray(Wg[i], float)
            observed = np.asarray(Obs[i], bool)
            scored = np.asarray(Sc[i], bool)
            cond = str(Cond[i])
            K = int(scored.sum())
            stamps = pd.date_range(args.start, periods=W.shape[0], freq=args.freq).to_numpy()
            nan = np.full(K, np.nan, np.float32)
            qnan = np.full((len(Q), K), np.nan, np.float32)

            # if something not wanted
            if cond in INTERIOR:
                mus.append(nan); vars_.append(nan); qs.append(qnan)
                los.append(nan); his.append(nan)
                status.append("unsupported_interior_imputation")
                continue

            if cond not in conds:
                mus.append(nan); vars_.append(nan); qs.append(qnan)
                los.append(nan); his.append(nan); status.append("cond_skipped")
                continue

            try:
                ps, qq = [], []
                for c in range(W.shape[1]):
                    if not scored[:, c].any():
                        continue
                    r = forecast_channel(pipe, W, observed, scored, c, stamps)
                    if r is not None:
                        ps.append(r[0])
                        qq.append(r[1])

                mu = np.concatenate(ps).astype(np.float32)
                qv = np.concatenate(qq, axis=1).astype(np.float32)
                if mu.shape != (K,) or qv.shape != (len(Q), K):
                    raise ValueError(f"shape mismatch: {mu.shape}, {qv.shape}, K={K}")

                mus.append(mu)
                vars_.append(nan) 
                qs.append(qv)
                los.append(qv[0])
                his.append(qv[-1])
                status.append("ok")
                n_done += 1
                if n_done % 10 == 0:
                    el = time.time() - t_start
                    print(f'  {os.path.basename(path)} ok={n_done} elapsed={el:.0f}s ({el / n_done:.1f}s/group)', flush=True)

            except NotHorizon as e:
                # imputation is different
                mus.append(nan); vars_.append(nan); qs.append(qnan)
                los.append(nan); his.append(nan)
                status.append(f"not_horizon_shaped:{e}"[:120])
            except Exception as e:
                mus.append(nan); vars_.append(nan); qs.append(qnan)
                los.append(nan); his.append(nan)
                status.append(f"error:{type(e).__name__}:{e}"[:300])

        tag = args.tag
        out = path.replace("prep_", f"{tag}_")
        np.savez_compressed(out, **{f'mu_{tag}': obj(mus), f'var_{tag}': obj(vars_), f'q_{tag}': obj(qs), f'qlevels_{tag}': Q, f'lo_{tag}': obj(los), f'hi_{tag}': obj(his), f'temporal_features_{tag}': np.array(str(feat_mode)), 'status': np.asarray(status)})
        print(out, dict(zip(*np.unique(status, return_counts=True))))

if __name__ == "__main__":
    main()