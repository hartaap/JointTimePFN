# Single entry point for MIMIC windows, so values and missingness cannot be separated
import numpy as np


def save_mimic(path, values_nan, stay_ids, channels, window):
    # save values_nan [S, W, N], naN at missing cells (note: do not pre-fill)
    v = np.asarray(values_nan, dtype=np.float32)
    assert v.ndim == 3 and v.shape[1] == window, f'expected [S,{window},N], got {v.shape}'
    np.savez_compressed(path, values=v, stay_ids=np.asarray(stay_ids), channels=np.asarray([str(c) for c in channels]), window=int(window))


def load_mimic(path):
    # -> (values, mask, meta)
    z = np.load(path, allow_pickle=False)
    v = z['values'].astype(np.float32)
    mask = ~np.isfinite(v)
    v = np.where(mask, np.float32(0.0), v)
    meta = dict(stay_ids=z['stay_ids'], channels=[str(c) for c in z['channels']], window=int(z['window']), n_stays=v.shape[0], n_channels=v.shape[2], missing_frac=float(mask.mean()))
    return v, mask, meta
