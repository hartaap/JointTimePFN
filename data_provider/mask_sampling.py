import os
import numpy as np


PATTERNS = ('forecast', 'mcar', 'block_channel', 'per_channel_rate', 'mixed', 'forecast_partial', 'forecast_original', 'covariate_forecast', 'condA', 'crazy', 'global_blackout')


# Pretraining mask mixture (paper Table 4), in the order of PATTERNS.
WEIGHT_PRESETS = {
    'foundation_mix': ( # the one used in the final model
        0.00,  # forecast (superseded by the patterns below)
        0.10,  # mcar: uniform interior missingness
        0.15,  # block_channel: per-channel block outage
        0.00,  # per_channel_rate (superseded by crazy)
        0.10,  # mixed: random combinations within one window
        0.10,  # forecast_partial: per-channel origin, random horizon
        0.10,  # forecast_original: shared-origin forecast
        0.15,  # covariate_forecast: future partially known on a subset of channels
        0.05,  # condA: one target future hidden
        0.15,  # crazy: heterogeneous sparse, strongly uneven per-channel rates
        0.10,  # global_blackout: all channels unobserved over a shared interval
    ),
}

_preset = os.environ.get('PFN_MASK_PRESET', 'foundation_mix')
if _preset not in WEIGHT_PRESETS:
    raise ValueError(f"PFN_MASK_PRESET={_preset!r} not in {list(WEIGHT_PRESETS)}")
DEFAULT_WEIGHTS = WEIGHT_PRESETS[_preset]
assert abs(sum(DEFAULT_WEIGHTS) - 1.0) < 1e-8, f"preset {_preset} weights must sum to 1"
print(f"[mask] preset={_preset}  DEFAULT_WEIGHTS={DEFAULT_WEIGHTS}")



def sample_mask(T, N, rng, forced_pattern=None):
    # samples a [T, N] mask (1 = queried, 0 = observed) from the pretraining mixture
    if forced_pattern is None:
        pattern = rng.choice(PATTERNS, p=DEFAULT_WEIGHTS)
    else:
        if forced_pattern not in PATTERNS:
            raise ValueError(f"forced_pattern {forced_pattern!r} not in {PATTERNS}")
        pattern = forced_pattern
    M = np.zeros((T, N), dtype=np.float32)

    if pattern == 'forecast':
        t_star = int(rng.integers(T // 2, T))
        M[t_star:, :] = 1.0

    elif pattern == 'condA':
        t_star = T//2
        target = int(rng.integers(0,N))
        M[t_star:,target] = 1.0

    elif pattern == 'mcar': # D
        rate = float(rng.uniform(0.1, 0.5))
        M = (rng.random((T, N)) < rate).astype(np.float32)

    elif pattern == 'crazy':
        for i in range(N):
            rate = float(rng.uniform(0.1, 0.9))
            M[:,i] = (rng.random(T) < rate).astype(np.float32)

    elif pattern == 'block_channel':
        n_blocked = int(rng.integers(1, max(2, N // 2 + 1)))
        for c in rng.choice(N, size=n_blocked, replace=False):
            min_len = max(2, T // 20)
            max_len = max(min_len + 1, T // 2)
            start = int(rng.integers(0, T - min_len))
            length = int(rng.integers(min_len, min(max_len, T - start) + 1))
            M[start:start + length, c] = 1.0

    elif pattern == 'per_channel_rate':
        for c in range(N):
            rate = float(rng.uniform(0.0, 0.7))
            M[:, c] = (rng.random(T) < rate).astype(np.float32)

    elif pattern == 'mixed':
        M1, _ = sample_mask(T, N, rng)
        M2, _ = sample_mask(T, N, rng)
        M = np.maximum(M1, M2)

    elif pattern == 'forecast_original':
        t_star = T//2
        M[t_star:, :] = 1.0

    elif pattern == 'global_blackout':
        min_len = max(2, T // 10)
        max_len = max(min_len + 1, T // 3)
        start = int(rng.integers(T // 5, T - max_len - T // 5)) # data left at both ends
        length = int(rng.integers(min_len, max_len))
        M[start:start + length, :] = 1.0

    elif pattern == 'forecast_partial':
        n_targets = int(rng.integers(1, N + 1))  # 1..N
        targets = rng.choice(N, size=n_targets, replace=False)
        for c in targets:
            t_star_c = int(rng.integers(T//2, T))
            M[t_star_c:, c] = 1.0

    elif pattern == 'covariate_forecast':
        max_t = max(1, N // 4)
        n_targets = 1 + int(rng.beta(0.5, 2.5) * max_t)
        n_targets = min(n_targets, N - 1)  # keep >=1 fully-observed covariate
        targets = rng.choice(N, size=n_targets, replace=False)
        for c in targets:
            t_star_c = int(rng.integers(T // 2, T))  # per-target cutoff
            M[t_star_c:, c] = 1.0

    # keep at least one observed cell
    if M.sum() >= T * N:
        idx = int(rng.integers(0, T * N))
        M.reshape(-1)[idx] = 0.0

    return M, pattern
