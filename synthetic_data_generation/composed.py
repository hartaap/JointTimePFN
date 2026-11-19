#!/usr/bin/env python3
# Fidelity datasets drawn from the pretraining prior's own kernel sampler.

#    python -m synthetic_data_generation.composed --num_channels 4 --Q 2 --prior_mode imd ...

import argparse
import os
import pickle

import numpy as np

from synthetic_data_generation.LMC_Synth_uusi import (
    _compose_random_kernel, sample_latent_num, sample_from_gp_prior_efficient, _loguniform,
    PRIOR_BANDS,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_datasets', type=int, default=10)
    p.add_argument('--num_channels', type=int, default=15)
    p.add_argument('--max_kernels', type=int, default=5)
    p.add_argument('--length', type=int, default=1024,
                   help='= LMC_Synth_uusi.LENGTH')
    p.add_argument('--dirichlet_min', type=float, default=0.1)
    p.add_argument('--dirichlet_max', type=float, default=10.0,
                   help='as in pretraining')
    p.add_argument('--private_frac', type=float, default=0.0,
                   help='0 for the fidelity suite (pretraining: 0.2)')
    p.add_argument('--noise_after', action='store_true', default=True)
    p.add_argument('--Q', type=int, default=2,
                   help='fixed latent count (0 = hierarchical prior)')
    p.add_argument('--prior_mode', type=str, default='imd', choices=sorted(PRIOR_BANDS),
                   help='PRIOR_BANDS key')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_root', type=str, default='dataset')
    p.add_argument('--tag', type=str, default='')
    a = p.parse_args()

    LENGTH = a.length
    X = np.linspace(0.0, 1.0, LENGTH)

    for di in range(a.n_datasets):
        np.random.seed(a.seed * 10_000 + di)  # LMC_Synth_uusi uses the global RNG

        Q = sample_latent_num(a.num_channels, a.Q)
        nk = np.random.randint(1, a.max_kernels + 1, size=Q)
        kernels = [_compose_random_kernel(int(n), noise_after=a.noise_after,
                                          bands=a.prior_mode) for n in nk]
        latents = np.array([sample_from_gp_prior_efficient(kernel=k, X=X) for k in kernels])

        dirichlet = np.random.uniform(a.dirichlet_min, a.dirichlet_max)
        # loading law as LMC_Synth_uusi.generate_time_series
        raw = np.random.normal(loc=0.0, scale=1.0, size=(a.num_channels, Q))
        directions = raw / np.linalg.norm(raw, ord=2, axis=1, keepdims=True)
        amp = np.random.dirichlet(dirichlet * np.ones(Q), size=a.num_channels)
        W = directions * np.linalg.norm(amp, ord=2, axis=1, keepdims=True)

        ts = W @ latents  # [N, LENGTH]

        # a private channel becomes an extra latent with a one-hot loading
        n_private = int(round(a.private_frac * a.num_channels))
        priv_idx = (np.random.choice(a.num_channels, size=n_private, replace=False)
                    if n_private else np.zeros(0, int))

        for ci in priv_idx:
            pk = _compose_random_kernel(int(np.random.randint(1, a.max_kernels + 1)),
                                        noise_after=a.noise_after, bands=a.prior_mode)
            ts[ci] = sample_from_gp_prior_efficient(kernel=pk, X=X)
            kernels.append(pk)
            col = np.zeros((a.num_channels, 1)); col[ci, 0] = 1.0
            W[ci, :] = 0.0
            W = np.concatenate([W, col], axis=1)

        # one noise std per dataset, from the band's noise range
        noise_lo, noise_hi = PRIOR_BANDS[a.prior_mode]['noise']
        noise_std = _loguniform(noise_lo, noise_hi) if a.noise_after else 0.0
        Y = (ts + noise_std * np.random.standard_normal(ts.shape)).T  # [LENGTH, N]

        # the realised Q (clipped to N) goes in the tag, except at 2
        q_eff = int(min(a.Q, a.num_channels)) if a.Q else 0
        tag = a.tag or (f'k{a.max_kernels}_{a.prior_mode}' + (f'_Q{q_eff}' if q_eff not in (0, 2) else ''))
        out = f'{a.out_root}/toy_lmc_composed_{tag}_N{a.num_channels}_d{di}'
        os.makedirs(out, exist_ok=True)
        np.savetxt(f'{out}/toy_oracle.csv', Y, delimiter=',',
                   header=','.join(f'ch{i}' for i in range(a.num_channels)), comments='')
        np.savez(f'{out}/params.npz',
                 family='lmc_composed', kernel='composed',
                 W=W, noise_var=float(noise_std ** 2), n_shared=int(Q),
                 n_latents=int(W.shape[1]), private_idx=priv_idx,
                 max_kernels=int(a.max_kernels), length=int(LENGTH),
                 prior_mode=str(a.prior_mode), Q_fixed=int(a.Q),
                 noise_band=np.array([noise_lo, noise_hi], dtype=float),
                 kernel_repr=np.array([repr(k) for k in kernels], dtype=object))
        with open(f'{out}/kernels.pkl', 'wb') as f:
            pickle.dump(kernels, f)

    print(f'wrote {a.n_datasets} datasets to {a.out_root}/toy_lmc_composed_{tag}_N{a.num_channels}_d*')


if __name__ == '__main__':
    main()
