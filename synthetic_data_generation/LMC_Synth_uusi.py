# Builds on the kernel-synth method of Chronos (https://github.com/amazon-science/chronos-forecasting); modified.
# "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: Apache-2.0"
# Randomized-kernel variant of LMC_Synth.py provided by TimePFN in https://github.com/egetaga/TimePFN.
# Changes: continuous hyperparameters from PRIOR_BANDS instead of a fixed KERNEL_BANK, Matern and changepoint kernels, signed loadings, a hierarchical latent count, and (-NA) noise after mixing.

import argparse
import functools
from typing import Optional

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)
from tqdm.auto import tqdm


class ChangepointKernel(Kernel):
    # k = σ(x) k1 σ(x') + (1−σ(x)) k2 (1−σ(x')) + base_variance, σ a sigmoid at the changepoint

    def __init__(self, kernel1, kernel2, changepoint=0.5, steepness=20.0, base_variance=0.1):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.changepoint = changepoint
        self.steepness = steepness
        self.base_variance = base_variance

    def get_params(self, deep=True):
        params = dict(kernel1=self.kernel1, kernel2=self.kernel2,
                      changepoint=self.changepoint, steepness=self.steepness,
                      base_variance=self.base_variance)
        if deep:
            params.update({"kernel1__" + k: v for k, v in self.kernel1.get_params(deep).items()})
            params.update({"kernel2__" + k: v for k, v in self.kernel2.get_params(deep).items()})
        return params

    def _sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-self.steepness * (X[:, 0] - self.changepoint)))

    def __call__(self, X, Y=None, eval_gradient=False):
        if X.ndim == 1:
            X = X[:, None]
        if eval_gradient:
            K1, _ = self.kernel1(X, eval_gradient=True)
            K2, _ = self.kernel2(X, eval_gradient=True)
            sx = self._sigmoid(X)
            w1 = np.outer(sx, sx)
            w2 = np.outer(1.0 - sx, 1.0 - sx)
            return w1 * K1 + w2 * K2 + self.base_variance, np.zeros((X.shape[0], X.shape[0], 0))

        if Y is None:
            Y = X
        elif Y.ndim == 1:
            Y = Y[:, None]

        sx = self._sigmoid(X)
        sy = self._sigmoid(Y)
        w1 = np.outer(sx, sy)
        w2 = np.outer(1.0 - sx, 1.0 - sy)

        cp_cov = w1 * self.kernel1(X, Y) + w2 * self.kernel2(X, Y)
        return cp_cov + self.base_variance

    def diag(self, X):
        if X.ndim == 1:
            X = X[:, None]
        sx = self._sigmoid(X)

        diag_cov = sx ** 2 * self.kernel1.diag(X) + (1.0 - sx) ** 2 * self.kernel2.diag(X)
        return diag_cov + self.base_variance

    def is_stationary(self):
        return False

    def __repr__(self):
        return (f"ChangepointKernel({self.kernel1}, {self.kernel2}, "
                f"cp={self.changepoint:.2f}, s={self.steepness}, base_var={self.base_variance})")


LENGTH = 1024


def _loguniform(lo, hi):
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))


# hyperparameter ranges: 'imd' is the pretraining prior; 'oomd' moves every range outside it in one
# direction (shorter, rougher, noisier, sharper) and drops the canonical calendar periods
PRIOR_BANDS = {
    'imd': dict(ls=(0.05, 3.0), es_ls=(0.005, 1.0), es_period=(0.05, 1.0),
                use_canonical=True, noise=(0.01, 0.5),
                cp_steep=(5.0, 50.0), cp_basevar=(0.01, 0.5)),
    # about 10% of oomd slots have no effect on a 192-step window
    'oomd': dict(ls=(0.005, 0.04), es_ls=(0.0005, 0.004), es_period=(0.005, 0.04),
                 use_canonical=False, noise=(0.6, 1.5),
                 cp_steep=(60.0, 500.0), cp_basevar=(0.6, 2.0)),
}

CANONICAL_PERIODS = [8.0, 12.0, 24.0, 48.0, 168.0]   # hours: shift / ultradian / circadian / weekly


def _bands(bands):
    # Resolve a band: None, a PRIOR_BANDS key, or a dict
    if bands is None:
        return PRIOR_BANDS['imd']

    if isinstance(bands, str):
        return PRIOR_BANDS[bands]

    return bands


def sample_random_kernel(allow_cp=True, noise_after=False, bands=None):
    # One base kernel with hyperparameters drawn from the band
    band = _bands(bands)
    ls_lo, ls_hi = band['ls']
    noise_lo, noise_hi = band['noise']
    # 0 RBF, 1 Matern, 2 ExpSineSquared, 3 White, 4 Changepoint, 5 DotProduct, 6 RationalQuadratic.
    # with noise_after the WhiteKernel is left out
    if noise_after:
        t = int(np.random.choice([0, 1, 2, 5, 6] + ([4] if allow_cp else [])))
    else:
        t = int(np.random.choice([0, 1, 2, 3, 5, 6] + ([4] if allow_cp else [])))

    if t == 0:
        return RBF(length_scale=_loguniform(ls_lo, ls_hi))
    if t == 1:
        return Matern(length_scale=_loguniform(ls_lo, ls_hi),
                      nu=float(np.random.choice([0.5, 1.5, 2.5])))
    if t == 2:
        # half of the periods from the canonical set, half continuous
        if band['use_canonical'] and np.random.rand() < 0.5:
            period = np.random.choice(CANONICAL_PERIODS) / LENGTH
        else:
            period = _loguniform(*band['es_period'])
        return ExpSineSquared(
            length_scale=_loguniform(*band['es_ls']),
            periodicity=period
        )
    if t == 3:
        return WhiteKernel(noise_level=_loguniform(noise_lo, noise_hi))
    if t == 5:
        return DotProduct(sigma_0=_loguniform(0.1, 1.0))
    if t == 6:
        return RationalQuadratic(
            length_scale=_loguniform(ls_lo, ls_hi),
            alpha=_loguniform(0.1, 10.0)
        )
    sampled_base_var = _loguniform(*band['cp_basevar'])

    return ChangepointKernel(
        sample_random_kernel(allow_cp=False, noise_after=noise_after, bands=bands),
        sample_random_kernel(allow_cp=False, noise_after=noise_after, bands=bands),
        changepoint=float(np.random.uniform(0.3, 0.7)),
        steepness=_loguniform(*band['cp_steep']),
        base_variance=sampled_base_var
    )


def random_binary_map(a: Kernel, b: Kernel):
    # a + b or a * b with equal prob.
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)


def _compose_random_kernel(num_kernels, noise_after=False, bands=None):
    return functools.reduce(random_binary_map,
                            [sample_random_kernel(noise_after=noise_after, bands=bands)
                             for _ in range(num_kernels)])


def sample_from_gp_prior_efficient(kernel: Kernel, X: np.ndarray,
                                   random_seed: Optional[int] = None, method: str = "eigh"):
    if X.ndim == 1:
        X = X[:, None]
    assert X.ndim == 2
    cov = kernel(X)
    return np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=cov, method=method)


def sample_latent_num(num_channels, fixed_rank=0):
    if fixed_rank:
        return int(min(fixed_rank, num_channels))

    # upper limit grows with the channel count
    max_scale = max(10.0, num_channels / 4.0)
    shape = np.random.uniform(1.5, 2.5)
    scale = np.exp(np.random.uniform(np.log(2.0), np.log(max_scale)))
    latent_num = np.rint(np.random.weibull(shape) * scale + 1)
    return int(np.clip(latent_num, 2, num_channels))


def generate_time_series(dirichlet_min, dirichlet_max, max_kernels: int = 5,
                         num_channels: int = 10, private_frac: float = 0.0,
                         noise_after: bool = False):
    while True:
        X = np.linspace(0, 1, LENGTH)

        latent_num = sample_latent_num(num_channels)
        kernel_numbers = np.random.randint(1, max_kernels + 1, size=latent_num)
        latent_kernels = [_compose_random_kernel(nk, noise_after=noise_after)
                          for nk in kernel_numbers]

        try:
            latent_functions = np.array(
                [sample_from_gp_prior_efficient(kernel=k, X=X) for k in latent_kernels])
            dirichlet = np.random.uniform(dirichlet_min, dirichlet_max)
            # signed loadings: a unit-sphere direction scaled by a Dirichlet norm
            raw_weights = np.random.normal(loc=0.0, scale=1.0, size=(num_channels, latent_num))
            directions = raw_weights / np.linalg.norm(raw_weights, ord=2, axis=1, keepdims=True)
            amp = np.random.dirichlet(dirichlet * np.ones(latent_num), size=num_channels)
            weights = directions * np.linalg.norm(amp, ord=2, axis=1, keepdims=True)

            ts = np.dot(weights, latent_functions)
            # private channels: replaced by an independent latent
            n_private = int(round(private_frac * num_channels))
            if n_private > 0:
                for ci in np.random.choice(num_channels, size=n_private, replace=False):
                    pk = _compose_random_kernel(int(np.random.randint(1, max_kernels + 1)), noise_after=noise_after)

                    ts[ci] = sample_from_gp_prior_efficient(kernel=pk, X=X)

            if noise_after:
                # one noise std per channel, log-uniform on the imd band
                nlo, nhi = PRIOR_BANDS['imd']['noise']
                for c in range(num_channels):
                    ts[c] = ts[c] + np.random.normal(0.0, _loguniform(nlo, nhi),
                                                     size=ts.shape[1])
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue

        return {"start": np.datetime64("2000-01-01 00:00", "s"), "target": ts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-series", type=int, default=15000)
    parser.add_argument("-J", "--max-kernels", type=int, default=5)
    parser.add_argument("-C", "--num-channels", type=int, default=160)
    parser.add_argument("-O", "--output", type=str, default="LMC_synth_randomized_MTS.arrow")
    parser.add_argument("-D", "--directory", type=str, default="./")
    parser.add_argument("-L", "--length", type=int, default=1024)
    parser.add_argument("-M", "--dirichlet_min", type=float)
    parser.add_argument("-X", "--dirichlet_max", type=float)
    parser.add_argument("-P", "--private_frac", type=float, default=0.0,
                        help="fraction of channels made independent (own private latent) per series")
    parser.add_argument("-NA", "--noise_after", action="store_true",
                        help="drop the White kernel from the bank and add i.i.d. observation noise "
                             "per channel after mixing")

    args = parser.parse_args()
    path = args.directory + args.output

    LENGTH = args.length

    generated_dataset = Parallel(n_jobs=-1)(
        delayed(generate_time_series)(
            max_kernels=args.max_kernels, num_channels=args.num_channels,
            dirichlet_min=args.dirichlet_min, dirichlet_max=args.dirichlet_max,
            private_frac=args.private_frac, noise_after=args.noise_after)
        for _ in tqdm(range(args.num_series))
    )

    ArrowWriter(compression="lz4").write_to_file(generated_dataset, path=path)
