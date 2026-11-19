#!/usr/bin/env python3
# Differentiable torch evaluator of the generator's composed sklearn kernels.

# Reproduces sklearn's k(X): a WhiteKernel adds noise on the diagonal there but is zero in k(X, Y).

import math

import numpy as np
import torch

_SQ3, _SQ5 = math.sqrt(3.0), math.sqrt(5.0)

# each slot carries the transform from sampler space back to the parameter
_LOG, _UNIT = 'log', 'unit'


class Spec:
    # One kernel node: (kind, children, fixed, slots)
    __slots__ = ('kind', 'children', 'fixed', 'slots')

    def __init__(self, kind, children=(), fixed=None, slots=()):
        self.kind, self.children = kind, tuple(children)
        self.fixed, self.slots = dict(fixed or {}), list(slots)

    def n_params(self):
        return len(self.slots) + sum(c.n_params() for c in self.children)

    def true_vector(self):
        # The generator's hyperparameters in sampler space.
        out = []
        for _, tr, v in self.slots:
            out.append(math.log(v) if tr == _LOG else math.log(v / (1.0 - v)))
        for c in self.children:
            out.extend(c.true_vector())
        return out


def spec_from_sklearn(k):
    cn = type(k).__name__
    if cn == 'Sum':
        return Spec('sum', [spec_from_sklearn(k.k1), spec_from_sklearn(k.k2)])
    if cn == 'Product':
        return Spec('prod', [spec_from_sklearn(k.k1), spec_from_sklearn(k.k2)])
    if cn == 'RBF':
        return Spec('rbf', slots=[('ls', _LOG, float(np.ravel(k.length_scale)[0]))])
    if cn == 'Matern':
        return Spec('matern', fixed={'nu': float(k.nu)}, slots=[('ls', _LOG, float(np.ravel(k.length_scale)[0]))])
    if cn == 'ExpSineSquared':
        return Spec('expsine', slots=[('ls', _LOG, float(k.length_scale)), ('period', _LOG, float(k.periodicity))])
    if cn == 'WhiteKernel':
        return Spec('white', slots=[('noise', _LOG, float(k.noise_level))])
    if cn == 'ConstantKernel':
        return Spec('const', slots=[('c', _LOG, float(k.constant_value))])
    if cn == 'DotProduct':
        return Spec('dot', slots=[('sigma_0', _LOG, float(k.sigma_0))])
    if cn == 'RationalQuadratic':
        return Spec('rq', slots=[('ls', _LOG, float(k.length_scale)), ('alpha', _LOG, float(k.alpha))])

    if cn == 'ChangepointKernel':
        return Spec('changepoint',
                    [spec_from_sklearn(k.kernel1), spec_from_sklearn(k.kernel2)],
                    slots=[('cp', _UNIT, float(k.changepoint)),
                           ('steep', _LOG, float(k.steepness)),
                           ('base_var', _LOG, float(k.base_variance))])
    raise ValueError(f'kernel {cn} is not in the generator bank!')


def active_mask(spec, x):
    # Slots the Gram matrix depends on, measured by autograd
    seed = 0
    tol = 1e-10
    n_probe = 6
    g = torch.Generator().manual_seed(seed)
    tv = torch.tensor(spec.true_vector(), dtype=torch.float64)
    seen = torch.zeros(spec.n_params(), dtype=torch.bool)

    for j in range(n_probe):
        # probes on the CPU generator, independent of the default device
        th = tv.clone() if j == 0 else tv + 0.35 * torch.randn(
            tv.shape, generator=g, dtype=torch.float64, device='cpu').to(tv.device)
        th.requires_grad_(True)
        K = composed_gram(spec, th, x)
        # sum |K|, not sum K
        K.abs().sum().backward()
        seen |= th.grad.abs() > tol
    return seen.tolist()


def _take(spec, theta, i):
    # (values dict, next index) for this node's slots
    vals = {}

    for name, tr, _ in spec.slots:
        v = theta[i]
        vals[name] = torch.exp(v) if tr == _LOG else torch.sigmoid(v)
        i += 1

    return vals, i


def _gram(spec, theta, i, xa, xb, y_is_none):
    p, i = _take(spec, theta, i)
    kind = spec.kind

    if kind in ('sum', 'prod'):
        a, i = _gram(spec.children[0], theta, i, xa, xb, y_is_none)
        b, i = _gram(spec.children[1], theta, i, xa, xb, y_is_none)
        return (a + b if kind == 'sum' else a * b), i

    if kind == 'changepoint':
        # sigmoid gates on the raw coordinate, as ChangepointKernel._sigmoid
        sa = torch.sigmoid(p['steep'] * (xa - p['cp'])).unsqueeze(-1)
        sb = torch.sigmoid(p['steep'] * (xb - p['cp'])).unsqueeze(-2)

        # sub-kernels as k(X, Y), as sklearn does
        k1, i = _gram(spec.children[0], theta, i, xa, xb, False)
        k2, i = _gram(spec.children[1], theta, i, xa, xb, False)

        return sa * sb * k1 + (1.0 - sa) * (1.0 - sb) * k2 + p['base_var'], i

    if kind == 'dot':
        # sklearn: k(x, y) = sigma_0^2 + x y
        return p['sigma_0'] ** 2 + xa.unsqueeze(-1) * xb.unsqueeze(-2), i

    d = (xa.unsqueeze(-1) - xb.unsqueeze(-2)).abs()

    if kind == 'rq':
        # sklearn: k = (1 + d^2 / (2 * alpha * ls^2)) ** (-alpha)
        return (1.0 + d ** 2 / (2.0 * p['alpha'] * p['ls'] ** 2)) ** (-p['alpha']), i
    if kind == 'rbf':
        return torch.exp(-0.5 * (d / p['ls']) ** 2), i
    if kind == 'expsine':
        return torch.exp(-2.0 * torch.sin(math.pi * d / p['period']) ** 2 / p['ls'] ** 2), i
    if kind == 'const':
        return p['c'] * torch.ones_like(d), i
    if kind == 'white':
        if not y_is_none:
            return torch.zeros_like(d), i  # sklearn: k(X, Y) is identically zero
        # eye() by position, as np.eye(n)
        n = d.shape[-1]
        eye = torch.eye(n, dtype=d.dtype)
        return p['noise'] * eye, i

    nu = spec.fixed['nu']  # for matern
    r = d / p['ls']

    if abs(nu - 0.5) < 1e-8:
        return torch.exp(-r), i

    if abs(nu - 2.5) < 1e-8:
        a = _SQ5 * r
        return (1.0 + a + a ** 2 / 3.0) * torch.exp(-a), i
    a = _SQ3 * r

    return (1.0 + a) * torch.exp(-a), i


def composed_gram(spec, theta, xa):
    # k(xa, xa) for one composed latent, reproducing sklearn's k(X)
    xb = xa

    if not torch.is_tensor(theta):
        theta = torch.as_tensor(theta, dtype=xa.dtype)

    K, used = _gram(spec, theta, 0, xa, xb, True)
    assert used == spec.n_params(), f'consumed {used} of {spec.n_params()} parameters'

    return K
