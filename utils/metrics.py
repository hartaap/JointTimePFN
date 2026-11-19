# The scoring rules of the paper, in one place:
#   per cell        gaussian_nll, covered95, crps          (benchmark/merge.py, fidelity/marginal.py)
#   joint, real     joint_scores                          (benchmark/merge.py)
#   marginal fid.   kl_terms, cell_scores                 (fidelity/marginal.py)
#   joint fid.      gauss_nll, kl_gauss, score_joint_ppd  (fidelity/joint.py)
import math

import numpy as np
from scipy.linalg import solve_triangular


# per cell

def gaussian_nll(mu, var, y):
    return 0.5 * (np.log(var) + (y - mu) ** 2 / var + np.log(2.0 * np.pi))


def covered95(mu, var, y):
    sd = np.sqrt(np.maximum(var, 1e-8))
    return (y >= mu - 1.96 * sd) & (y <= mu + 1.96 * sd)


DECILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Z_DECILES = np.array([-1.281552, -0.841621, -0.524401, -0.253347, 0.0, 0.253347, 0.524401, 0.841621, 1.281552])


def crps(q, y):
    # pinball loss on the deciles, trapezoid over the levels
    tau = DECILES[:, None]
    loss = (tau - (y[None, :] < q).astype(float)) * (y[None, :] - q)
    w = np.zeros(len(DECILES))
    steps = np.diff(DECILES)
    w[:-1] += steps / 2.0
    w[1:] += steps / 2.0
    return 2.0 * (w[:, None] * loss).sum(axis=0)


# joint, real data (mimic)

def joint_scores(mu, var, W, y):
    # joint NLL with Sigma = diag(d) + W W^T (Woodbury), the diagonal NLL, and their difference
    ridge, d_min = 1e-06, 0.0001
    if W.ndim != 2 or W.shape[0] != y.size or not np.all(np.isfinite(W)):
        return None
    K, R = W.shape
    d = np.maximum(var - (W ** 2).sum(1), d_min) + ridge          # same d as the joint head
    r = y - mu
    B = np.eye(R) + W.T @ (W / d[:, None])

    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        return None
    
    Wu = W.T @ (r / d)
    quad = float(r @ (r / d)) - float(np.linalg.solve(B, Wu) @ Wu)
    logdet = float(np.log(d).sum()) + 2.0 * float(np.log(np.diag(L)).sum())
    nll_joint = 0.5 * (K * np.log(2 * np.pi) + logdet + quad) / K
    v = var + ridge
    nll_diag = 0.5 * (K * np.log(2 * np.pi) + float(np.log(v).sum()) + float((r ** 2 / v).sum())) / K
    return nll_joint, nll_diag, nll_diag - nll_joint


#marginal fidelity

def kl_terms(mu, var, mu_o, var_o):
    var = np.maximum(var, 1e-12)
    var_o = np.maximum(var_o, 1e-12)
    vt = 0.5 * np.log(var_o / var) + var / (2 * var_o) - 0.5
    mt = (mu - mu_o)**2 / (2 * var_o)

    return (vt+mt), vt, mt


def cell_scores(mu, var, y):
    # Squared error, cov95 and normalized squared error per cell
    var = np.maximum(var, 1e-12)
    return dict(mse=(mu-y)**2, cov=covered95(mu, var, y).astype(float), z2=(mu - y) ** 2 / var)


# joint fidelity: Gaussian NLL against the oracle

JITTER_USED = []  # relative jitter used for each NLL calculation
COND_SEEN = []    # minimum eigenvalue divided by mean diagonal of oracle covariance


def gauss_nll(y, mu, S):
    # Gaussian NLL, with extra jitter if the covariance is nearly singular
    jitter = 1e-06
    K = len(y)
    S = np.asarray(S, dtype=np.float64)
    S = (0.5) * (S + S.T)
    scale = max(float(np.mean(np.diag(S))), 1e-300)

    # Check conditioning to verify, even if cholesky succeeded
    COND_SEEN.append(float(np.linalg.eigvalsh(S).min()) / scale)
    L, used = None, 0.0
    j = jitter * scale

    for _ in range(10):
        try:
            L = np.linalg.cholesky(S + j * np.eye(K))
            used = j / scale
            break
        except np.linalg.LinAlgError:
            j *= 10.0

    if L is None:  # clip small eigenvalues
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        wmax = max(float(w.max()), 1e-300)
        floor = max(jitter * scale, 1e-10 * wmax)

        for _ in range(12):
            Sc = (V * np.clip(w, floor, None)) @ V.T
            Sc = 0.5 * (Sc + Sc.T) + floor * np.eye(K)
            try:
                L = np.linalg.cholesky(Sc)
                break
            except np.linalg.LinAlgError:
                floor *= 10.0
        if L is None:
            # NaN: the row is dropped
            JITTER_USED.append(float('inf'))
            return float('nan')
        used = float('inf')


    JITTER_USED.append(used)
    a = np.linalg.solve(L, y - mu)
    nll = float(0.5 * (a @ a) + np.log(np.diag(L)).sum() + 0.5 * K * math.log(2 * math.pi))
    if not np.isfinite(nll) or abs(nll) > 1e4 * max(K, 1):
        return float('nan')
    return nll


# joint fidelity: a joint Gaussian against the NUTS posterior predictive mixture

_JIT = 1e-9
_CHOL_REPAIRS = 0

def _tri_solve(L, B):
    return solve_triangular(L, B, lower=True)


def _chol(S, name='matrix'):
    # Cholesky with increasing jitter
    S = 0.5*(S + S.T)
    scale = float(np.mean(np.diag(S))) or 1.0
    for k in range(8):
        jit = 0.0 if k == 0 else _JIT * (10.0 ** k) * scale
        try:
            return np.linalg.cholesky(S + jit * np.eye(len(S)))
        except np.linalg.LinAlgError:
            continue

    w, V = np.linalg.eigh(S)
    top = float(w.max())
    if not np.isfinite(top) or top <= 0.0:
        raise np.linalg.LinAlgError(f'{name} has no positive eigenvalue (max {top:.3e})')

    floor = top * 1e-10
    n_neg = int((w < 0).sum())
    global _CHOL_REPAIRS
    _CHOL_REPAIRS += 1
    print(f'    [ppd] _chol repair on {name}: {n_neg} negative eigenvalue(s), min {w.min():.3e} vs max {top:.3e}, clipped to {floor:.3e} (repair #{_CHOL_REPAIRS})', flush=True)

    S = 0.5*(((V * np.clip(w, floor, None)) @ V.T) + ((V * np.clip(w, floor, None)) @ V.T).T)

    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        pass

    print(f' [ppd] _chol repair INSUFFICIENT on {name}: shape {S.shape} finite={bool(np.isfinite(S).all())} sym_err={float(np.abs(S - S.T).max()):.3e} eig[min,max]=[{w.min():.3e},{top:.3e}] diag[min,max]=[{float(np.min(np.diag(S))):.3e},{float(np.max(np.diag(S))):.3e}]', flush=True)
    if not np.isfinite(S).all():
        raise np.linalg.LinAlgError(f'{name} contains non-finite entries')

    for k in range(1, 12):
        jit = top * (10.0 ** (-12 + k))
        try:
            L = np.linalg.cholesky(S + jit * np.eye(len(S)))
            print(f' [ppd] forced jitter {jit:.3e} ({10.0 ** (-12 + k):.0e} x top eigenvalue) succeeded on {name}', flush=True)
            return L
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(f'{name}: not factorisable even at 1e-1 relative jitter (top {top:.3e})')


def _logdet(L):
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def gauss_logpdf(Y, mu, L):
    # Log Gaussian density for Y [S, K]
    z = _tri_solve(L, (Y - mu).T)
    return -0.5 * ((z ** 2).sum(0) + _logdet(L) + len(mu) * np.log(2 * np.pi))


def _component_logpdf(Y, MU, SIG):
    # Log densities for each mixture component [J, S]
    return np.stack([gauss_logpdf(Y, MU[j], _chol(SIG[j], f'SIG[{j}]')) for j in range(len(MU))])


def _lse(A, axis=0):
    m = A.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def mixture_sample(MU, SIG, n, rng):
    src = rng.integers(0, len(MU), size=n)
    Y = np.empty((n, MU.shape[1]))
    for j in np.unique(src):
        w = src == j
        Y[w] = MU[j] + rng.standard_normal((int(w.sum()), MU.shape[1])) @ _chol(SIG[j]).T
    return Y, src


def moment_match(MU, SIG):
    # moments of the finite mixture, so divide by J
    mu = MU.mean(0)
    D = MU - mu
    return mu, SIG.mean(0) + D.T @ D / max(len(MU), 1)


def kl_gauss(mu0, S0, mu1, S1):
    # KL(N0 || N1) in nats
    L0, L1 = _chol(S0, 'S0'), _chol(S1, 'S1')
    M = _tri_solve(L1, L0)
    d = _tri_solve(L1, (mu1 - mu0).reshape(-1, 1))
    return 0.5 * (float((M ** 2).sum()) + float((d ** 2).sum()) - len(mu0) + _logdet(L1) - _logdet(L0))


def score_joint_ppd(MU, SIG, mu_phi, Sig_phi, n_mc=4000, seed=0):
    # KL decomposition in nats per cell
    MU = np.asarray(MU, float); SIG = np.asarray(SIG, float)
    mu_phi = np.asarray(mu_phi, float); Sig_phi = np.asarray(Sig_phi, float)
    J, K = MU.shape
    rng = np.random.default_rng(seed)

    Y, src = mixture_sample(MU, SIG, n_mc, rng)
    LOG = _component_logpdf(Y, MU, SIG)
    lp = gauss_logpdf(Y, mu_phi, _chol(Sig_phi, 'Sig_phi'))

    lq_self = _lse(LOG, 0) - np.log(J)
    LOO = LOG.copy()
    LOO[src, np.arange(n_mc)] = -np.inf
    lq = _lse(LOO, 0) - np.log(J - 1)

    kl_total = float(np.mean(lq - lp))
    H_q = float(-np.mean(lq))
    mu_G, S_G = moment_match(MU, SIG)
    H_G = 0.5 * (_logdet(_chol(S_G, 'S_G')) + K * np.log(2 * np.pi * np.e))
    kl_amort = kl_gauss(mu_G, S_G, mu_phi, Sig_phi)
    floor_mc = H_G - H_q

    return dict(kl_total_self=float(np.mean(lq_self - lp)) / K, kl_amort=kl_amort / K, decomp_residual=(kl_total - floor_mc - kl_amort) /K,
                kl_G_vs_diag=kl_gauss(mu_G, S_G, mu_G,np.diag(np.diag(S_G))) / K)
