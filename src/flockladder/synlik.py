"""Gaussian synthetic likelihood (Wood 2010) on a summary vector, and its optimisation.

Used in notebooks 03–05. θ is handled as log θ = (log η, log r,
log v0). Each evaluation simulates B replicas with the same design as the data (``DESIGN``); with a
fixed ``seed`` the objective is a deterministic function of θ (common random numbers). Every
simulation goes through the disk cache, so re-running a notebook replays optimisations for free.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

import numpy as np
from scipy.optimize import minimize

from flockladder.observables import run_summaries
from flockladder.vicsek import VicsekParams

SUMMARIES = ("speed", "cos_turn", "corr_0_1", "corr_1_2", "phi", "n_clusters", "nn_dist")
DESIGN: dict[str, Any] = {"n": 40, "L": 10.0}
RUN = {"k": 10, "n_frames": 501, "burn_in": 5000}
PARAMS = ("eta", "r", "v0")
TRUTH = np.array([2.0, 1.0, 0.03])
BOX = np.array([[0.5, 5.0], [0.5, 2.0], [0.015, 0.06]])
DATA_SEED = 10**6


def params(theta: np.ndarray) -> VicsekParams:
    eta, r, v0 = (float(x) for x in theta)
    return VicsekParams(eta=eta, r=r, v0=v0, **DESIGN)


def summary_matrix(theta: np.ndarray, n_replicas: int, seed: int) -> np.ndarray:
    """(n_replicas, d) summaries of independent runs at θ (natural scale)."""
    s = run_summaries(params(theta), n_replicas=n_replicas, seed=seed, **RUN)
    return np.stack([s[name] for name in SUMMARIES], axis=-1)


def observed(seed: int = DATA_SEED, theta: np.ndarray = TRUTH) -> np.ndarray:
    """The data: one run at ``theta`` (default the truth) with a held-out seed. (d,)"""
    return summary_matrix(theta, 1, seed)[0]


def shrunk_cov(S: np.ndarray) -> tuple[np.ndarray, float]:
    """Covariance of the rows of S with correlations shrunk toward 0 (Schäfer & Strimmer 2005,
    target D): the variances are kept, off-diagonal correlations are scaled by (1 - λ*) with the
    analytic λ* = Σ var(r_ij) / Σ r_ij² (i ≠ j), clipped to [0, 1]. Returns (cov, λ*)."""
    n = len(S)
    sd = S.std(axis=0, ddof=1)
    X = (S - S.mean(axis=0)) / sd
    W = X[:, :, None] * X[:, None, :]  # (n, d, d)
    R = W.sum(axis=0) / (n - 1)
    var_r = n / (n - 1) ** 3 * ((W - W.mean(axis=0)) ** 2).sum(axis=0)
    off = ~np.eye(S.shape[1], dtype=bool)
    lam = float(np.clip(var_r[off].sum() / (R[off] ** 2).sum(), 0.0, 1.0))
    R_star = np.where(off, (1 - lam) * R, 1.0)
    return R_star * np.outer(sd, sd), lam


def synlik(
    log_theta: np.ndarray,
    s_obs: np.ndarray,
    n_replicas: int = 40,
    seed: int = 0,
    shrink: bool = False,
) -> float:
    """Gaussian synthetic log-likelihood; -inf outside the box. ``shrink`` uses ``shrunk_cov``."""
    theta = np.exp(log_theta)
    if np.any(theta < BOX[:, 0]) or np.any(theta > BOX[:, 1]):
        return -np.inf
    S = summary_matrix(theta, n_replicas, seed)
    mu = S.mean(axis=0)
    cov = shrunk_cov(S)[0] if shrink else np.cov(S, rowvar=False)
    diff = s_obs - mu
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    return float(-0.5 * diff @ np.linalg.solve(cov, diff) - 0.5 * logdet)


@dataclass
class Fit:
    start: np.ndarray
    seed: int
    theta: np.ndarray = field(default_factory=lambda: np.empty(0))
    loglik: float = float("nan")
    nfev: int = 0
    path: list = field(default_factory=list)  # (theta, loglik) for every evaluation


def fit(
    s_obs: np.ndarray,
    start: np.ndarray,
    seed: int = 0,
    n_replicas: int = 40,
    maxfev: int = 100,
    step=0.2,
    shrink: bool = False,
    whiten: np.ndarray | None = None,
    y0: np.ndarray | None = None,
    xatol: float = 0.01,
) -> Fit:
    """Nelder–Mead from ``start`` (natural scale), CRN with ``seed``.

    Default: optimise x = log θ; ``step`` is the initial simplex edge per log-parameter.
    With ``whiten`` = L (3×3): optimise y with log θ = log(start) + L y, starting at ``y0``
    (default 0);
    ``step`` and ``xatol`` are then in y units."""
    out = Fit(start=np.asarray(start, float), seed=seed)
    base = np.log(out.start)
    to_x = (lambda y: base + whiten @ y) if whiten is not None else (lambda y: y)

    def f(y):
        x = to_x(y)
        ll = synlik(x, s_obs, n_replicas, seed, shrink)
        out.path.append((np.exp(x), ll))
        return 1e10 if not np.isfinite(ll) else -ll

    if whiten is not None:
        z0 = np.zeros(len(base)) if y0 is None else np.asarray(y0, float)
    else:
        z0 = base
    steps = np.broadcast_to(np.asarray(step, float), z0.shape)
    simplex = np.vstack([z0] + [z0 + h * e for h, e in zip(steps, np.eye(len(z0)), strict=True)])
    res = minimize(
        f,
        z0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": xatol, "fatol": 0.1, "maxfev": maxfev},
    )
    out.theta, out.loglik, out.nfev = np.exp(to_x(res.x)), -res.fun, res.nfev
    return out


def _fit_kw(kw):
    return fit(**kw)


def fit_jobs(jobs: list[dict], workers: int = 6) -> list[Fit]:
    """Run ``fit(**job)`` for each job dict in parallel processes (one BLAS thread each)."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        return list(ex.map(_fit_kw, jobs))


def _fit_args(args):
    return fit(*args)


def fit_many(s_obs: np.ndarray, starts: list, seeds: list, workers: int = 6, **kw) -> list[Fit]:
    """Run ``fit`` for each (start, seed) pair in parallel processes (one BLAS thread each)."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # inherited by the spawned workers
    jobs = [
        (
            s_obs,
            st,
            sd,
            kw.get("n_replicas", 40),
            kw.get("maxfev", 100),
            kw.get("step", 0.2),
            kw.get("shrink", False),
        )
        for st, sd in zip(starts, seeds, strict=True)
    ]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        return list(ex.map(_fit_args, jobs))


def latin_hypercube(n: int, rng: np.random.Generator, box: np.ndarray = BOX) -> np.ndarray:
    """n points in ``box`` (natural scale, (3, 2)), stratified in log θ. (n, 3)"""
    lo, hi = np.log(box[:, 0]), np.log(box[:, 1])
    u = (np.stack([rng.permutation(n) for _ in lo], axis=1) + rng.uniform(size=(n, len(lo)))) / n
    return np.exp(lo + u * (hi - lo))


def _synlik_args(args):
    return synlik(*args)


def synlik_many(
    points: list, s_obs: np.ndarray, seeds: list, n_replicas: int = 40, workers: int = 6
) -> np.ndarray:
    """ℓ at each natural-scale θ in ``points`` with the matching seed, in parallel."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    jobs = [
        (np.log(np.asarray(t, float)), s_obs, n_replicas, sd)
        for t, sd in zip(points, seeds, strict=True)
    ]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        return np.array(list(ex.map(_synlik_args, jobs)))


def _one_run(args):
    theta, seed = args
    return summary_matrix(theta, 1, seed)[0]


@dataclass
class Pilot:
    thetas: np.ndarray  # (M, 3) design
    S: np.ndarray  # (M, d) one run each
    coef: np.ndarray  # (d + 1, 3) regression of log θ on standardized summaries
    center: np.ndarray
    scale: np.ndarray
    resid_sd: np.ndarray  # (3,) in log θ

    def predict(self, s: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(s) - self.center) / self.scale
        return np.exp(np.hstack([np.ones((len(z), 1)), z]) @ self.coef)


def pilot(m: int = 100, seed: int = 11, workers: int = 6) -> Pilot:
    """Semi-automatic pilot (Fearnhead & Prangle 2012): one run at each of m Latin-hypercube θ,
    linear regression of log θ on standardized summaries."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    thetas = latin_hypercube(m, np.random.default_rng(seed))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        S = np.array(list(ex.map(_one_run, [(t, 1000 + i) for i, t in enumerate(thetas)])))
    center, scale = S.mean(axis=0), S.std(axis=0, ddof=1)
    X = np.hstack([np.ones((m, 1)), (S - center) / scale])
    Y = np.log(thetas)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid_sd = (Y - X @ coef).std(axis=0, ddof=X.shape[1])
    return Pilot(thetas, S, coef, center, scale, resid_sd)


def fit_profile(
    s_obs: np.ndarray,
    r: float,
    start: np.ndarray,
    seed: int = 0,
    n_replicas: int = 40,
    maxfev: int = 60,
    step=0.05,
    shrink: bool = True,
) -> Fit:
    """Profile fit: r fixed, Nelder–Mead over log (η, v0) from ``start`` = (η, v0)."""
    out = Fit(start=np.array([start[0], r, start[1]], float), seed=seed)

    def f(y):
        x = np.array([y[0], np.log(r), y[1]])
        ll = synlik(x, s_obs, n_replicas, seed, shrink)
        out.path.append((np.exp(x), ll))
        return 1e10 if not np.isfinite(ll) else -ll

    y0 = np.log(np.asarray(start, float))
    steps = np.broadcast_to(np.asarray(step, float), y0.shape)
    simplex = np.vstack([y0] + [y0 + h * e for h, e in zip(steps, np.eye(2), strict=True)])
    res = minimize(
        f,
        y0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 0.002, "fatol": 0.1, "maxfev": maxfev},
    )
    out.theta, out.loglik, out.nfev = (
        np.array([np.exp(res.x[0]), r, np.exp(res.x[1])]),
        -res.fun,
        res.nfev,
    )
    return out


def _profile_args(args):
    return fit_profile(*args)


def profile_many(
    s_obs: np.ndarray, rs: list, start: np.ndarray, step, workers: int = 6, **kw
) -> list[Fit]:
    """``fit_profile`` at each r in parallel."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    jobs = [
        (
            s_obs,
            r,
            start,
            kw.get("seed", 0),
            kw.get("n_replicas", 40),
            kw.get("maxfev", 60),
            step,
            kw.get("shrink", True),
        )
        for r in rs
    ]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        return list(ex.map(_profile_args, jobs))


def gaussian_terms(theta: np.ndarray, n_replicas: int = 40, seed: int = 0):
    """Data-independent pieces of ``synlik`` at θ: (μ̂, Σ̂⁻¹, log det Σ̂). Plain covariance."""
    S = summary_matrix(theta, n_replicas, seed)
    cov = np.cov(S, rowvar=False)
    return S.mean(axis=0), np.linalg.inv(cov), np.linalg.slogdet(cov)[1]


def synlik_from_terms(terms, s_obs: np.ndarray) -> np.ndarray:
    """ℓ for one or many datasets s_obs (..., d) at a point with precomputed ``gaussian_terms``."""
    mu, prec, logdet = terms
    d = np.atleast_2d(s_obs) - mu
    return -0.5 * np.einsum("bi,ij,bj->b", d, prec, d) - 0.5 * logdet


def quadratic_max(Y: np.ndarray, ell: np.ndarray):
    """Least-squares ℓ ≈ c + gᵀy − ½ yᵀHy over points Y (n, k). Returns (y*, H, residual sd);
    y* is None
    if H is not positive definite."""
    n, k = Y.shape
    iu = np.triu_indices(k)
    quad = np.stack(
        [Y[:, i] * Y[:, j] * (0.5 if i == j else 1.0) for i, j in zip(*iu, strict=True)], axis=1
    )
    X = np.hstack([np.ones((n, 1)), Y, -quad])
    coef, *_ = np.linalg.lstsq(X, ell, rcond=None)
    g = coef[1 : 1 + k]
    H = np.zeros((k, k))
    H[iu] = coef[1 + k :]
    H = H + H.T - np.diag(np.diag(H))
    resid = ell - X @ coef
    sd = float(resid.std(ddof=X.shape[1]))
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return None, H, sd
    return np.linalg.solve(H, g), H, sd
