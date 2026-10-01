"""Unbiased Bayesian synthetic likelihood (uBSL) and a pseudo-marginal Metropolis–Hastings chain.

The estimator is Ghurye & Olkin's (1969) unbiased estimator of a Gaussian density, as used by Price,
Drovandi, Lee & Nott (2018); the formula follows eq. (3) of An et al. (2019, arXiv 1907.10940).
Used in notebook 06. Every proposal uses fresh random numbers (no CRN), as pseudo-marginal MCMC
requires.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from scipy.special import gammaln

from flockladder.synlik import BOX, summary_matrix


def log_c(k: int, v: int) -> float:
    """log c(k, v) = log[2^(−kv/2) π^(−k(k−1)/4) / Π_{i=1..k} Γ((v−i+1)/2)]."""
    i = np.arange(1, k + 1)
    return -k * v / 2 * np.log(2) - k * (k - 1) / 4 * np.log(np.pi) - gammaln((v - i + 1) / 2).sum()


def log_ubsl(S: np.ndarray, s_obs: np.ndarray) -> float:
    """log of the unbiased Gaussian-density estimator at s_obs from simulations S (n, d).

    -inf if Ψ = 0."""
    n, d = S.shape
    if n <= d + 3:
        raise ValueError("uBSL needs n > d + 3")
    mu = S.mean(axis=0)
    M = (S - mu).T @ (S - mu)  # (n − 1) Σ_n
    diff = s_obs - mu
    A = M - np.outer(diff, diff) / (1 - 1 / n)
    sign_a, logdet_a = np.linalg.slogdet(A)
    if sign_a <= 0:
        return -np.inf
    try:
        np.linalg.cholesky(A)  # Ψ(A) = |A| only if A is positive definite
    except np.linalg.LinAlgError:
        return -np.inf
    _, logdet_m = np.linalg.slogdet(M)
    return float(
        -d / 2 * np.log(2 * np.pi)
        + log_c(d, n - 2)
        - log_c(d, n - 1)
        - d / 2 * np.log(1 - 1 / n)
        - (n - d - 2) / 2 * logdet_m
        + (n - d - 3) / 2 * logdet_a
    )


def in_box(theta: np.ndarray) -> bool:
    return bool(np.all(theta >= BOX[:, 0]) and np.all(theta <= BOX[:, 1]))


def loglik(theta: np.ndarray, s_obs: np.ndarray, n: int, seed: int) -> float:
    return log_ubsl(summary_matrix(theta, n, seed), s_obs) if in_box(theta) else -np.inf


def chain(
    s_obs: np.ndarray, x0: np.ndarray, prop_chol: np.ndarray, n: int, iters: int, chain_id: int
) -> dict[str, np.ndarray]:
    """Pseudo-marginal RW-MH in x = log θ with uniform prior on the box.

    The seed of evaluation t in chain c is 10**7 + 10**5 * c + t (fresh random numbers at every
    proposal). Returns the path of x, log p̂ and the accept flags."""
    rng = np.random.default_rng(500 + chain_id)
    x = np.asarray(x0, float)
    ll = loglik(np.exp(x), s_obs, n, 10**7 + 10**5 * chain_id)
    xs, lls, acc = [x], [ll], []
    for t in range(1, iters + 1):
        y = x + prop_chol @ rng.standard_normal(len(x))
        lly = loglik(np.exp(y), s_obs, n, 10**7 + 10**5 * chain_id + t)
        a = np.log(rng.uniform()) < lly - ll  # symmetric proposal, flat prior inside the box
        if a:
            x, ll = y, lly
        xs.append(x)
        lls.append(ll)
        acc.append(a)
    return {"x": np.array(xs), "ll": np.array(lls), "acc": np.array(acc)}


def _chain_kw(kw):
    return chain(**kw)


def run_chains(jobs: list[dict], workers: int = 6) -> list[dict]:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        return list(ex.map(_chain_kw, jobs))


def split_rhat(chains: np.ndarray) -> float:
    """Split-R̂ (Gelman et al.) for one scalar: chains (m, T)."""
    T = chains.shape[1]
    h = T // 2
    c = np.vstack([chains[:, :h], chains[:, h : 2 * h]])
    w = c.var(axis=1, ddof=1).mean()
    b = h * c.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((h - 1) / h * w + b / h) / w))


def ess(chains: np.ndarray) -> float:
    """Bulk ESS: m·T / (1 + 2 Σ ρ_k), autocorrelation pooled over chains.

    Truncated with Geyer's initial positive sequence."""
    m, T = chains.shape
    x = chains - chains.mean()
    var = (x * x).mean()
    rho = [1.0]
    for k in range(1, T // 2):
        rho.append(float((x[:, :-k] * x[:, k:]).mean() / var))
        if k % 2 == 1 and rho[-1] + rho[-2] < 0:
            rho = rho[:-2]
            break
    return float(m * T / (1 + 2 * sum(rho[1:])))
