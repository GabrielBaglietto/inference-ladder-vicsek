"""Local identifiability from the synthetic likelihood, I = Jᵀ Σ⁻¹ J (per run).

J = ∂μ/∂log θ comes from ``flockladder.sensitivity`` (±10% central differences, common random
numbers),
Σ is the covariance of the summaries at θ, and both are estimated from the same B replicas.
"""

from __future__ import annotations

import numpy as np

PARAMS = ("eta", "r", "v0")


def local_info(sens: dict, names: list[str]) -> dict[str, np.ndarray]:
    """Per-replica values and derivatives for the summaries in ``names``:
    V (B, d) summary values at θ, D (B, d, 3) per-replica d summary / d log θ."""
    V = np.stack([sens[s]["value"] for s in names], axis=-1)
    D = np.stack([np.stack([sens[s][f"d_{q}"] for q in PARAMS], axis=-1) for s in names], axis=1)
    return {"V": V, "D": D}


def fisher(V: np.ndarray, D: np.ndarray) -> np.ndarray:
    """I = Jᵀ Σ⁻¹ J with J = mean over replicas of D, Σ = cov of V. (3, 3)

    Summaries that are constant across replicas (e.g. n_clusters in a dense box) carry no
    information and would make Σ singular; they are dropped."""
    keep = V.var(axis=0) > 0
    V, D = V[:, keep], D[:, keep]
    J = D.mean(axis=0)
    cov = np.atleast_2d(np.cov(V, rowvar=False))
    return J.T @ np.linalg.solve(cov, J)


def summarize(info: np.ndarray) -> dict[str, np.ndarray]:
    """se of log θ (from I⁻¹; inf if I is singular) and the correlation matrix of I⁻¹."""
    try:
        C = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return {"se": np.full(3, np.inf), "corr": np.full((3, 3), np.nan)}
    se = np.sqrt(np.diag(C))
    return {"se": se, "corr": C / np.outer(se, se)}


def bootstrap(V: np.ndarray, D: np.ndarray, n: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """Resample replicas jointly; returns arrays of se (n, 3) and corr (n, 3, 3)."""
    rng = np.random.default_rng(seed)
    se, corr = [], []
    for _ in range(n):
        idx = rng.integers(0, len(V), len(V))
        s = summarize(fisher(V[idx], D[idx]))
        se.append(s["se"])
        corr.append(s["corr"])
    return {"se": np.array(se), "corr": np.array(corr)}
