"""Small statistics helpers for time series of observables."""

from __future__ import annotations

import numpy as np


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of x (T, B), averaged over the B series, lags 0..max_lag."""
    x = x - x.mean(axis=0)
    var = (x * x).mean()
    return np.array([(x[: len(x) - k] * x[k:]).mean() / var for k in range(max_lag + 1)])


def tau_int(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time in frames, with Sokal's automatic window (M >= c·tau)."""
    rho = autocorr(x, len(x) // 4)
    tau = 1.0
    for m in range(1, len(rho)):
        tau = 1.0 + 2.0 * rho[1 : m + 1].sum()
        if m >= c * tau:
            break
    return float(max(tau, 1.0))
