"""Relaxation rates from a disordered start (notebook 02), by a protocol fixed before any fit.

The gap g(t) = |y_inf - y(t)| to a late plateau is fitted log-linearly on a window fixed by
fractions of the initial gap (0.8 -> 0.2), never chosen by eye.
"""

from __future__ import annotations

import numpy as np

from flockladder.cache import disk_cache
from flockladder.observables import series
from flockladder.observe import observe
from flockladder.vicsek import VicsekParams


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(t, y, 1)[0]) if len(t) >= 3 else np.nan


def gap_rate(
    t: np.ndarray,
    y: np.ndarray,
    plateau: tuple[float, float] = (4000, 6000),
    lo: float = 0.8,
    hi: float = 0.2,
) -> dict[str, float]:
    """Rate of approach of y(t) to its plateau. ``t`` in steps, ``y`` one (group-mean) series.

    Returns lam (1/steps), t_half (steps), the window, the rates of its two halves, and whether a
    single rate is adequate (halves agree within 30%). NaN where the protocol cannot be applied.
    """
    y_inf = y[(t >= plateau[0]) & (t <= plateau[1])].mean()
    g = np.abs(y_inf - y)
    g0 = g[0]

    def first(frac):
        idx = np.flatnonzero(g <= frac * g0)
        return idx[0] if len(idx) else None

    a, b, h = first(lo), first(hi), first(0.5)
    out = {
        "lam": np.nan,
        "t_half": t[h] if h is not None else np.nan,
        "t_a": np.nan,
        "t_b": np.nan,
        "lam_1": np.nan,
        "lam_2": np.nan,
        "single": np.nan,
    }
    if a is None or b is None or b - a < 3:
        return out
    tw, lg = t[a : b + 1], np.log(g[a : b + 1])
    m = len(tw) // 2
    lam, lam_1, lam_2 = -_slope(tw, lg), -_slope(tw[: m + 1], lg[: m + 1]), -_slope(tw[m:], lg[m:])
    single = abs(lam_1 - lam_2) <= 0.3 * max(abs(lam_1), abs(lam_2))
    out.update(lam=lam, t_a=tw[0], t_b=tw[-1], lam_1=lam_1, lam_2=lam_2, single=float(single))
    return out


@disk_cache
def run_relaxation(
    p: VicsekParams,
    n_groups: int = 8,
    group_size: int = 5,
    n_steps: int = 6000,
    k: int = 10,
    seed: int = 0,
    observables: tuple[str, ...] = ("phi", "nn_dist"),
) -> dict[str, np.ndarray]:
    """Random start, observe every k steps, apply ``gap_rate`` per group of replicas.

    Returns ``<obs>_<field>`` -> (n_groups,) arrays, plus ``<obs>_curve`` -> (T, n_groups) group
    means and ``<obs>_t`` -> (T,) steps, for plotting (T differs: displacement-based series have one
    point fewer than frames).
    """
    obs = observe(p, n_steps // k + 1, k, n_groups * group_size, seed=seed)
    s = series(obs)
    out = {}
    for name in observables:
        x = s[name]
        t = k * np.arange(len(x))
        curves = x.reshape(len(x), n_groups, group_size).mean(axis=2)
        fits = [gap_rate(t, curves[:, j]) for j in range(n_groups)]
        for field in fits[0]:
            out[f"{name}_{field}"] = np.array([f[field] for f in fits])
        out[f"{name}_curve"] = curves
        out[f"{name}_t"] = t
    return out
