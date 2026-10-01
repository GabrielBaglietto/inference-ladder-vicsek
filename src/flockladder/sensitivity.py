"""Local sensitivity of summaries to parameters, by central differences in log-parameter.

Every perturbed run reuses the seed of the central run (common random numbers). Vicsek dynamics
decorrelate over long runs, so this reduces noise only partly; the standard error reported is
computed across replicas from the per-replica differences, so it is honest either way.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from flockladder.observables import run_summaries
from flockladder.vicsek import VicsekParams


def sensitivity(
    p: VicsekParams,
    names: tuple[str, ...] = ("eta", "r", "v0"),
    rel: float = 0.1,
    runner=run_summaries,
    **run_kw,
) -> dict[str, dict[str, np.ndarray]]:
    """For each summary: its per-unit values at ``p`` ("value") and, for each parameter in
    ``names``, the per-unit derivative d summary / d log(parameter) ("d_<name>").

    ``runner(p, **run_kw)`` returns summary -> (units,) arrays; units are replicas by default
    (``run_summaries``), or groups of replicas (``flockladder.rates.run_relaxation``)."""
    center = {s: v for s, v in runner(p, **run_kw).items() if v.ndim == 1 and len(v) != 0}
    out = {s: {"value": v} for s, v in center.items()}
    for name in names:
        base = getattr(p, name)
        hi = runner(replace(p, **{name: base * np.exp(rel)}), **run_kw)
        lo = runner(replace(p, **{name: base * np.exp(-rel)}), **run_kw)
        for s in center:
            out[s][f"d_{name}"] = (hi[s] - lo[s]) / (2 * rel)
    return out


def table(sens: dict[str, dict[str, np.ndarray]], names=("eta", "r", "v0"), step: float = 0.1):
    """Rows: summary -> mean, sd across replicas, and for each parameter the shift produced by a
    ``step`` (10%) change, in units of one run's sd: z = step * mean(d)/sd, with its standard
    error."""
    rows = {}
    for s, d in sens.items():
        v = d["value"]
        sd = v.std(ddof=1)
        row: dict[str, float] = {"mean": float(v.mean()), "sd": float(sd)}
        for name in names:
            g = d[f"d_{name}"]
            row[f"z_{name}"] = step * g.mean() / sd if sd > 0 else np.nan
            row[f"z_{name}_se"] = step * g.std(ddof=1) / np.sqrt(len(g)) / sd if sd > 0 else np.nan
        rows[s] = row
    return rows
