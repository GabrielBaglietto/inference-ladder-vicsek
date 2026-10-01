"""Precomputed results of the long runs, stored in ``results/`` as ``.npz``.

Each file is written by a script in ``scripts/`` (which also regenerates it from scratch) and read
by the notebooks, so that every notebook opens and runs in minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flockladder.synlik import Fit

RESULTS = Path(__file__).resolve().parents[2] / "results"


def save(name: str, **arrays: np.ndarray) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.npz"
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]  # numpy stub mis-types **kwds
    return path


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(RESULTS / f"{name}.npz") as f:
        return dict(f)


def pack_fits(prefix: str, fits: list[Fit]) -> dict[str, np.ndarray]:
    """Fits as arrays: start, theta, loglik, nfev, and the evaluation paths padded with NaN."""
    m = max(len(f.path) for f in fits)
    path_theta = np.full((len(fits), m, 3), np.nan)
    path_ll = np.full((len(fits), m), np.nan)
    for i, f in enumerate(fits):
        path_theta[i, : len(f.path)] = [p[0] for p in f.path]
        path_ll[i, : len(f.path)] = [p[1] for p in f.path]
    return {
        f"{prefix}_start": np.array([f.start for f in fits]),
        f"{prefix}_theta": np.array([f.theta for f in fits]),
        f"{prefix}_loglik": np.array([f.loglik for f in fits]),
        f"{prefix}_nfev": np.array([f.nfev for f in fits]),
        f"{prefix}_path_theta": path_theta,
        f"{prefix}_path_ll": path_ll,
    }


def unpack_fits(prefix: str, d: dict[str, np.ndarray]) -> list[Fit]:
    """Inverse of ``pack_fits``. The seed is not stored and is set to -1."""
    fits = []
    for i in range(len(d[f"{prefix}_theta"])):
        n = int(np.sum(~np.isnan(d[f"{prefix}_path_theta"][i, :, 0])))
        f = Fit(start=d[f"{prefix}_start"][i], seed=-1)
        f.theta = d[f"{prefix}_theta"][i]
        f.loglik = float(d[f"{prefix}_loglik"][i])
        f.nfev = int(d[f"{prefix}_nfev"][i])
        f.path = [
            (d[f"{prefix}_path_theta"][i, t], float(d[f"{prefix}_path_ll"][i, t])) for t in range(n)
        ]
        fits.append(f)
    return fits


def pack_sensitivity(prefix: str, sens: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """``flockladder.sensitivity`` output as flat arrays: ``<prefix>__<summary>__<field>``."""
    return {f"{prefix}__{s}__{k}": v for s, d in sens.items() for k, v in d.items()}


def unpack_sensitivity(prefix: str, d: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    head = f"{prefix}__"
    for key, v in d.items():
        if key.startswith(head):
            s, k = key[len(head) :].split("__")
            out.setdefault(s, {})[k] = v
    return out
