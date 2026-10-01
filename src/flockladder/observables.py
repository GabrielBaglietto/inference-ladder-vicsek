"""Observables. All reduce over the particle axis and keep leading (time, replica) axes.

``polarization`` uses the true heading angles and is for the simulator side only. Everything else is
computed from an ``Observation`` (positions every k steps) — the only thing inference may see.
Distances used here (cluster linkage ``ell``, correlation bins) are fixed in data units by the
analyst; they are *not* the unknown interaction radius r.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from flockladder.cache import disk_cache
from flockladder.observe import Observation, displacements, observe, wrap
from flockladder.vicsek import VicsekParams

CORR_BINS = (0.0, 1.0, 2.0, 4.0)


def polarization(heading: np.ndarray) -> np.ndarray:
    """Order parameter |<exp(i α)>| of the heading angles α: 1 when all agree, ~sqrt(pi/(4N)) when
    random."""
    return np.abs(np.exp(1j * heading).mean(axis=-1))


def _unit(d: np.ndarray) -> np.ndarray:
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _pair_distances(x: np.ndarray, L: float) -> np.ndarray:
    """(..., N, N) minimum-image distances for positions (..., N, 2)."""
    return np.linalg.norm(wrap(x[..., :, None, :] - x[..., None, :, :], L), axis=-1)


def cluster_labels(x: np.ndarray, L: float, ell: float) -> np.ndarray:
    """(N,) connected-component labels of one frame, linking particles closer than ``ell``."""
    return connected_components(csr_array(_pair_distances(x, L) < ell), directed=False)[1]


def component_labels(adj: np.ndarray) -> np.ndarray:
    """Connected-component labels for a batch of graphs, adj (..., N, N) boolean and symmetric.

    Each node takes the minimum label among itself and its neighbours until nothing changes, so a
    component is labelled by its smallest node index. Vectorised over the leading axes; the number
    of sweeps is at most the largest component diameter.
    """
    n = adj.shape[-1]
    adj = adj | np.eye(n, dtype=bool)
    labels = np.broadcast_to(np.arange(n), adj.shape[:-1]).copy()
    while True:
        new = np.where(adj, labels[..., None, :], n).min(axis=-1)
        if np.array_equal(new, labels):
            return labels
        labels = new


def cluster_sizes(x: np.ndarray, L: float, ell: float) -> np.ndarray:
    """(N,) size of the cluster each particle belongs to, for one frame."""
    labels = cluster_labels(x, L, ell)
    return np.bincount(labels)[labels]


def series(obs: Observation, ell: float = 1.0) -> dict[str, np.ndarray]:
    """Per-frame observables, each (T', B). T' differs by one or two between observables because
    displacements need two frames and turning angles three."""
    d = displacements(obs)
    u = _unit(d)
    n = obs.pos.shape[2]
    out = {
        "phi": np.linalg.norm(u.mean(axis=2), axis=-1),
        "speed": np.linalg.norm(d, axis=-1).mean(axis=2) / obs.k,
        "cos_turn": (u[1:] * u[:-1]).sum(-1).mean(axis=2),
    }
    nn, ncl, big, corr = [], [], [], []
    for t in range(obs.pos.shape[0]):
        dist = _pair_distances(obs.pos[t], obs.L)  # (B, N, N)
        nn.append(np.where(np.eye(n, dtype=bool), np.inf, dist).min(axis=-1).mean(axis=-1))
        labels = component_labels(
            dist < ell
        )  # (B, N), a component is labelled by its smallest index
        ncl.append((labels == np.arange(n)).sum(axis=-1))
        big.append((labels[..., None] == np.arange(n)).sum(axis=-2).max(axis=-1) / n)
        if t < d.shape[0]:
            dot = np.einsum("bik,bjk->bij", u[t], u[t])
            off = ~np.eye(n, dtype=bool)
            row = []
            for lo, hi in pairwise(CORR_BINS):
                m = (dist >= lo) & (dist < hi) & off
                row.append((dot * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1))
            corr.append(np.stack(row, axis=-1))
    out["nn_dist"] = np.array(nn)
    out["n_clusters"] = np.array(ncl, dtype=float)
    out["largest_cluster"] = np.array(big)
    corr_arr = np.array(corr)
    for j, (lo, hi) in enumerate(pairwise(CORR_BINS)):
        out[f"corr_{lo:g}_{hi:g}"] = corr_arr[..., j]
    return out


def summaries(s: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Time-average each series per replica, (B,), plus the susceptibility N·var_t(phi)."""
    out = {name: x.mean(axis=0) for name, x in s.items()}
    out["chi"] = n * s["phi"].var(axis=0)
    return out


@disk_cache
def run_summaries(
    p: VicsekParams,
    k: int,
    n_frames: int,
    burn_in: int,
    n_replicas: int,
    seed: int,
    ell: float = 1.0,
) -> dict[str, np.ndarray]:
    """Simulate, observe every k steps, and return the per-replica summaries (cached on disk)."""
    obs = observe(p, n_frames, k, n_replicas, seed=seed, burn_in=burn_in)
    return summaries(series(obs, ell), p.n)
