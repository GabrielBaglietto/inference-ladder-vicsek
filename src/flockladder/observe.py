"""The observation model: what an experiment would record from a simulated flock.

Positions of all particles every ``k`` simulation steps, at a constant rate. Headings are not
observed, and there is no tracking noise. Everything that is
used for inference must be computed from an ``Observation`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flockladder.vicsek import VicsekParams, simulate


@dataclass(frozen=True)
class Observation:
    pos: np.ndarray
    """(T, B, N, 2) positions, one frame every ``k`` steps."""
    k: int
    """Simulation steps per frame (known)."""
    L: float
    """Box side (known)."""


def observe(
    p: VicsekParams,
    n_frames: int,
    k: int,
    n_replicas: int = 1,
    seed: int | np.random.Generator | None = None,
    burn_in: int = 0,
) -> Observation:
    """Simulate ``burn_in`` unrecorded steps, then record ``n_frames`` frames every ``k`` steps."""
    rng = np.random.default_rng(seed)
    state = None
    if burn_in:
        tr = simulate(p, burn_in, n_replicas, seed=rng, record_every=burn_in)
        state = (tr.pos[-1], tr.heading[-1])
    tr = simulate(p, (n_frames - 1) * k, n_replicas, seed=rng, record_every=k, state=state)
    return Observation(tr.pos, k, p.L)


def wrap(d: np.ndarray, L: float) -> np.ndarray:
    """Minimum-image convention for separations in a periodic box."""
    return d - L * np.round(d / L)


def displacements(obs: Observation) -> np.ndarray:
    """(T-1, B, N, 2) displacement between consecutive frames (requires k*v0 < L/2)."""
    return wrap(np.diff(obs.pos, axis=0), obs.L)
