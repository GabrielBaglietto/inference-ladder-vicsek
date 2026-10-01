"""Standard Vicsek model in a periodic box, vectorised over independent replicas.

Each particle moves at constant speed ``v0``. At every step it takes the mean heading of all
particles within distance ``r`` (itself included) plus uniform angular noise in
``[-eta/2, eta/2]`` (Vicsek et al. 1995).

Arrays carry a leading replica axis: positions are ``(B, N, 2)``, headings ``(B, N)``. With N of
order 20-100 all pairs are cheap, so no neighbour lists. The neighbour average is a compiled
(numba) loop; the noise is drawn from the numpy generator, so seeds mean the same as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np

UpdateRule = Literal["angle_first", "move_first"]


@dataclass(frozen=True)
class VicsekParams:
    eta: float
    """Noise amplitude; the angular noise is uniform in [-eta/2, eta/2]. 2*pi is full noise."""
    r: float = 1.0
    """Interaction radius."""
    v0: float = 0.03
    """Speed (distance per step)."""
    n: int = 40
    """Number of particles."""
    L: float = 5.0
    """Side of the periodic square box."""
    update: UpdateRule = "angle_first"
    """Order of the two sub-steps within one time step.

    - ``angle_first``: align on x(t), then move with the new heading α(t+1):
      x(t+1) = x(t) + v0 (cos α(t+1), sin α(t+1)).
    - ``move_first``: move with the old heading, then align on x(t+1).

    The two define different dynamics; the choice matters near the transition.
    """

    @property
    def density(self) -> float:
        return self.n / self.L**2


def box_for_density(n: int, density: float) -> float:
    """Box side L that gives ``n`` particles at the given density."""
    return float(np.sqrt(n / density))


def initial_state(
    p: VicsekParams, n_replicas: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random positions and headings."""
    pos = rng.uniform(0.0, p.L, size=(n_replicas, p.n, 2))
    heading = rng.uniform(-np.pi, np.pi, size=(n_replicas, p.n))
    return pos, heading


@numba.njit(cache=True)
def _aligned(pos: np.ndarray, heading: np.ndarray, noise: np.ndarray, L: float, r2: float):
    """Angle of the summed unit vectors of all particles within sqrt(r2) (self included), plus
    ``noise``, wrapped to [-pi, pi). Each pair is visited once and credited to both particles."""
    n_rep, n = heading.shape
    out = np.empty((n_rep, n))
    half = 0.5 * L
    s0, c0, ss, cc = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for b in range(n_rep):
        for i in range(n):
            s0[i], c0[i] = np.sin(heading[b, i]), np.cos(heading[b, i])
            ss[i], cc[i] = s0[i], c0[i]
        for i in range(n):
            xi, yi = pos[b, i, 0], pos[b, i, 1]
            for j in range(i + 1, n):
                dx = xi - pos[b, j, 0]
                dx = dx - L if dx > half else (dx + L if dx < -half else dx)  # minimum image
                dy = yi - pos[b, j, 1]
                dy = dy - L if dy > half else (dy + L if dy < -half else dy)
                if dx * dx + dy * dy <= r2:
                    ss[i] += s0[j]
                    cc[i] += c0[j]
                    ss[j] += s0[i]
                    cc[j] += c0[i]
        for i in range(n):
            a = np.arctan2(ss[i], cc[i]) + noise[b, i]
            out[b, i] = a - 2 * np.pi * np.floor((a + np.pi) / (2 * np.pi))
    return out


@numba.njit(cache=True)
def _moved(pos: np.ndarray, heading: np.ndarray, v0: float, L: float) -> np.ndarray:
    out = np.empty_like(pos)
    for b in range(heading.shape[0]):
        for i in range(heading.shape[1]):
            out[b, i, 0] = (pos[b, i, 0] + v0 * np.cos(heading[b, i])) % L
            out[b, i, 1] = (pos[b, i, 1] + v0 * np.sin(heading[b, i])) % L
    return out


def _align(pos: np.ndarray, heading: np.ndarray, p: VicsekParams, rng: np.random.Generator):
    noise = rng.uniform(-p.eta / 2, p.eta / 2, size=heading.shape)
    return _aligned(pos, heading, noise, p.L, p.r**2)


def _move(pos: np.ndarray, heading: np.ndarray, p: VicsekParams) -> np.ndarray:
    return _moved(pos, heading, p.v0, p.L)


def step(pos: np.ndarray, heading: np.ndarray, p: VicsekParams, rng: np.random.Generator):
    """Advance every replica by one time step. Returns new (pos, heading)."""
    if p.update == "angle_first":
        heading = _align(pos, heading, p, rng)
        pos = _move(pos, heading, p)
    else:
        pos = _move(pos, heading, p)
        heading = _align(pos, heading, p, rng)
    return pos, heading


@dataclass(frozen=True)
class Trajectory:
    pos: np.ndarray
    """(T, B, N, 2) positions at recorded steps."""
    heading: np.ndarray
    """(T, B, N) headings at recorded steps."""
    steps: np.ndarray
    """(T,) step index of each record; 0 is the initial state."""
    params: VicsekParams


def simulate(
    p: VicsekParams,
    n_steps: int,
    n_replicas: int = 1,
    seed: int | np.random.Generator | None = None,
    record_every: int = 1,
    state: tuple[np.ndarray, np.ndarray] | None = None,
) -> Trajectory:
    """Run ``n_replicas`` independent copies for ``n_steps`` steps.

    ``state`` continues from given (pos, heading) instead of a random start; the replica count is
    then taken from it.
    """
    rng = np.random.default_rng(seed)
    pos, heading = state if state is not None else initial_state(p, n_replicas, rng)
    rec_pos, rec_heading, rec_steps = [pos], [heading], [0]
    for t in range(1, n_steps + 1):
        pos, heading = step(pos, heading, p, rng)
        if t % record_every == 0:
            rec_pos.append(pos)
            rec_heading.append(heading)
            rec_steps.append(t)
    return Trajectory(np.stack(rec_pos), np.stack(rec_heading), np.array(rec_steps), p)
