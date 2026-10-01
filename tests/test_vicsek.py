import numpy as np
import pytest

from flockladder.observables import polarization
from flockladder.vicsek import VicsekParams, box_for_density, simulate


@pytest.mark.parametrize("update", ["angle_first", "move_first"])
def test_same_seed_same_trajectory(update):
    p = VicsekParams(eta=1.0, update=update)
    a = simulate(p, 50, n_replicas=3, seed=7)
    b = simulate(p, 50, n_replicas=3, seed=7)
    np.testing.assert_array_equal(a.pos, b.pos)
    np.testing.assert_array_equal(a.heading, b.heading)


def test_positions_stay_in_box_and_headings_wrapped():
    p = VicsekParams(eta=2.0, v0=0.5, L=2.0)
    tr = simulate(p, 200, n_replicas=4, seed=1)
    assert tr.pos.min() >= 0.0 and tr.pos.max() < p.L
    assert np.all(np.abs(tr.heading) <= np.pi)


@pytest.mark.parametrize("update", ["angle_first", "move_first"])
def test_constant_speed_along_heading(update):
    p = VicsekParams(eta=1.0, v0=0.1, update=update)
    tr = simulate(p, 20, n_replicas=2, seed=3)
    d = np.diff(tr.pos, axis=0)
    d -= p.L * np.round(d / p.L)
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), p.v0)
    # angle_first moves with the heading chosen in the same step; move_first with the previous one
    heading = tr.heading[1:] if update == "angle_first" else tr.heading[:-1]
    np.testing.assert_allclose(np.arctan2(d[..., 1], d[..., 0]), heading, atol=1e-9)


def test_no_neighbours_no_noise_keeps_headings():
    p = VicsekParams(eta=0.0, r=0.0)
    tr = simulate(p, 30, n_replicas=2, seed=4)
    np.testing.assert_allclose(tr.heading[-1], tr.heading[0], atol=1e-12)


def test_noiseless_dense_system_orders():
    p = VicsekParams(eta=0.0, n=40, L=box_for_density(40, 4.0))
    tr = simulate(p, 400, n_replicas=5, seed=5)
    assert polarization(tr.heading[-1]).min() > 0.99


def test_full_noise_gives_random_polarization():
    n = 40
    p = VicsekParams(eta=2 * np.pi, n=n)
    tr = simulate(p, 5, n_replicas=4000, seed=6)
    phi = polarization(tr.heading[-1]).mean()
    assert phi == pytest.approx(np.sqrt(np.pi / (4 * n)), rel=0.05)
