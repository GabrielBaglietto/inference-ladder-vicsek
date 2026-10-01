import numpy as np
import pytest

from flockladder.observables import cluster_sizes, series, summaries
from flockladder.observe import Observation, displacements, observe
from flockladder.stats import tau_int
from flockladder.vicsek import VicsekParams, simulate


def test_observation_matches_simulation_every_k_steps():
    p = VicsekParams(eta=1.0, L=4.0)
    obs = observe(p, n_frames=6, k=5, n_replicas=2, seed=3)
    tr = simulate(p, 25, n_replicas=2, seed=3)
    np.testing.assert_array_equal(obs.pos, tr.pos[::5])


def test_displacements_undo_periodic_wrapping():
    p = VicsekParams(eta=0.0, r=0.0, v0=0.2, L=3.0)
    obs = observe(p, n_frames=20, k=3, n_replicas=2, seed=1)
    # no interaction, no noise: straight lines, each displacement is exactly k*v0 long
    np.testing.assert_allclose(np.linalg.norm(displacements(obs), axis=-1), 3 * 0.2)


def test_k1_straight_motion_gives_unit_turning_and_true_speed():
    p = VicsekParams(eta=0.0, r=0.0, v0=0.05)
    s = series(observe(p, n_frames=10, k=1, n_replicas=2, seed=2))
    np.testing.assert_allclose(s["cos_turn"], 1.0)
    np.testing.assert_allclose(s["speed"], 0.05)


def test_sampled_speed_is_at_most_v0():
    p = VicsekParams(eta=3.0, v0=0.03)
    s = series(observe(p, n_frames=30, k=10, n_replicas=3, seed=4))
    assert np.all(s["speed"] <= p.v0 + 1e-12)
    assert s["speed"].mean() < 0.9 * p.v0  # turning shortens net displacement


def test_clusters_on_a_hand_made_frame():
    x = np.array([[0.1, 0.1], [0.5, 0.1], [3.9, 0.1], [2.0, 2.0]])  # 0-1 close, 2 wraps to 0
    np.testing.assert_array_equal(cluster_sizes(x, L=4.0, ell=0.5), [3, 3, 3, 1])


def test_summaries_shapes_and_ranges():
    p = VicsekParams(eta=1.0)
    obs = observe(p, n_frames=15, k=4, n_replicas=3, seed=5)
    summ = summaries(series(obs), n=p.n)
    for name, v in summ.items():
        assert v.shape == (3,), name
    assert np.all((summ["phi"] >= 0) & (summ["phi"] <= 1))
    assert np.all((summ["largest_cluster"] > 0) & (summ["largest_cluster"] <= 1))


def test_tau_int_of_ar1():
    rng = np.random.default_rng(0)
    a, x = 0.8, np.zeros((20000, 4))
    for t in range(1, len(x)):
        x[t] = a * x[t - 1] + rng.normal(size=4)
    assert tau_int(x) == pytest.approx((1 + a) / (1 - a), rel=0.15)


def test_observation_is_positions_only():
    assert set(Observation.__dataclass_fields__) == {"pos", "k", "L"}


def test_gap_rate_recovers_exponential():
    from flockladder.rates import gap_rate

    t = np.arange(0, 6001, 10.0)
    y = 0.8 - 0.6 * np.exp(-t / 500)
    f = gap_rate(t, y)
    assert f["lam"] == pytest.approx(1 / 500, rel=1e-3)  # plateau still holds the e^-8 tail
    assert f["single"] == 1.0
    assert f["t_half"] == pytest.approx(500 * np.log(2), abs=10)


def test_gap_rate_flags_two_timescales():
    from flockladder.rates import gap_rate

    t = np.arange(0, 6001, 10.0)
    y = 1 - 0.5 * np.exp(-t / 50) - 0.5 * np.exp(-t / 1000)
    assert gap_rate(t, y)["single"] == 0.0


def test_vectorised_components_match_scipy():
    from flockladder.observables import _pair_distances, cluster_labels, component_labels

    rng = np.random.default_rng(8)
    x = rng.uniform(0, 6, size=(50, 40, 2))
    ours = component_labels(_pair_distances(x, 6.0) < 1.0)
    for b in range(len(x)):
        ref = cluster_labels(x[b], 6.0, 1.0)
        # same partition: two particles share a label in one iff they do in the other
        np.testing.assert_array_equal(ours[b][:, None] == ours[b], ref[:, None] == ref)


def test_synlik_is_minus_inf_outside_box_and_lhs_inside():
    from flockladder.synlik import BOX, latin_hypercube, synlik

    assert synlik(np.log([10.0, 1.0, 0.03]), np.zeros(7)) == -np.inf
    pts = latin_hypercube(16, np.random.default_rng(0))
    assert np.all((pts >= BOX[:, 0]) & (pts <= BOX[:, 1]))


def test_shrunk_cov_keeps_variances_and_shrinks_correlations():
    from flockladder.synlik import shrunk_cov

    rng = np.random.default_rng(1)
    S = rng.normal(size=(40, 5)) @ rng.normal(size=(5, 5))
    C, lam = shrunk_cov(S)
    np.testing.assert_allclose(np.diag(C), S.var(axis=0, ddof=1))
    assert 0 <= lam <= 1
    R0, R1 = np.corrcoef(S, rowvar=False), C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    off = ~np.eye(5, dtype=bool)
    np.testing.assert_allclose(R1[off], (1 - lam) * R0[off], atol=1e-12)


def test_fisher_recovers_linear_gaussian_model():
    from flockladder.fisher import fisher, summarize

    rng = np.random.default_rng(2)
    J = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0], [1.0, 1.0, 1.0]])
    B = 200000
    V = rng.normal(size=(B, 4))
    D = np.broadcast_to(J, (B, 4, 3))
    s = summarize(fisher(V, D))
    np.testing.assert_allclose(s["se"], np.sqrt(np.diag(np.linalg.inv(J.T @ J))), rtol=0.02)


def test_whitened_fit_maps_back_to_log_theta(monkeypatch):
    import flockladder.synlik as sl

    target = np.log(np.array([2.2, 0.9, 0.031]))
    monkeypatch.setattr(
        sl, "synlik", lambda x, *a, **k: -0.5 * float(((x - target) / 0.01) @ ((x - target) / 0.01))
    )
    L = np.diag([0.02, 0.03, 0.005])
    f = sl.fit(np.zeros(7), np.array([2.0, 1.0, 0.03]), whiten=L, step=2.0, xatol=0.01, maxfev=2000)
    np.testing.assert_allclose(np.log(f.theta), target, atol=2e-3)


def test_quadratic_max_recovers_known_surface():
    from flockladder.synlik import quadratic_max

    rng = np.random.default_rng(3)
    H = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 3.0]])
    y0 = np.array([0.3, -0.7, 0.4])
    Y = rng.uniform(-3, 3, size=(500, 3))
    ell = 5.0 - 0.5 * np.einsum("ni,ij,nj->n", Y - y0, H, Y - y0) + rng.normal(0, 0.05, 500)
    ystar, Hf, sd = quadratic_max(Y, ell)
    np.testing.assert_allclose(ystar, y0, atol=0.02)
    np.testing.assert_allclose(Hf, H, atol=0.02)
    assert sd < 0.07


def test_ubsl_is_unbiased_for_gaussian_draws():
    from scipy.stats import multivariate_normal

    from flockladder.ubsl import log_ubsl

    rng = np.random.default_rng(4)
    d, n = 3, 12
    mu = np.array([0.5, -1.0, 2.0])
    A = rng.normal(size=(d, d))
    cov = A @ A.T + np.eye(d)
    s = mu + np.array([0.8, 0.3, -0.5])
    est = np.array(
        [np.exp(log_ubsl(rng.multivariate_normal(mu, cov, size=n), s)) for _ in range(40000)]
    )
    exact = multivariate_normal(mu, cov).pdf(s)
    assert est.mean() == pytest.approx(exact, rel=0.03)


def test_split_rhat_and_ess_on_iid_chains():
    from flockladder.ubsl import ess, split_rhat

    x = np.random.default_rng(5).normal(size=(4, 2000))
    assert split_rhat(x) == pytest.approx(1.0, abs=0.01)
    assert 0.7 * 8000 < ess(x) < 1.3 * 8000


def test_fit_and_sensitivity_packing_round_trip():
    from flockladder.results import pack_fits, pack_sensitivity, unpack_fits, unpack_sensitivity
    from flockladder.synlik import Fit

    fits = []
    for k in (2, 5):
        f = Fit(start=np.array([1.0, 2.0, 3.0]), seed=0)
        f.path = [(np.array([1.0, 1.0, 1.0]) * t, -float(t) if t else -np.inf) for t in range(k)]
        f.theta, f.loglik, f.nfev = np.array([4.0, 5.0, 6.0]), -1.5, k
        fits.append(f)
    back = unpack_fits("x", pack_fits("x", fits))
    assert [len(f.path) for f in back] == [2, 5]
    assert back[0].path[0][1] == -np.inf and back[1].loglik == -1.5
    sens = {"phi": {"value": np.arange(3.0), "d_eta": np.ones(3)}}
    assert unpack_sensitivity("p", pack_sensitivity("p", sens))["phi"]["d_eta"].tolist() == [
        1,
        1,
        1,
    ]
