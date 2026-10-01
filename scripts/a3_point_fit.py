"""Notebook 03 (and its second attempt): objective slices, multistart fits, Monte Carlo re-fits,
the (η, r) grid, restarts, the pilot regression and the two arms of the second attempt."""

import _common  # noqa: F401
import numpy as np

from flockladder.results import pack_fits, save
from flockladder.synlik import (
    BOX,
    TRUTH,
    fit_many,
    latin_hypercube,
    observed,
    pilot,
    summary_matrix,
    synlik_many,
)

if __name__ == "__main__":
    s_obs = observed()
    out: dict[str, np.ndarray] = {"s_obs": s_obs}
    factors = np.exp(np.linspace(np.log(0.7), np.log(1.3), 7))
    pts, fresh_seeds = [], []
    for j in range(3):
        for i, f in enumerate(factors):
            th = TRUTH.copy()
            th[j] *= f
            pts.append(th)
            fresh_seeds.append(100 + 7 * j + i)
    out["slice_factors"] = factors
    out["slice_crn"] = synlik_many(pts, s_obs, [0] * len(pts)).reshape(3, 7)
    out["slice_fresh"] = synlik_many(pts, s_obs, fresh_seeds).reshape(3, 7)
    out["ll_at_truth"] = synlik_many([TRUTH] * 8, s_obs, list(range(200, 208)))
    out["slice_summaries"] = np.array([summary_matrix(t, 40, 0) for t in pts])  # (21, 40, d)
    out["summaries_at_truth"] = summary_matrix(TRUTH, 40, 0)

    starts = latin_hypercube(8, np.random.default_rng(3))
    fits = fit_many(s_obs, list(starts), [0] * 8)
    out |= pack_fits("multistart", fits)
    best = max(fits, key=lambda f: f.loglik)
    out |= pack_fits("mc", fit_many(s_obs, [best.theta] * 4, [1, 2, 3, 4]))
    out |= pack_fits("restart", fit_many(s_obs, [f.theta for f in fits], [0] * 8))

    g_eta = np.exp(np.linspace(np.log(1.2), np.log(3.4), 11))
    g_r = np.exp(np.linspace(np.log(0.55), np.log(1.6), 11))
    grid = [np.array([e, r, 0.03]) for r in g_r for e in g_eta]
    out["grid_eta"], out["grid_r"] = g_eta, g_r
    out["grid_ll"] = synlik_many(grid, s_obs, [0] * len(grid)).reshape(len(g_r), len(g_eta))

    pl = pilot()
    th_tilde = pl.predict(s_obs)[0]
    out |= {"pilot_theta": th_tilde, "pilot_resid_sd": pl.resid_sd}
    region = np.clip(
        np.stack([th_tilde * np.exp(-2 * pl.resid_sd), th_tilde * np.exp(2 * pl.resid_sd)], 1),
        BOX[:, :1],
        BOX[:, 1:],
    )
    starts_a = latin_hypercube(8, np.random.default_rng(3))
    starts_b = latin_hypercube(8, np.random.default_rng(5), box=region)
    arms = fit_many(s_obs, list(starts_a) + list(starts_b), [0] * 16, step=pl.resid_sd, shrink=True)
    out |= pack_fits("arm_a", arms[:8]) | pack_fits("arm_b", arms[8:])
    print(save("a3_point_fit", **out))
