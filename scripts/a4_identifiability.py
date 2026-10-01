"""Notebook 04: Fisher ingredients (200 replicas) at three densities, the ±3% check, the profile
of r, the summaries on notebook 03's (η, r) grid, and the pair of flocks along the φ-type ridge."""

import _common  # noqa: F401
import numpy as np

from flockladder.fisher import fisher, local_info, summarize
from flockladder.observables import run_summaries
from flockladder.results import pack_fits, pack_sensitivity, save
from flockladder.sensitivity import sensitivity
from flockladder.synlik import RUN, SUMMARIES, observed, params, profile_many
from flockladder.vicsek import VicsekParams, box_for_density

RUN200 = {"k": 10, "n_frames": 501, "burn_in": 5000, "n_replicas": 200, "seed": 0}

if __name__ == "__main__":
    out: dict[str, np.ndarray] = {}
    sens = {}
    for rho in (0.4, 1.0, 4.0):
        p = VicsekParams(eta=2.0, r=1.0, v0=0.03, n=40, L=box_for_density(40, rho))
        sens[rho] = sensitivity(p, **RUN200)
        out |= pack_sensitivity(f"rho{rho}", sens[rho])
    p04 = VicsekParams(eta=2.0, r=1.0, v0=0.03, n=40, L=10.0)
    out |= pack_sensitivity("rho0.4_rel0.03", sensitivity(p04, rel=0.03, **RUN200))

    li = local_info(sens[0.4], list(SUMMARIES))
    se = summarize(fisher(li["V"], li["D"]))["se"]
    rs = [0.8, 0.9, 1.0, 1.1, 1.25]
    prof = profile_many(observed(), rs, np.array([2.0035, 0.03001]), step=2 * se[[0, 2]], workers=5)
    out |= pack_fits("profile", prof) | {"profile_r": np.array(rs)}

    all10 = [*SUMMARIES, "corr_2_4", "largest_cluster", "chi"]
    g_eta = np.exp(np.linspace(np.log(1.2), np.log(3.4), 11))
    g_r = np.exp(np.linspace(np.log(0.55), np.log(1.6), 11))
    grid = np.array(
        [
            [
                np.stack(
                    [
                        run_summaries(
                            params(np.exp(np.log(np.array([e, r, 0.03])))),
                            n_replicas=40,
                            seed=0,
                            **RUN,
                        )[s]
                        for s in all10
                    ],
                    -1,
                )
                for e in g_eta
            ]
            for r in g_r
        ]
    )
    obs = run_summaries(params(np.array([2.0, 1.0, 0.03])), n_replicas=1, seed=10**6, **RUN)
    out |= {
        "grid_eta": g_eta,
        "grid_r": g_r,
        "grid_summaries": grid,  # (r, η, replica, summary)
        "grid_summary_names": np.array(all10),
        "obs_all10": np.array([obs[s][0] for s in all10]),
    }
    print(save("a4_identifiability", **out))
