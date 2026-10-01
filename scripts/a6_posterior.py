"""Notebook 06: six pseudo-marginal uBSL chains of 4000 iterations (n = 30 simulations per
estimate).

From scratch this is ~12 h on 6 cores. Every likelihood evaluation goes through the disk cache,
so an interrupted run resumes by replaying."""

import _common  # noqa: F401
import numpy as np
from _common import THETA_HAT

from flockladder.fisher import fisher, local_info
from flockladder.results import save
from flockladder.sensitivity import sensitivity
from flockladder.synlik import SUMMARIES, observed
from flockladder.ubsl import run_chains
from flockladder.vicsek import VicsekParams

N_SIM, ITERS, CHAINS = 30, 4000, 6

if __name__ == "__main__":
    sens = sensitivity(
        VicsekParams(eta=2.0, r=1.0, v0=0.03, n=40, L=10.0),
        k=10,
        n_frames=501,
        burn_in=5000,
        n_replicas=200,
        seed=0,
    )
    li = local_info(sens, list(SUMMARIES))
    chol = np.linalg.cholesky(np.linalg.inv(fisher(li["V"], li["D"])))
    s_obs = observed()
    jobs = []
    for c in range(CHAINS):
        z = np.random.default_rng(c).standard_normal(3)
        x0 = np.log(np.array(THETA_HAT)) + chol @ z
        jobs.append(
            {"s_obs": s_obs, "x0": x0, "prop_chol": chol, "n": N_SIM, "iters": ITERS, "chain_id": c}
        )
    out = run_chains(jobs, workers=CHAINS)
    print(
        save(
            "a6_posterior",
            x=np.array([o["x"] for o in out]),
            ll=np.array([o["ll"] for o in out]),
            acc=np.array([o["acc"] for o in out]),
        )
    )
