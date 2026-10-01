"""Notebook 05 (and its second attempt): parametric bootstrap refits in whitened coordinates,
the start-bias control, and the data-independent Gaussian terms at every evaluated point (for the
response surface)."""

import _common  # noqa: F401
import numpy as np
from _common import THETA_HAT

from flockladder.fisher import fisher, local_info
from flockladder.results import pack_fits, save
from flockladder.sensitivity import sensitivity
from flockladder.synlik import SUMMARIES, fit_jobs, gaussian_terms, observed
from flockladder.vicsek import VicsekParams

if __name__ == "__main__":
    th_hat = np.array(THETA_HAT)
    sens = sensitivity(
        VicsekParams(eta=2.0, r=1.0, v0=0.03, n=40, L=10.0),
        k=10,
        n_frames=501,
        burn_in=5000,
        n_replicas=200,
        seed=0,
    )
    li = local_info(sens, list(SUMMARIES))
    cov = np.linalg.inv(fisher(li["V"], li["D"]))
    chol = np.linalg.cholesky(cov)
    data = [observed(2_000_000 + i, th_hat) for i in range(30)]
    base = {"start": th_hat, "seed": 0, "n_replicas": 40, "maxfev": 80, "step": 2.0}
    base |= {"whiten": chol, "xatol": 0.1}
    y0 = np.array([2.0, -2.0, 2.0])
    jobs = [base | {"s_obs": s} for s in data] + [base | {"s_obs": s, "y0": y0} for s in data[:5]]
    fits = fit_jobs(jobs)
    pts = np.unique(np.array([p[0] for f in fits for p in f.path]), axis=0)
    terms = [gaussian_terms(t) for t in pts]
    out = pack_fits("main", fits[:30]) | pack_fits("control", fits[30:])
    out |= {
        "data": np.array(data),
        "s_obs": observed(),
        "fisher_cov": cov,
        "points": pts,
        "terms_mu": np.array([t[0] for t in terms]),
        "terms_prec": np.array([t[1] for t in terms]),
        "terms_logdet": np.array([t[2] for t in terms]),
    }
    print(save("a5_bootstrap", **out))
