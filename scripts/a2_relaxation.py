"""Notebook 02: relaxation from a random start (8 groups of 5 replicas), its sensitivity, and v0 ×
½, 2."""

import _common  # noqa: F401

from flockladder.rates import run_relaxation
from flockladder.results import pack_sensitivity, save
from flockladder.sensitivity import sensitivity
from flockladder.vicsek import VicsekParams

if __name__ == "__main__":
    out = {}
    for eta in (2.0, 1.0):
        p = VicsekParams(eta=eta, n=40, L=10.0)
        out |= {f"eta{eta}__{k}": v for k, v in run_relaxation(p).items()}
        out |= pack_sensitivity(f"sens_eta{eta}", sensitivity(p, runner=run_relaxation))
        for v0 in (0.015, 0.06):
            q = VicsekParams(eta=eta, n=40, L=10.0, v0=v0)
            out |= {f"eta{eta}_v0{v0}__{k}": v for k, v in run_relaxation(q).items()}
    print(save("a2_relaxation", **out))
