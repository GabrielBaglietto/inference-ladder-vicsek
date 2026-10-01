"""Notebook 01: sensitivity of every summary to (η, r, v0) at four design points (40 replicas
each)."""

import _common  # noqa: F401

from flockladder.results import pack_sensitivity, save
from flockladder.sensitivity import sensitivity
from flockladder.vicsek import VicsekParams, box_for_density

RUN = {"k": 10, "n_frames": 501, "burn_in": 5000, "n_replicas": 40, "seed": 0}

if __name__ == "__main__":
    out = {}
    for rho in (0.4, 4.0):
        for eta in (1.0, 2.0):
            p = VicsekParams(eta=eta, n=40, L=box_for_density(40, rho))
            out |= pack_sensitivity(f"rho{rho}_eta{eta}", sensitivity(p, **RUN))
    print(save("a1_sensitivity", **out))
