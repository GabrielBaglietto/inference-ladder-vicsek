"""Notebook 06, §1: the pilot that chose n. At θ̂, the sd of log p̂ (uBSL) over 10 fresh seeds for
n ∈ {20, 30, 40}; the rule (written before looking) takes the smallest n with sd ≤ 2."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import _common  # noqa: F401
import numpy as np
from _common import THETA_HAT

from flockladder.results import save
from flockladder.synlik import observed
from flockladder.ubsl import loglik


def one(args):
    n, seed = args
    return loglik(np.array(THETA_HAT), observed(), n, seed)


if __name__ == "__main__":
    jobs = [(n, 9_000_000 + 100 * n + i) for n in (20, 30, 40) for i in range(10)]
    with ProcessPoolExecutor(6, mp_context=get_context("spawn")) as ex:
        ll = np.array(list(ex.map(one, jobs))).reshape(3, 10)
    print(save("a6_pilot", n=np.array([20, 30, 40]), loglik=ll))
