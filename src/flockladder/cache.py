"""Disk cache for expensive functions that return ``dict[str, ndarray]``.

Keyed by the function name and the ``repr`` of its arguments (dataclass params have a stable repr).
Files go to ``data/cache/`` (not versioned), or to ``$FLOCKLADDER_CACHE`` if set. Delete that
folder after changing the simulator.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import numpy as np

CACHE = Path(
    os.environ.get("FLOCKLADDER_CACHE", Path(__file__).resolve().parents[2] / "data" / "cache")
)


def disk_cache(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        path = CACHE / f"{fn.__name__}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
        if path.exists():
            with np.load(path) as f:
                return dict(f)
        out = fn(*args, **kwargs)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
        return out

    return wrapper
