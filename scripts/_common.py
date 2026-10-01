"""Shared settings for the scripts that build ``results/``. Run each script with ``uv run
python``."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

THETA_HAT = (
    2.0035,
    1.0141,
    0.03001,
)  # the best point fit of notebook 03 (rounded as used downstream)
RUN40 = {"k": 10, "n_frames": 501, "burn_in": 5000}
