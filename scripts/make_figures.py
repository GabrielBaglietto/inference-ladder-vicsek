"""The static figures of the README, drawn from results/ (plus one short simulation for the
observation figure). Run from the repository root: uv run python scripts/make_figures.py"""

from pathlib import Path

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from flockladder.observe import wrap
from flockladder.results import load, unpack_fits, unpack_sensitivity
from flockladder.sensitivity import table
from flockladder.synlik import SUMMARIES, TRUTH, shrunk_cov
from flockladder.vicsek import VicsekParams, simulate

FIG = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update(
    {"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150, "savefig.bbox": "tight"}
)
LABEL = {
    "speed": "apparent speed",
    "cos_turn": "turning angle",
    "corr_0_1": "velocity corr. (d < 1)",
    "corr_1_2": "velocity corr. (1 ≤ d < 2)",
    "phi": "polarization φ",
    "n_clusters": "number of clusters",
    "nn_dist": "nearest-neighbour distance",
}
PARAM = ("η (noise)", "r (radius)", "v₀ (speed)")


def plain_log_ticks(ax, x=None, y=None):
    """Label log axes with plain numbers at the given ticks, no minor labels."""
    for axis, ticks in ((ax.xaxis, x), (ax.yaxis, y)):
        if ticks is not None:
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
            axis.set_ticks(ticks)


def observation():
    """F2: what happens every step (grey) against what we record every 10 steps."""
    p = VicsekParams(eta=2.0, n=40, L=10.0)
    warm = simulate(p, 5000, 1, seed=11, record_every=5000)
    tr = simulate(p, 40, 1, seed=12, state=(warm.pos[-1], warm.heading[-1]))
    pos, hd = tr.pos[:, 0], tr.heading[:, 0]
    path = pos[0] + np.concatenate(
        [np.zeros((1, 40, 2)), np.cumsum(wrap(np.diff(pos, axis=0), p.L), axis=0)]
    )
    d0 = np.linalg.norm(wrap(pos[0] - pos[0, 0], p.L), axis=1)
    pick = np.argsort(d0)[:6]  # particle 0 and its five nearest neighbours
    ref = path[0, 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in pick:
        xy = path[:, i] - ref
        ax.plot(xy[:, 0], xy[:, 1], "-", c="0.75", lw=1)
        frames = xy[::10]
        ax.plot(frames[:, 0], frames[:, 1], "ko", ms=4)
        for a, b in zip(frames[:-1], frames[1:], strict=True):
            ax.annotate("", b, a, arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.2})
        for t in range(0, 41, 10):
            ax.arrow(
                *xy[t],
                0.08 * np.cos(hd[t, i]),
                0.08 * np.sin(hd[t, i]),
                color="C3",
                width=0.004,
                head_width=0.025,
                length_includes_head=True,
            )
    ax.set(
        aspect="equal",
        xlabel="x (units of r)",
        ylabel="y (units of r)",
        title="40 steps of six neighbouring particles",
    )
    ax.plot([], [], "-", c="0.75", label="true path, every step (not observed)")
    ax.plot([], [], "ko-", ms=4, label="recorded positions, every 10 steps")
    ax.plot([], [], "-", c="C3", label="true heading αᵢ (not observed)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(FIG / "observation.png")


def sensitivity():
    """F3: z of each summary for each parameter at the design point."""
    t = table(unpack_sensitivity("rho0.4_eta2.0", load("a1_sensitivity")))
    z = np.array([[t[s][f"z_{q}"] for q in ("eta", "r", "v0")] for s in SUMMARIES])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.imshow(np.sign(z) * np.log10(1 + np.abs(z)), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(len(SUMMARIES)):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{z[i, j]:+.1f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if abs(z[i, j]) >= 2 else "normal",
            )
    ax.set_xticks(range(3), PARAM)
    ax.set_yticks(range(len(SUMMARIES)), [LABEL[s] for s in SUMMARIES])
    ax.set_title(
        "z: shift per 10% change, in run-to-run standard deviations\n(bold: |z| ≥ 2)", fontsize=11
    )
    fig.savefig(FIG / "sensitivity.png")


def half_time():
    """F4: half-time of the approach to order against the speed."""
    rel = load("a2_relaxation")
    v0s = (0.015, 0.03, 0.06)
    fig, ax = plt.subplots(figsize=(6, 4))
    for eta, mk in ((2.0, "o"), (1.0, "s")):
        for name, lab in (("phi", "alignment φ"), ("nn_dist", "geometry (n.n. distance)")):
            vals = []
            for v0 in v0s:
                pre = f"eta{eta}__" if v0 == 0.03 else f"eta{eta}_v0{v0}__"
                vals.append(rel[f"{pre}{name}_t_half"])
            m = [np.nanmean(v) for v in vals]
            se = [np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()) for v in vals]
            ax.errorbar(v0s, m, yerr=se, marker=mk, capsize=3, label=f"{lab}, η = {eta:g}")
    v = np.array(v0s)
    ax.loglog(v, 267 * 0.03 / v, "k:", lw=1, label="∝ 1/v₀")
    ax.set(
        xlabel="speed v₀",
        ylabel="half-time (steps)",
        title="How fast the flock orders, against the speed",
    )
    plain_log_ticks(ax, x=list(v0s), y=[50, 100, 200, 500, 1000])
    ax.legend(fontsize=8)
    fig.savefig(FIG / "half_time.png")


def likelihood_map():
    """F5: the synthetic likelihood on the (η, r) plane, and where the eight fits stopped."""
    a3 = load("a3_point_fit")
    z = a3["grid_ll"] - a3["grid_ll"].max()
    fits = unpack_fits("multistart", a3)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    cs = ax.contourf(a3["grid_eta"], a3["grid_r"], np.maximum(z, -400), levels=20, cmap="viridis")
    fig.colorbar(cs, label="log-likelihood − maximum (clipped at −400)")
    for f in fits:
        ax.plot(*f.theta[:2], "wo", mfc="none", ms=8)
    ax.plot(*TRUTH[:2], "r*", ms=15, label="truth")
    ax.plot([], [], "wo", mfc="none", label="where the 8 fits stopped")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="noise η",
        ylabel="radius r",
        title="Synthetic likelihood at v₀ = 0.03",
    )
    plain_log_ticks(ax, x=[1.5, 2, 2.5, 3], y=[0.6, 0.8, 1, 1.2, 1.4])
    leg = ax.legend(fontsize=8, facecolor="0.3", labelcolor="w")
    leg.get_frame().set_alpha(0.8)
    fig.savefig(FIG / "likelihood_map.png")


def ridge_two_panels():
    """F6: the likelihood on (η, r), alignment observables only against all seven summaries."""
    a4 = load("a4_identifiability")
    names10 = list(a4["grid_summary_names"])
    gs, obs = a4["grid_summaries"], a4["obs_all10"]

    def ell(i, j, names):
        idx = [names10.index(n) for n in names]
        s = gs[i, j][:, idx]
        c = shrunk_cov(s)[0]
        d = obs[idx] - s.mean(0)
        return -0.5 * d @ np.linalg.solve(c, d) - 0.5 * np.linalg.slogdet(c)[1]

    sets = (
        (
            "alignment observables only\n(φ, its fluctuations, corr. at 1–2 and 2–4)",
            ["phi", "chi", "corr_1_2", "corr_2_4"],
        ),
        ("all seven summaries", list(SUMMARIES)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (title, names) in zip(axes, sets, strict=True):
        z = np.array(
            [
                [ell(i, j, names) for j in range(len(a4["grid_eta"]))]
                for i in range(len(a4["grid_r"]))
            ]
        )
        z -= z.max()
        cs = ax.contourf(
            a4["grid_eta"],
            a4["grid_r"],
            np.maximum(z, -60),
            levels=np.linspace(-60, 0, 13),
            cmap="viridis",
        )
        ax.contour(
            a4["grid_eta"],
            a4["grid_r"],
            z,
            levels=[-3],
            colors="w",
            linewidths=1.2,
            linestyles="solid",
        )
        ax.plot(*TRUTH[:2], "r*", ms=14)
        ax.set(xscale="log", yscale="log", xlabel="noise η", title=title)
        plain_log_ticks(ax, x=[1.5, 2, 2.5, 3], y=[0.6, 0.8, 1, 1.2, 1.4])
    axes[0].set_ylabel("radius r")
    fig.colorbar(
        cs,
        ax=axes,
        label="log-likelihood − maximum (clipped at −60)\nwhite line: 3 below the maximum",
    )
    fig.savefig(FIG / "ridge_two_panels.png")


def posterior():
    """F8: the posterior, as pairs of parameters (six chains, burn-in removed)."""
    x = np.exp(load("a6_posterior")["x"][:, 501:].reshape(-1, 3))
    pairs = ((0, 1), (0, 2), (1, 2))
    names = ("noise η", "radius r", "speed v₀")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (i, j) in zip(axes, pairs, strict=True):
        ax.plot(x[::4, i], x[::4, j], ".", ms=1.5, alpha=0.35, c="C0")
        ax.plot(TRUTH[i], TRUTH[j], "r*", ms=14, label="truth")
        ax.set(xlabel=names[i], ylabel=names[j])
    axes[0].legend(fontsize=9)
    fig.suptitle("Posterior samples (six MCMC chains)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "posterior.png")


if __name__ == "__main__":
    for f in (observation, sensitivity, half_time, likelihood_map, ridge_two_panels, posterior):
        f()
        print("wrote", f.__name__)
