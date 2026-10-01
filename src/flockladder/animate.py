"""MP4 videos of simulated trajectories, one panel per trajectory, side by side.

Particles are arrows coloured by heading (cyclic colormap), so aligned groups share a colour, or
(``color="cluster"``) by the size of the cluster they belong to, with linkage distance ``ell``.
Arrows show heading only, with a fixed length of ARROW = 0.6 in units of r (not v₀). Each panel
has a scale bar of length 1, the unit of length (the reference r* = 1; the panel's own r is in
the parameter line) and a line with all the parameters under it.
Videos go to ``media/`` (not versioned); ``show`` embeds one in a notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Video
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from flockladder.observables import cluster_sizes, polarization
from flockladder.vicsek import Trajectory, VicsekParams

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

MEDIA = Path(__file__).resolve().parents[2] / "media"
ARROW = 0.6  # drawn arrow length, in units of r


def param_line(p: VicsekParams) -> str:
    """All model parameters in one line, in units r = 1 and one step."""
    return (
        f"N={p.n}  L={p.L:.3g}  ρ={p.density:.3g}  r={p.r:.3g}  v₀={p.v0:.3g}  "
        f"η={p.eta:.3g}  {p.update}"
    )


def animate(
    trajectories: Trajectory | list[Trajectory],
    name: str,
    titles: list[str] | None = None,
    replica: int = 0,
    fps: int = 25,
    panel_size: float = 4.0,
    color: Literal["heading", "cluster"] = "heading",
    ell: float = 1.0,
    gif: Path | None = None,
    stride: int = 1,
    dpi: int = 70,
) -> Path:
    """Render ``replica`` of each trajectory as a panel and save ``media/<name>.mp4``.

    With ``gif`` set, write an animated GIF to that path instead (every ``stride``-th frame, at
    ``dpi``), small enough to show inside a README. All trajectories must have the same recorded
    steps. Returns the path of the file written.
    """
    trs = [trajectories] if isinstance(trajectories, Trajectory) else trajectories
    titles = titles or [f"η={t.params.eta:.2f}, r={t.params.r:g}" for t in trs]
    n_frames = len(trs[0].steps)

    def values(tr, f):
        if color == "heading":
            return tr.heading[f, replica]
        return cluster_sizes(tr.pos[f, replica], tr.params.L, ell)

    def colormap(tr):
        if color == "heading":
            return {"cmap": "hsv", "clim": (-np.pi, np.pi)}
        return {"cmap": "viridis", "clim": (1, tr.params.n)}

    fig, axes = plt.subplots(1, len(trs), figsize=(panel_size * len(trs), panel_size + 0.7))
    axes = np.atleast_1d(axes)
    arrows, labels = [], []
    for ax, tr, title in zip(axes, trs, titles, strict=True):
        L, x, th = tr.params.L, tr.pos[0, replica], tr.heading[0, replica]
        q = ax.quiver(
            x[:, 0],
            x[:, 1],
            np.cos(th),
            np.sin(th),
            values(tr, 0),
            **colormap(tr),
            angles="xy",
            scale_units="xy",
            scale=1 / ARROW,
            width=0.006,
            pivot="middle",
        )
        ax.set(xlim=(0, L), ylim=(0, L), aspect="equal", xticks=[], yticks=[], title=title)
        ax.set_xlabel(param_line(tr.params), fontsize=8)
        ax.add_artist(
            AnchoredSizeBar(
                ax.transData,
                1.0,
                "1 (unit)",
                loc="lower left",
                pad=0.3,
                sep=3,
                size_vertical=L / 150,
                frameon=True,
                fontproperties={"size": 8},
            )
        )
        if color == "cluster":
            fig.colorbar(q, ax=ax, shrink=0.7, label=f"cluster size (linkage {ell:g})")
        arrows.append(q)
        labels.append(ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9))
    fig.tight_layout()

    def draw(f):
        for q, label, tr in zip(arrows, labels, trs, strict=True):
            x, th = tr.pos[f, replica], tr.heading[f, replica]
            q.set_offsets(x)
            q.set_UVC(np.cos(th), np.sin(th), values(tr, f))
            label.set_text(f"t={tr.steps[f]}  φ={polarization(th):.2f}")
        return arrows + labels

    frames = range(0, n_frames, stride)
    anim = FuncAnimation(fig, draw, frames=frames, blit=True)
    if gif is not None:
        gif.parent.mkdir(parents=True, exist_ok=True)
        anim.save(gif, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        return gif
    MEDIA.mkdir(exist_ok=True)
    path = MEDIA / f"{name}.mp4"
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=900))
    plt.close(fig)
    return path


def show(path: Path, width: int = 900) -> Video | Image:
    """Embed a video (MP4) or an animated GIF in the notebook output."""
    if Path(path).suffix == ".gif":
        return Image(filename=str(path), width=width)
    return Video(str(path), embed=True, width=width)
