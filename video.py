"""Turn a sequence of stored simulation frames into a playable MP4, for the Streamlit
app's video players and the Report tab. Uses a bundled static ffmpeg binary
(via imageio-ffmpeg) so it works with no system-level ffmpeg install.
"""
from __future__ import annotations

import tempfile

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def field_frames_to_mp4(frames, xs, ys, title, cmap="RdBu_r", symmetric=True, unit="nm",
                         fps=10, mark_fn=None, dpi=110, figsize=(5.0, 4.2)) -> bytes:
    """Render a list of 2-D field snapshots (same shape as `frames[0]`) into an MP4.

    `frames`: list of 2-D numpy arrays (all the same shape).
    `xs`, `ys`: coordinate arrays (metres) matching the frame's axes, used for the extent.
    `mark_fn(ax)`: optional callback to draw extra static annotations (scatterer outlines,
    well markers, ...) on every frame.
    """
    if not frames:
        raise ValueError("no frames to render")
    x0, x1 = xs[0] / 1e-9, xs[-1] / 1e-9
    y0, y1 = ys[0] / 1e-9, ys[-1] / 1e-9
    vmax = max(float(np.abs(f).max()) for f in frames) or 1.0
    kwargs = dict(vmin=-vmax, vmax=vmax) if symmetric else dict(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(frames[0].T, origin="lower", extent=[x0, x1, y0, y1], cmap=cmap,
                    aspect="equal", animated=True, **kwargs)
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    title_artist = ax.set_title(title)
    if mark_fn is not None:
        mark_fn(ax)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    writer = imageio.get_writer(path, format="FFMPEG", fps=fps, codec="libx264",
                                 quality=7, macro_block_size=1, pixelformat="yuv420p")
    try:
        for i, frame in enumerate(frames):
            im.set_data(frame.T)
            title_artist.set_text(f"{title}  (frame {i + 1}/{len(frames)})")
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(rgb)
        # hold on the last frame for a beat so playback doesn't feel cut off
        for _ in range(max(1, fps // 2)):
            writer.append_data(rgb)
    finally:
        writer.close()
        plt.close(fig)

    with open(path, "rb") as f:
        return f.read()


def line_series_to_mp4(t, series: dict, xlabel, ylabel, title, fps=15, dpi=110,
                        figsize=(6.0, 3.6), window=None) -> bytes:
    """Render a growing line/scatter plot (e.g. an expectation value catching up to the
    current time) as an MP4 -- a small animated "value vs. time" video."""
    t = np.asarray(t)
    n = len(t)
    if n == 0:
        raise ValueError("no data to render")
    n_frames = min(n, 90)
    idxs = np.linspace(1, n, n_frames).astype(int)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    lines = {}
    for label in series:
        (line,) = ax.plot([], [], lw=1.6, label=label)
        lines[label] = line
    ax.set_xlim(t.min(), t.max())
    all_y = np.concatenate([np.asarray(v) for v in series.values()])
    pad = 0.05 * (all_y.max() - all_y.min() + 1e-30)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    writer = imageio.get_writer(path, format="FFMPEG", fps=fps, codec="libx264",
                                 quality=7, macro_block_size=1, pixelformat="yuv420p")
    try:
        for k in idxs:
            for label, ys in series.items():
                lines[label].set_data(t[:k], np.asarray(ys)[:k])
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(rgb)
        for _ in range(max(1, fps // 2)):
            writer.append_data(rgb)
    finally:
        writer.close()
        plt.close(fig)

    with open(path, "rb") as f:
        return f.read()
