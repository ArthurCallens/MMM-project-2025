"""Turn a sequence of stored simulation frames into a playable animation, for the
Streamlit app's video players and the Report tab.

Two backends are tried, in order:
1. MP4 (H.264) via a bundled static ffmpeg binary (imageio-ffmpeg) -- best UX
   (scrubbing, pause, native `st.video` player), but requires spawning a subprocess,
   which some restrictive hosting sandboxes disallow.
2. Animated GIF via Pillow only (no subprocess, no external binary at all) -- works
   anywhere Python/Pillow works, autoplays in the browser via `st.image`.

Every public function returns `(bytes, mime_type)` so the caller can pick the right
Streamlit widget (`st.video` for "video/mp4", `st.image` for "image/gif") without needing
to know which backend actually succeeded.
"""
from __future__ import annotations

import io
import logging
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _encode_mp4(rgb_frames: list[np.ndarray], fps: int) -> bytes:
    import imageio.v2 as imageio

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    writer = imageio.get_writer(path, format="FFMPEG", fps=fps, codec="libx264",
                                 quality=7, macro_block_size=1, pixelformat="yuv420p")
    try:
        for rgb in rgb_frames:
            writer.append_data(rgb)
    finally:
        writer.close()
    with open(path, "rb") as f:
        return f.read()


def _encode_gif(rgb_frames: list[np.ndarray], fps: int) -> bytes:
    images = [Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=200) for rgb in rgb_frames]
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                    duration=int(1000 / fps), loop=0, optimize=True)
    return buf.getvalue()


def _even(rgb: np.ndarray) -> np.ndarray:
    """H.264/yuv420p requires even width and height; crop by (at most) one row/column
    of pixels if matplotlib's rendered canvas came out odd (it very often does, since
    figsize*dpi rarely lands on an even integer by chance)."""
    h, w = rgb.shape[:2]
    return rgb[: h - (h % 2), : w - (w % 2)]


def _encode(rgb_frames: list[np.ndarray], fps: int) -> tuple[bytes, str]:
    rgb_frames = [_even(f) for f in rgb_frames]
    try:
        return _encode_mp4(rgb_frames, fps), "video/mp4"
    except Exception as exc:  # pragma: no cover - depends on host sandboxing
        log.warning("MP4 (ffmpeg) encoding failed (%s); falling back to animated GIF.", exc)
        return _encode_gif(rgb_frames, fps), "image/gif"


def field_frames_to_video(frames, xs, ys, title, cmap="RdBu_r", symmetric=True, unit="nm",
                           fps=10, mark_fn=None, dpi=90, figsize=(4.6, 3.9), hold_last=True) -> tuple[bytes, str]:
    """Render a list of 2-D field snapshots into a playable animation.

    Returns `(bytes, mime_type)` -- pass `mime_type == "video/mp4"` results to
    `st.video(...)` and `"image/gif"` results to `st.image(...)`.
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

    rgb_frames = []
    try:
        for i, frame in enumerate(frames):
            im.set_data(frame.T)
            title_artist.set_text(f"{title}  (frame {i + 1}/{len(frames)})")
            fig.canvas.draw()
            rgb_frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if hold_last:
            for _ in range(max(1, fps // 2)):
                rgb_frames.append(rgb_frames[-1])
    finally:
        plt.close(fig)

    return _encode(rgb_frames, fps)


def line_series_to_video(t, series: dict, xlabel, ylabel, title, fps=15, dpi=90,
                          figsize=(6.0, 3.6)) -> tuple[bytes, str]:
    """Render a growing line plot (e.g. an expectation value catching up to the current
    time) as a playable animation. Returns `(bytes, mime_type)` -- see field_frames_to_video."""
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

    rgb_frames = []
    try:
        for k in idxs:
            for label, ys in series.items():
                lines[label].set_data(t[:k], np.asarray(ys)[:k])
            fig.canvas.draw()
            rgb_frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        for _ in range(max(1, fps // 2)):
            rgb_frames.append(rgb_frames[-1])
    finally:
        plt.close(fig)

    return _encode(rgb_frames, fps)
