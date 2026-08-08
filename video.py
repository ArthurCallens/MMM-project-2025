"""Turn a sequence of stored simulation frames into a playable animation, for the
Streamlit app's video players and the Report tab.

**Animated GIF (via Pillow) is the primary, guaranteed-to-render format.** It needs no
subprocess, no external binary, and no video-codec/browser negotiation at all -- a
browser plays an animated GIF the instant it displays the image, via plain `<img>`
semantics, which is about as close to "always works" as the web platform gets. This
matters because MP4 playback turned out to be unreliable across hosting environments
(sandboxed subprocess restrictions, and separately a libx264 "odd pixel dimensions"
failure mode) -- rather than keep chasing every possible host-specific MP4 failure mode,
GIF is used for on-screen display everywhere, with MP4 offered as a best-effort *download*
option when it happens to build successfully (smaller file, scrubbable player, for users
who save it and open it in a real video player).
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


def _even(rgb: np.ndarray) -> np.ndarray:
    """H.264/yuv420p requires even width and height; crop by (at most) one row/column
    of pixels if matplotlib's rendered canvas came out odd (it very often does, since
    figsize*dpi rarely lands on an even integer by chance)."""
    h, w = rgb.shape[:2]
    return rgb[: h - (h % 2), : w - (w % 2)]


def _render_rgb_frames(update_fn, n_frames: int, fig, hold_last: int) -> list[np.ndarray]:
    rgb_frames = []
    for i in range(n_frames):
        update_fn(i)
        fig.canvas.draw()
        rgb_frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    for _ in range(hold_last):
        rgb_frames.append(rgb_frames[-1])
    return rgb_frames


def _encode_gif(rgb_frames: list[np.ndarray], fps: int) -> bytes:
    """Encode as an animated GIF with a *shared* palette built from a sample spanning the
    whole animation (not just the first frame, which -- e.g. before a pulse has switched
    on -- is often blank/near-uniform: quantizing every later, more colourful frame down
    to a palette derived only from a blank first frame crushes all the real content
    toward that frame's colour, i.e. makes the whole animation look empty)."""
    n = len(rgb_frames)
    sample_idx = np.linspace(0, n - 1, min(10, n)).astype(int)
    composite = np.concatenate([rgb_frames[i] for i in sample_idx], axis=0)
    palette_img = Image.fromarray(composite).convert("P", palette=Image.ADAPTIVE, colors=256)

    images = [Image.fromarray(rgb).quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG) for rgb in rgb_frames]
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                    duration=int(1000 / fps), loop=0, optimize=False, disposal=2)
    return buf.getvalue()


def _encode_mp4(rgb_frames: list[np.ndarray], fps: int) -> bytes | None:
    """Best-effort MP4 encode; returns None (never raises) if it doesn't work in this
    environment (missing/blocked ffmpeg subprocess, codec issue, etc.)."""
    try:
        import imageio.v2 as imageio

        frames = [_even(f) for f in rgb_frames]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            path = tmp.name
        writer = imageio.get_writer(
            path, format="FFMPEG", fps=fps, codec="libx264", quality=7,
            macro_block_size=1, pixelformat="yuv420p",
            output_params=["-movflags", "+faststart", "-profile:v", "baseline", "-level", "3.0"],
        )
        try:
            for rgb in frames:
                writer.append_data(rgb)
        finally:
            writer.close()
        with open(path, "rb") as f:
            data = f.read()
        return data if len(data) > 200 else None  # guard against a silently-empty file
    except Exception as exc:  # pragma: no cover - depends on host sandboxing
        log.warning("MP4 (ffmpeg) encoding unavailable (%s); GIF is used for display anyway.", exc)
        return None


def field_frames_to_gif(frames, xs, ys, title, cmap="RdBu_r", symmetric=True, unit="nm",
                         fps=18, mark_fn=None, dpi=80, figsize=(4.4, 3.7),
                         hold_last=None, also_mp4=True) -> dict:
    """Render a list of 2-D field snapshots into a playable animation.

    Returns a dict with `gif` (bytes, always present) and `mp4` (bytes or None,
    best-effort). Use `gif` for on-screen `st.image(...)` display (guaranteed to work);
    offer `mp4` as an optional download when not None.
    """
    if not frames:
        raise ValueError("no frames to render")
    x0, x1 = xs[0] / 1e-9, xs[-1] / 1e-9
    y0, y1 = ys[0] / 1e-9, ys[-1] / 1e-9
    vmax = max(float(np.abs(f).max()) for f in frames) or 1.0
    kwargs = dict(vmin=-vmax, vmax=vmax) if symmetric else dict(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # NB: deliberately no animated=True here -- that flag tells matplotlib this artist
    # is blit-managed and should be *skipped* during a normal full canvas.draw(), which
    # is exactly what we call per-frame below (no blitting). Setting it silently produced
    # blank frames (axes/colorbar drawn, image itself never rendered).
    im = ax.imshow(frames[0].T, origin="lower", extent=[x0, x1, y0, y1], cmap=cmap,
                    aspect="equal", **kwargs)
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    title_artist = ax.set_title(title)
    if mark_fn is not None:
        mark_fn(ax)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()

    def update(i):
        im.set_data(frames[i].T)
        title_artist.set_text(f"{title}  (frame {i + 1}/{len(frames)})")

    if hold_last is None:
        hold_last = max(1, fps // 2)
    try:
        rgb_frames = _render_rgb_frames(update, len(frames), fig, hold_last=max(1, hold_last))
    finally:
        plt.close(fig)

    gif_bytes = _encode_gif(rgb_frames, fps)
    mp4_bytes = _encode_mp4(rgb_frames, fps) if also_mp4 else None
    return dict(gif=gif_bytes, mp4=mp4_bytes)


def line_series_to_gif(t, series: dict, xlabel, ylabel, title, fps=18, dpi=80,
                        figsize=(5.6, 3.3), also_mp4=True, n_frames=110) -> dict:
    """Render a growing line plot (e.g. an expectation value catching up to the current
    time) as a playable animation. Returns the same `{gif, mp4}` dict as
    `field_frames_to_gif`."""
    t = np.asarray(t)
    n = len(t)
    if n == 0:
        raise ValueError("no data to render")
    n_frames = min(n, n_frames)
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

    def update(k_idx):
        k = idxs[k_idx]
        for label, ys in series.items():
            lines[label].set_data(t[:k], np.asarray(ys)[:k])

    try:
        rgb_frames = _render_rgb_frames(update, n_frames, fig, hold_last=max(1, fps // 3))
    finally:
        plt.close(fig)

    gif_bytes = _encode_gif(rgb_frames, fps)
    mp4_bytes = _encode_mp4(rgb_frames, fps) if also_mp4 else None
    return dict(gif=gif_bytes, mp4=mp4_bytes)
