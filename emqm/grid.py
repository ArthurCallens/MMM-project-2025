"""Nonuniform 1-D/2-D Yee-grid construction.

The user specifies the spatial cell size as a function of position via a handful of
"anchor" (position, step) points (e.g. coarse in free space, fine around a scatterer or
a quantum well). Cell size is interpolated geometrically (log-linear) between anchors,
which yields the smooth grading recommended for nonuniform FDTD grids, and the axis is
then marched out from x=0 accumulating cells of that local size until the requested
length is spanned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def nonuniform_axis(length: float, anchors, min_cells: int = 8, max_cells: int = 400_000) -> np.ndarray:
    """Build 1-D nonuniform node coordinates spanning [0, length].

    Parameters
    ----------
    length : total physical length of the axis [m]
    anchors : iterable of (position, step) pairs describing the desired local cell size
        at given positions. Positions outside [0, length] are clipped. At least one
        anchor must be given; the step profile is held constant beyond the outermost
        anchors.
    """
    anchors = sorted(anchors, key=lambda a: a[0])
    ax = np.array([min(max(a[0], 0.0), length) for a in anchors], dtype=float)
    ad = np.array([a[1] for a in anchors], dtype=float)
    if ax[0] > 0:
        ax = np.insert(ax, 0, 0.0)
        ad = np.insert(ad, 0, ad[0])
    if ax[-1] < length:
        ax = np.append(ax, length)
        ad = np.append(ad, ad[-1])
    # merge duplicate positions (keep smallest step -> conservative/finer)
    ax_u, idx = np.unique(ax, return_index=True)
    if len(ax_u) < len(ax):
        ad_u = np.array([ad[ax == v].min() for v in ax_u])
        ax, ad = ax_u, ad_u
    log_d = np.log(ad)

    def step_at(x):
        return np.exp(np.interp(x, ax, log_d))

    nodes = [0.0]
    x = 0.0
    n = 0
    while x < length and n < max_cells:
        h = float(step_at(x))
        h = max(h, length * 1e-12)
        x_next = x + h
        if x_next >= length:
            break
        nodes.append(x_next)
        x = x_next
        n += 1
    nodes.append(length)
    nodes = np.array(nodes)
    # avoid a degenerate last (near-zero) cell
    if len(nodes) > 2 and (nodes[-1] - nodes[-2]) < 0.25 * (nodes[-2] - nodes[-3]):
        nodes = np.delete(nodes, -2)
    if len(nodes) - 1 < min_cells:
        nodes = np.linspace(0.0, length, min_cells + 1)
    return nodes


def uniform_axis(length: float, step: float) -> np.ndarray:
    n = max(2, int(round(length / step)) + 1)
    return np.linspace(0.0, length, n)


@dataclass
class Axis:
    """One Cartesian axis of a Yee grid: primary nodes + staggered dual nodes."""

    nodes: np.ndarray  # primary node coordinates, shape (N+1,)

    def __post_init__(self):
        self.n = len(self.nodes) - 1
        self.length = self.nodes[-1] - self.nodes[0]
        self.d = np.diff(self.nodes)                      # primary steps, (n,)
        self.dual = 0.5 * (self.nodes[:-1] + self.nodes[1:])  # dual (staggered) nodes, (n,)
        self.dd = np.diff(self.dual)                       # dual steps (interior), (n-1,)

    def index_of(self, x: float) -> int:
        """Nearest primary-node index to physical position x."""
        return int(np.argmin(np.abs(self.nodes - x)))

    def dual_index_of(self, x: float) -> int:
        return int(np.argmin(np.abs(self.dual - x)))


class Grid2D:
    """2-D nonuniform Yee grid for the TE (Ex, Ey, Hz) field set.

    Field placement (standard Yee staggering):
      Hz[i, j]  at (xd[i], yd[j])          shape (Nx, Ny)
      Ex[i, j]  at (xd[i],  y[j])          shape (Nx, Ny+1)
      Ey[i, j]  at ( x[i], yd[j])          shape (Nx+1, Ny)
    """

    def __init__(self, x_nodes: np.ndarray, y_nodes: np.ndarray):
        self.ax = Axis(np.asarray(x_nodes, dtype=float))
        self.ay = Axis(np.asarray(y_nodes, dtype=float))
        self.Nx, self.Ny = self.ax.n, self.ay.n
        self.Lx, self.Ly = self.ax.length, self.ay.length

        # broadcastable spacing arrays for vectorized updates
        self.dx = self.ax.d[:, None]          # (Nx,1)  primary steps in x
        self.dy = self.ay.d[None, :]          # (1,Ny)  primary steps in y
        self.dxd = self.ax.dd[:, None]        # (Nx-1,1) dual steps in x (interior)
        self.dyd = self.ay.dd[None, :]        # (1,Ny-1) dual steps in y (interior)

        self.x, self.y = self.ax.nodes, self.ay.nodes
        self.xd, self.yd = self.ax.dual, self.ay.dual

        self.X_hz, self.Y_hz = np.meshgrid(self.xd, self.yd, indexing="ij")
        self.X_ex, self.Y_ex = np.meshgrid(self.xd, self.y, indexing="ij")
        self.X_ey, self.Y_ey = np.meshgrid(self.x, self.yd, indexing="ij")

    def min_step(self) -> float:
        return min(self.ax.d.min(), self.ay.d.min())

    def cfl_dt(self, courant: float = 0.99) -> float:
        """Conservative CFL-stable time step estimate (uniform-grid style bound)."""
        from .constants import C0
        dmin = self.min_step()
        return courant * dmin / (C0 * np.sqrt(2.0))
