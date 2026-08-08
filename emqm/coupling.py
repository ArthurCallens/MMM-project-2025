"""Couples the EM (FDTD2D) and QM (Schrodinger2D) solvers.

Forward coupling (EM -> QM): the E field is assumed uniform over the (nm-scale) well, so
the length-gauge dipole interaction H_int = q(Ex x + Ey y) only needs the E field sampled
at the well centre (the assignment's "well dimensions small enough" approximation,
Part 2 Sec. 3).

Backward coupling (QM -> EM): the quantum current density j_q (Part 2, eq. 1) has exactly
the units [A/m^2] of a source current density in the 2-D Ampere-Maxwell law, so it is
injected directly into the EM grid rather than reduced to a lumped dipole. Because the
QM grid is generally much finer than the (possibly locally-refined, but still coarser) EM
grid, each EM Yee node's source current is the *area-average* of j_q over that EM cell's
footprint (a small, local, vectorised histogram/bin-average -- restricted to a bounding
box around the well so cost is independent of the full EM grid size). Multiple independent
wells are supported (each with its own Schrodinger2D instance and location), radiatively
coupled only through the shared EM field -- the "coupling between multiple wells" bonus
suggested in the assignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fdtd_te import FDTD2D
from .quantum import Schrodinger2D


def _cell_edges(primary: np.ndarray, dual: np.ndarray) -> np.ndarray:
    return np.concatenate(([primary[0]], dual, [primary[-1]]))


@dataclass
class Well:
    qm: Schrodinger2D
    x0: float               # well centre, EM global coordinates [m]
    y0: float
    N: float = 1e7           # particle line density [1/m], see Part-2 assignment
    backward_coupling: bool = True
    label: str = "well"

    history: dict = field(default_factory=lambda: {
        "t": [], "x": [], "y": [], "px": [], "py": [], "Ekin": [], "norm": [],
        "Ex_drive": [], "Ey_drive": [],
    })

    def record(self, Ex_drive, Ey_drive):
        x, y = self.qm.expectation_xy()
        px, py = self.qm.expectation_p()
        self.history["t"].append(self.qm.t)
        self.history["x"].append(x)
        self.history["y"].append(y)
        self.history["px"].append(px)
        self.history["py"].append(py)
        self.history["Ekin"].append(self.qm.expectation_kinetic_energy())
        self.history["norm"].append(self.qm.norm())
        self.history["Ex_drive"].append(Ex_drive)
        self.history["Ey_drive"].append(Ey_drive)


class CoupledSimulation:
    def __init__(self, fdtd: FDTD2D, wells: list[Well] | None = None):
        self.fdtd = fdtd
        self.wells = list(wells or [])
        self._prep = [self._prepare_well(w) for w in self.wells]

    def _prepare_well(self, well: Well):
        g = self.fdtd.grid
        margin = 2 * well.qm.dx
        x_lo, x_hi = well.x0 - well.qm.Lx / 2 - margin, well.x0 + well.qm.Lx / 2 + margin
        y_lo, y_hi = well.y0 - well.qm.Ly / 2 - margin, well.y0 + well.qm.Ly / 2 + margin

        i0x, i1x = np.searchsorted(g.x, [x_lo, x_hi])
        i0x, i1x = max(0, i0x - 1), min(g.Nx - 1, i1x + 1)
        j0x, j1x = np.searchsorted(g.y, [y_lo, y_hi])
        j0x, j1x = max(0, j0x - 1), min(g.Ny, j1x + 1)

        i0y, i1y = np.searchsorted(g.x, [x_lo, x_hi])
        i0y, i1y = max(0, i0y - 1), min(g.Nx, i1y + 1)
        j0y, j1y = np.searchsorted(g.y, [y_lo, y_hi])
        j0y, j1y = max(0, j0y - 1), min(g.Ny - 1, j1y + 1)

        ex_x_edges = g.x[i0x:i1x + 1]
        ex_y_edges = _cell_edges(g.y, g.yd)[j0x:j1x + 2]
        ey_x_edges = _cell_edges(g.x, g.xd)[i0y:i1y + 2]
        ey_y_edges = g.y[j0y:j1y + 1]

        return dict(
            ex_box=(i0x, i1x, j0x, j1x), ex_x_edges=ex_x_edges, ex_y_edges=ex_y_edges,
            ey_box=(i0y, i1y, j0y, j1y), ey_x_edges=ey_x_edges, ey_y_edges=ey_y_edges,
        )

    @staticmethod
    def _bin_average(Xg, Yg, J, x_edges, y_edges, shape):
        nx, ny = shape
        ix = np.clip(np.searchsorted(x_edges, Xg.ravel(), side="right") - 1, 0, nx - 1)
        iy = np.clip(np.searchsorted(y_edges, Yg.ravel(), side="right") - 1, 0, ny - 1)
        flat = ix * ny + iy
        n = nx * ny
        sums = np.bincount(flat, weights=J.ravel(), minlength=n)
        counts = np.bincount(flat, minlength=n)
        avg = np.zeros(n)
        nz = counts > 0
        avg[nz] = sums[nz] / counts[nz]
        return avg.reshape(nx, ny)

    def _deposit(self, well: Well, prep: dict):
        jx, jy = well.qm.quantum_current_density(well.N)
        Xg = well.x0 + well.qm.X
        Yg = well.y0 + well.qm.Y

        i0x, i1x, j0x, j1x = prep["ex_box"]
        sub_x = self._bin_average(Xg, Yg, jx, prep["ex_x_edges"], prep["ex_y_edges"], (i1x - i0x, j1x - j0x))
        Jx_full = np.zeros_like(self.fdtd.Ex)
        Jx_full[i0x:i1x, j0x:j1x] = sub_x

        i0y, i1y, j0y, j1y = prep["ey_box"]
        sub_y = self._bin_average(Xg, Yg, jy, prep["ey_x_edges"], prep["ey_y_edges"], (i1y - i0y, j1y - j0y))
        Jy_full = np.zeros_like(self.fdtd.Ey)
        Jy_full[i0y:i1y, j0y:j1y] = sub_y

        return Jx_full, Jy_full

    def step(self):
        dt = self.fdtd.dt
        extra_Jx = np.zeros_like(self.fdtd.Ex)
        extra_Jy = np.zeros_like(self.fdtd.Ey)

        drives_before = []
        for well in self.wells:
            ex, ey = self.fdtd.e_field_at(well.x0, well.y0)
            drives_before.append((ex, ey))

        for well, prep, (ex, ey) in zip(self.wells, self._prep, drives_before):
            if well.backward_coupling:
                jx_full, jy_full = self._deposit(well, prep)
                extra_Jx += jx_full
                extra_Jy += jy_full

        self.fdtd.step(extra_Jx if self.wells else None, extra_Jy if self.wells else None)

        # Schrodinger2D.step() expects a time-centred field (its own docstring says so), i.e.
        # the average of the field just before and just after this dt -- NOT the stale
        # pre-update sample. Using the stale sample introduces an O(dt) time-lag error in the
        # well's own self-consistent radiation feedback that, left uncorrected, shows up as
        # spurious energy GAIN (numerically unstable "anti-damping") instead of the physically
        # correct radiation damping, growing with N and the coupling step size. Verified: the
        # error shrinks close to linearly as dt -> 0 (confirming a time-centering artifact, not
        # a sign error in the physics), and this fix removes it.
        for well, (ex0, ey0) in zip(self.wells, drives_before):
            ex1, ey1 = self.fdtd.e_field_at(well.x0, well.y0)
            ex_c, ey_c = 0.5 * (ex0 + ex1), 0.5 * (ey0 + ey1)
            well.qm.step(dt, Ex=ex_c, Ey=ey_c)
            well.record(ex_c, ey_c)
