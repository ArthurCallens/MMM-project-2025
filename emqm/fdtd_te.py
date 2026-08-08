"""2-D TE nonuniform Yee-FDTD solver: assembles grid + materials + CPML + TFSF sources
into a single explicit leapfrog time-stepping engine.

Update equations (from Maxwell's 2-D TE curl equations, Sec. 2.1 of the assignment):
    dHz/dt = -(1/mu0) [ dEy/dx - dEx/dy ]
    dEx/dt =  (1/eps) [ dHz/dy - Jx ]        (Jx: Drude polarisation / quantum current)
    dEy/dt =  (1/eps) [ -dHz/dx - Jy ]
with dEy/dx etc. replaced by their PML-stretched-coordinate counterparts inside the CPML
layer (see pml.py) and eps = eps_inf(x,y) inside Drude/dielectric scatterers, whose bound
polarisation current J obeys the auxiliary differential equation dJ/dt + gamma J = eps0
omega_p^2 E (see materials.py / numba_kernels.py).

All array operations are vectorised (no Python loops over grid points, per the project's
own performance hint); the Drude ADE update additionally has an optional Numba-JIT path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import EPS0, MU0
from .grid import Grid2D
from .materials import MaterialMaps
from .pml import CPML2D, PMLParams
from .tfsf import TFSFBox, PlaneWave
from .numba_kernels import drude_update


def _drude_coeffs(eps_inf, omega_p2, gamma, dt):
    C1 = (2.0 - gamma * dt) / (2.0 + gamma * dt)
    C2 = (EPS0 * omega_p2 * dt) / (2.0 + gamma * dt)
    beta = dt * C2 / (2.0 * eps_inf * EPS0)
    denom_inv = 1.0 / (1.0 + beta)
    factor_Jn = dt * (1.0 + C1) / (2.0 * eps_inf * EPS0)
    factor_RHS = dt / (eps_inf * EPS0)
    return dict(C1=C1, C2=C2, beta=beta, denom_inv=denom_inv, factor_Jn=factor_Jn, factor_RHS=factor_RHS)


def bilinear_interp(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, xo: float, yo: float) -> float:
    """Bilinearly interpolate `field` (defined on the tensor grid xs (x-axis) x ys)
    at the physical point (xo, yo). xs, ys must be sorted ascending."""
    i = np.clip(np.searchsorted(xs, xo) - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, yo) - 1, 0, len(ys) - 2)
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[j], ys[j + 1]
    tx = 0.0 if x1 == x0 else (xo - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (yo - y0) / (y1 - y0)
    f00, f10 = field[i, j], field[i + 1, j]
    f01, f11 = field[i, j + 1], field[i + 1, j + 1]
    return float((1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11)


@dataclass
class ObservationPoint:
    x: float
    y: float
    name: str = ""
    Ex: list = field(default_factory=list)
    Ey: list = field(default_factory=list)
    Hz: list = field(default_factory=list)
    t: list = field(default_factory=list)


class FDTD2D:
    def __init__(
        self,
        grid: Grid2D,
        dt: float,
        materials: MaterialMaps,
        pml_x: PMLParams | None = None,
        pml_y: PMLParams | None = None,
        outer_bc: str = "pec",
        engine: str = "numpy",
    ):
        self.grid = grid
        self.dt = dt
        self.mat = materials
        self.outer_bc = outer_bc
        self.engine = engine
        self.n = 0
        self.t = 0.0

        Nx, Ny = grid.Nx, grid.Ny
        self.Ex = np.zeros((Nx, Ny + 1))
        self.Ey = np.zeros((Nx + 1, Ny))
        self.Hz = np.zeros((Nx, Ny))
        self.Jx = np.zeros((Nx, Ny + 1))
        self.Jy = np.zeros((Nx + 1, Ny))

        self.pml = CPML2D(grid, dt, pml_x, pml_y)
        self.tfsf_boxes: list[TFSFBox] = []
        self.observers: list[ObservationPoint] = []

        # Drude ADE coefficients, precomputed on the *interior* slices that actually
        # get the curl update (j=1..Ny-1 for Ex, i=1..Nx-1 for Ey); the outer PEC edge
        # rows/columns are left at Ex=Ey=0 for all time (or, if periodic, wrapped).
        self._ex_int = (slice(None), slice(1, Ny))
        self._ey_int = (slice(1, Nx), slice(None))
        self._ex_coef = _drude_coeffs(
            materials.eps_inf_ex[self._ex_int], materials.omega_p2_ex[self._ex_int],
            materials.gamma_ex[self._ex_int], dt,
        )
        self._ey_coef = _drude_coeffs(
            materials.eps_inf_ey[self._ey_int], materials.omega_p2_ey[self._ey_int],
            materials.gamma_ey[self._ey_int], dt,
        )

    def add_tfsf(self, wave: PlaneWave, iL: int, iR: int, jL: int, jR: int) -> TFSFBox:
        box = TFSFBox(self.grid, wave, iL, iR, jL, jR)
        box.set_dt(self.dt)
        self.tfsf_boxes.append(box)
        return box

    def add_observer(self, x: float, y: float, name: str = "") -> ObservationPoint:
        obs = ObservationPoint(x, y, name or f"({x:.3g},{y:.3g})")
        self.observers.append(obs)
        return obs

    # ------------------------------------------------------------------ stepping

    def _hz_update(self):
        g = self.grid
        raw_dEy_dx = (self.Ey[1:, :] - self.Ey[:-1, :]) / g.dx
        raw_dEx_dy = (self.Ex[:, 1:] - self.Ex[:, :-1]) / g.dy
        ddx_ey = self.pml.stretched_ddx_ey(raw_dEy_dx)
        ddy_ex = self.pml.stretched_ddy_ex(raw_dEx_dy)
        self.Hz -= self.dt / MU0 * (ddx_ey - ddy_ex)

    def _e_update(self, extra_Jx=None, extra_Jy=None):
        g = self.grid
        # --- Ex ---
        raw_dHz_dy = (self.Hz[:, 1:] - self.Hz[:, :-1]) / g.dyd
        ddy_hz = self.pml.stretched_ddy_hz(raw_dHz_dy)
        c = self._ex_coef
        Jx_int = self.Jx[self._ex_int]
        RHS = c["factor_RHS"] * ddy_hz
        if extra_Jx is not None:
            RHS = RHS - c["factor_RHS"] * extra_Jx[self._ex_int]
        Ex_int = self.Ex[self._ex_int]
        Ex_new, Jx_new = drude_update(Ex_int, Jx_int, RHS, c["C1"], c["C2"], c["beta"], c["denom_inv"], c["factor_Jn"], self.engine)
        self.Ex[self._ex_int] = Ex_new
        self.Jx[self._ex_int] = Jx_new

        # --- Ey ---
        raw_dHz_dx = (self.Hz[1:, :] - self.Hz[:-1, :]) / g.dxd
        ddx_hz = self.pml.stretched_ddx_hz(raw_dHz_dx)
        c = self._ey_coef
        Jy_int = self.Jy[self._ey_int]
        RHS = -c["factor_RHS"] * ddx_hz
        if extra_Jy is not None:
            RHS = RHS - c["factor_RHS"] * extra_Jy[self._ey_int]
        Ey_int = self.Ey[self._ey_int]
        Ey_new, Jy_new = drude_update(Ey_int, Jy_int, RHS, c["C1"], c["C2"], c["beta"], c["denom_inv"], c["factor_Jn"], self.engine)
        self.Ey[self._ey_int] = Ey_new
        self.Jy[self._ey_int] = Jy_new

        if self.outer_bc == "periodic":
            self.Ex[:, 0] = self.Ex[:, -1]
            self.Ey[0, :] = self.Ey[-1, :]

    def step(self, extra_Jx=None, extra_Jy=None):
        """Advance one full leapfrog cycle: Hz at t+dt/2, then Ex,Ey at t+dt."""
        t_h = self.t + 0.5 * self.dt
        self._hz_update()
        for box in self.tfsf_boxes:
            box.correct_hz(self.Hz, t_h)
        if np.any(self.mat.pmc_hz):
            self.Hz[self.mat.pmc_hz] = 0.0

        t_e = self.t + self.dt
        self._e_update(extra_Jx, extra_Jy)
        for box in self.tfsf_boxes:
            box.correct_e(self.Ex, self.Ey, t_e)
        if np.any(self.mat.pec_ex):
            self.Ex[self.mat.pec_ex] = 0.0
        if np.any(self.mat.pec_ey):
            self.Ey[self.mat.pec_ey] = 0.0

        self.t = t_e
        self.n += 1
        self._record()

    def _record(self):
        for obs in self.observers:
            obs.t.append(self.t)
            obs.Ex.append(bilinear_interp(self.Ex, self.grid.xd, self.grid.y, obs.x, obs.y))
            obs.Ey.append(bilinear_interp(self.Ey, self.grid.x, self.grid.yd, obs.x, obs.y))
            obs.Hz.append(bilinear_interp(self.Hz, self.grid.xd, self.grid.yd, obs.x, obs.y))

    def e_field_at(self, xo: float, yo: float) -> tuple[float, float]:
        """Instantaneous (Ex, Ey) at an arbitrary point, for QM forward-coupling."""
        ex = bilinear_interp(self.Ex, self.grid.xd, self.grid.y, xo, yo)
        ey = bilinear_interp(self.Ey, self.grid.x, self.grid.yd, xo, yo)
        return ex, ey
