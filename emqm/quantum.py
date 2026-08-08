"""2-D time-dependent Schrodinger solver for a single electron in a (harmonic-oscillator)
well, driven by a spatially-uniform electric field via the length-gauge dipole
interaction Hamiltonian H_int(t) = -q [Ex(t) x + Ey(t) y] (project Part 2, Sec. 3).

Because the harmonic-oscillator potential and the length-gauge interaction are both
separable, V(x,y,t) = Vx(x,t) + Vy(y,t), the x- and y- pieces of the Hamiltonian commute
exactly ([Hx, Hy] = 0), so a sequential (unsplit-error) ADI Crank-Nicolson scheme is used:
one full implicit x-sweep followed by one full implicit y-sweep per time step. Each sweep
is a *batched* tridiagonal solve (the same matrix for every row/column, many
right-hand-sides at once) via `scipy.linalg.solve_banded`, i.e. fully vectorised, no
explicit loop over grid points. Crank-Nicolson (a Cayley transform of a Hermitian
operator) is unconditionally unitary, so the total norm is conserved to machine
precision regardless of the time step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from .constants import HBAR, ME, QE


@dataclass
class Schrodinger2D:
    Lx: float                    # well width [m], domain spans [-Lx/2, Lx/2]
    Ly: float
    dx: float                    # grid step [m] (uniform)
    m_eff: float = 0.15 * ME     # effective mass
    omega_ho: float = 50e14      # harmonic-oscillator angular frequency [rad/s]
    charge: float = -QE          # electron charge
    boundary_radius: float | None = None  # circular Dirichlet boundary; default: inscribed circle

    def __post_init__(self):
        self.dy = self.dx
        self.Nx = max(4, int(round(self.Lx / self.dx)) | 1)  # odd -> node exactly at 0
        self.Ny = max(4, int(round(self.Ly / self.dy)) | 1)
        self.x = np.linspace(-self.Lx / 2, self.Lx / 2, self.Nx)
        self.y = np.linspace(-self.Ly / 2, self.Ly / 2, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        if self.boundary_radius is None:
            self.boundary_radius = 0.5 * min(self.Lx, self.Ly)
        self.inside = (self.X ** 2 + self.Y ** 2) <= self.boundary_radius ** 2

        # interior (Dirichlet-free) index ranges -- the outermost ring is pinned to 0
        self.ix = slice(1, self.Nx - 1)
        self.iy = slice(1, self.Ny - 1)
        self.xi = self.x[self.ix]
        self.yi = self.y[self.iy]

        k = HBAR ** 2 / (2 * self.m_eff)
        self._kin_diag = k * 2.0 / self.dx ** 2       # scalar (uniform grid)
        self._kin_off = -k / self.dx ** 2              # scalar

        self.Vx0 = 0.5 * self.m_eff * self.omega_ho ** 2 * self.xi ** 2
        self.Vy0 = 0.5 * self.m_eff * self.omega_ho ** 2 * self.yi ** 2

        self.psi = np.zeros((self.Nx, self.Ny), dtype=complex)
        self.t = 0.0
        self._prev_prob = None

    # ------------------------------------------------------------- initial states

    def ground_state(self) -> np.ndarray:
        a = self.m_eff * self.omega_ho / HBAR
        psi = (a / np.pi) ** 0.5 * np.exp(-a * (self.X ** 2 + self.Y ** 2) / 2.0)
        return psi.astype(complex)

    def coherent_state(self, x0: float = 0.0, y0: float = 0.0, px0: float = 0.0, py0: float = 0.0) -> np.ndarray:
        a = self.m_eff * self.omega_ho / HBAR
        env = (a / np.pi) ** 0.5 * np.exp(-a * ((self.X - x0) ** 2 + (self.Y - y0) ** 2) / 2.0)
        phase = np.exp(1j * (px0 * self.X + py0 * self.Y) / HBAR)
        return (env * phase).astype(complex)

    def set_state(self, psi0: np.ndarray):
        self.psi = psi0.astype(complex).copy()
        self.psi[~self.inside] = 0.0
        self._prev_prob = np.abs(self.psi) ** 2

    # ------------------------------------------------------------------ dynamics

    def _sweep(self, psi, diag_extra, axis, dt):
        """One implicit Crank-Nicolson sweep along `axis` (0=x, 1=y) of the interior
        array `psi` (shape (Nxi, Nyi)), with per-node extra potential `diag_extra`
        (1-D, along `axis`) added to the kinetic diagonal."""
        n = psi.shape[axis]
        c = 1j * dt / (2 * HBAR)
        diag = self._kin_diag + diag_extra
        off = self._kin_off

        if axis == 0:
            p = psi
        else:
            p = psi.T

        rhs = (1.0 - c * diag)[:, None] * p
        rhs[:-1, :] += (-c * off) * p[1:, :]
        rhs[1:, :] += (-c * off) * p[:-1, :]

        ab = np.zeros((3, n), dtype=complex)
        ab[0, 1:] = c * off
        ab[1, :] = 1.0 + c * diag
        ab[2, :-1] = c * off
        p_new = solve_banded((1, 1), ab, rhs)

        return p_new if axis == 0 else p_new.T

    def step(self, dt: float, Ex: float = 0.0, Ey: float = 0.0):
        """Advance the wavefunction by one time step under a spatially-uniform field
        (Ex, Ey) (already time-centred by the caller, e.g. averaged over the step)."""
        interior = self.psi[self.ix, self.iy]
        # Length-gauge dipole interaction: H_int = -q(Ex x + Ey y) (Cohen-Tannoudji "E1"
        # form, equivalently the ordinary Lorentz force F = qE on the charge in a uniform
        # field -- NOT +q(Ex x + Ey y), which integrates to the opposite-sign force).
        vx_extra = self.Vx0 - self.charge * Ex * self.xi
        vy_extra = self.Vy0 - self.charge * Ey * self.yi

        interior = self._sweep(interior, vx_extra, axis=0, dt=dt)
        interior = self._sweep(interior, vy_extra, axis=1, dt=dt)

        self._prev_prob = np.abs(self.psi) ** 2
        self.psi[:, :] = 0.0
        self.psi[self.ix, self.iy] = interior
        self.psi[~self.inside] = 0.0
        self.t += dt

    # -------------------------------------------------------------- observables

    def norm(self) -> float:
        return float(np.sum(np.abs(self.psi) ** 2) * self.dx * self.dy)

    def expectation_xy(self):
        p = np.abs(self.psi) ** 2
        ex = np.sum(p * self.X) * self.dx * self.dy
        ey = np.sum(p * self.Y) * self.dx * self.dy
        return float(ex), float(ey)

    def _grad(self):
        dpsidx = np.gradient(self.psi, self.dx, axis=0)
        dpsidy = np.gradient(self.psi, self.dy, axis=1)
        return dpsidx, dpsidy

    def expectation_p(self):
        dpsidx, dpsidy = self._grad()
        px = np.sum(np.conj(self.psi) * (-1j * HBAR * dpsidx)) * self.dx * self.dy
        py = np.sum(np.conj(self.psi) * (-1j * HBAR * dpsidy)) * self.dx * self.dy
        return complex(px).real, complex(py).real

    def expectation_kinetic_energy(self) -> float:
        dpsidx, dpsidy = self._grad()
        dens = np.abs(dpsidx) ** 2 + np.abs(dpsidy) ** 2
        return float(HBAR ** 2 / (2 * self.m_eff) * np.sum(dens) * self.dx * self.dy)

    def probability_current(self):
        """j_prob = (hbar/m*) Im[psi* grad psi]  (probability current, units 1/(m s))."""
        dpsidx, dpsidy = self._grad()
        jx = (HBAR / self.m_eff) * np.imag(np.conj(self.psi) * dpsidx)
        jy = (HBAR / self.m_eff) * np.imag(np.conj(self.psi) * dpsidy)
        return jx, jy

    def quantum_current_density(self, N: float):
        """j_q, eq. (1) of the Part-2 assignment: j_q = (q*hbar*N/m*) Im[psi* grad psi]."""
        jx, jy = self.probability_current()
        scale = self.charge * N
        return scale * jx, scale * jy

    def continuity_residual(self, dt: float):
        """||d|psi|^2/dt + div(j_prob)||, evaluated with the *previous* step's data
        (call right after step()). Returns the residual field (should be ~0)."""
        if self._prev_prob is None:
            return None
        dprob_dt = (np.abs(self.psi) ** 2 - self._prev_prob) / dt
        jx, jy = self.probability_current()
        djx = np.gradient(jx, self.dx, axis=0)
        djy = np.gradient(jy, self.dy, axis=1)
        return dprob_dt + djx + djy
