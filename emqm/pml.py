"""Convolutional PML (CPML) = ADE realisation of a uniaxial PML (UPML) for the 2-D TE set.

Maxwell's curl equations are solved in "stretched" coordinates
    d/dx -> (1/kappa_x) d/dx + psi_x ,    d/dy -> (1/kappa_y) d/dy + psi_y
with psi the recursive-convolution auxiliary variable

    psi^{n+1} = b * psi^n + a * (raw central difference of the field being differentiated)

b = exp(-(sigma/kappa + alpha) dt/eps0),
a = sigma / (sigma*kappa + alpha*kappa^2) * (b - 1)   (a=0 where sigma=alpha=0).

This is mathematically equivalent to the split-field UPML (both are exact discretisations
of the complex-coordinate-stretched Maxwell equations) but is implemented here as a
single unsplit auxiliary-differential-equation update, which is simpler to combine with
a nonuniform grid, Drude media and interior PEC/PMC scatterers. sigma/kappa/alpha are
graded polynomially with distance into the PML layer, so they are naturally compatible
with a nonuniform grid (they are evaluated directly at each node's physical position).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import EPS0, ETA0


@dataclass
class PMLParams:
    thickness: float           # physical thickness of the PML layer [m]
    m: float = 3.0              # sigma/kappa grading order
    ma: float = 1.0             # alpha grading order
    kappa_max: float = 7.0
    alpha_max: float = 0.0      # CFS term; 0 disables it (pure graded-conductivity PML)
    r0: float = 1e-8            # target theoretical reflection coefficient at normal incidence
    sides: tuple = ("lo", "hi")  # which sides of the axis get a PML layer


def _profile_1d(nodes: np.ndarray, p: PMLParams):
    """Return sigma, kappa, alpha arrays (same shape as `nodes`) along one axis."""
    x0, x1 = nodes[0], nodes[-1]
    sigma_max = -(p.m + 1) * np.log(p.r0) / (2.0 * ETA0 * p.thickness)
    sigma = np.zeros_like(nodes)
    kappa = np.ones_like(nodes)
    alpha = np.zeros_like(nodes)

    if "lo" in p.sides:
        d = (x0 + p.thickness) - nodes
        mask = d > 0
        rho = np.clip(d[mask] / p.thickness, 0.0, 1.0)
        sigma[mask] = sigma_max * rho ** p.m
        kappa[mask] = 1.0 + (p.kappa_max - 1.0) * rho ** p.m
        alpha[mask] = p.alpha_max * (1.0 - rho) ** p.ma
    if "hi" in p.sides:
        d = nodes - (x1 - p.thickness)
        mask = d > 0
        rho = np.clip(d[mask] / p.thickness, 0.0, 1.0)
        # combine with 'lo' contributions (max, in case thickness > half-domain)
        sigma[mask] = np.maximum(sigma[mask], sigma_max * rho ** p.m)
        kappa[mask] = np.maximum(kappa[mask], 1.0 + (p.kappa_max - 1.0) * rho ** p.m)
        alpha[mask] = np.maximum(alpha[mask], p.alpha_max * (1.0 - rho) ** p.ma)
    return sigma, kappa, alpha


def ba_coeffs(sigma, kappa, alpha, dt):
    b = np.exp(-(sigma / kappa + alpha) * dt / EPS0)
    denom = sigma * kappa + alpha * kappa ** 2
    a = np.zeros_like(sigma)
    nz = denom > 0
    a[nz] = sigma[nz] / denom[nz] * (b[nz] - 1.0)
    return b, a


class CPML2D:
    """Holds the psi auxiliary arrays and b/a coefficients for the four stretched
    derivatives needed by the 2-D TE update, at the exact staggered location of each."""

    def __init__(self, grid, dt: float, px: PMLParams | None, py: PMLParams | None):
        self.grid = grid
        Nx, Ny = grid.Nx, grid.Ny
        x_int = grid.x[1:-1]   # interior primary x-nodes (Ey update, i=1..Nx-1)
        y_int = grid.y[1:-1]   # interior primary y-nodes (Ex update, j=1..Ny-1)

        def flat(sh, val=0.0):
            return np.full(sh, val)

        # default: no PML on an axis -> kappa=1, sigma=alpha=0 everywhere (psi stays 0)
        if px is not None:
            sig_xd, kap_xd, alp_xd = _profile_1d(grid.xd, px)   # at Hz location
            sig_x, kap_x, alp_x = _profile_1d(x_int, px)        # at Ey(x) location
        else:
            sig_xd, kap_xd, alp_xd = flat((Nx,)), flat((Nx,), 1.0), flat((Nx,))
            sig_x, kap_x, alp_x = flat((Nx - 1,)), flat((Nx - 1,), 1.0), flat((Nx - 1,))
        if py is not None:
            sig_yd, kap_yd, alp_yd = _profile_1d(grid.yd, py)   # at Hz location
            sig_y, kap_y, alp_y = _profile_1d(y_int, py)        # at Ex(y) location
        else:
            sig_yd, kap_yd, alp_yd = flat((Ny,)), flat((Ny,), 1.0), flat((Ny,))
            sig_y, kap_y, alp_y = flat((Ny - 1,)), flat((Ny - 1,), 1.0), flat((Ny - 1,))

        self.kappa_xd = kap_xd[:, None]
        self.kappa_x = kap_x[:, None]
        self.kappa_yd = kap_yd[None, :]
        self.kappa_y = kap_y[None, :]

        self.b_xd, self.a_xd = (c[:, None] for c in ba_coeffs(sig_xd, kap_xd, alp_xd, dt))
        self.b_x, self.a_x = (c[:, None] for c in ba_coeffs(sig_x, kap_x, alp_x, dt))
        self.b_yd, self.a_yd = (c[None, :] for c in ba_coeffs(sig_yd, kap_yd, alp_yd, dt))
        self.b_y, self.a_y = (c[None, :] for c in ba_coeffs(sig_y, kap_y, alp_y, dt))

        # psi for: Hz update (d Ey/dx at xd, d Ex/dy at yd);
        # Ex update (d Hz/dy at interior y); Ey update (d Hz/dx at interior x)
        self.psi_hz_ey_x = np.zeros((Nx, Ny))
        self.psi_hz_ex_y = np.zeros((Nx, Ny))
        self.psi_ex_hz_y = np.zeros((Nx, Ny - 1))
        self.psi_ey_hz_x = np.zeros((Nx - 1, Ny))

    def stretched_ddx_ey(self, dEy_dx_raw):
        self.psi_hz_ey_x = self.b_xd * self.psi_hz_ey_x + self.a_xd * dEy_dx_raw
        return dEy_dx_raw / self.kappa_xd + self.psi_hz_ey_x

    def stretched_ddy_ex(self, dEx_dy_raw):
        self.psi_hz_ex_y = self.b_yd * self.psi_hz_ex_y + self.a_yd * dEx_dy_raw
        return dEx_dy_raw / self.kappa_yd + self.psi_hz_ex_y

    def stretched_ddy_hz(self, dHz_dy_raw):
        self.psi_ex_hz_y = self.b_y * self.psi_ex_hz_y + self.a_y * dHz_dy_raw
        return dHz_dy_raw / self.kappa_y + self.psi_ex_hz_y

    def stretched_ddx_hz(self, dHz_dx_raw):
        self.psi_ey_hz_x = self.b_x * self.psi_ey_hz_x + self.a_x * dHz_dx_raw
        return dHz_dx_raw / self.kappa_x + self.psi_ey_hz_x
