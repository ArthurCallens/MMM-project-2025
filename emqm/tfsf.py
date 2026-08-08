"""Total-field/scattered-field (TFSF) plane-wave injection for the 2-D TE grid.

Following the project brief's own suggestion (Sec. 3.3.1: replace the numerical phase
velocity by c = 1/sqrt(eps0 mu0) in the TFSF correction), the incident field used for
the correction terms is evaluated *analytically* rather than propagated on a 1-D
auxiliary grid. This has two convenient consequences: (1) it trivially supports
nonuniform grids (no auxiliary-grid interpolation needed) and (2) it supports oblique
incidence at *any* angle for free, since the correction only ever needs point samples
of the incident field p(t - (x cos(theta) + y sin(theta))/c) at grid-node coordinates.
Axis-aligned incidence (theta = 0 deg or 90 deg) is the special case explicitly required
by the assignment; general angles are the "oblique incidence" bonus feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import C0, ETA0


@dataclass
class PlaneWave:
    profile: callable       # p(t) -> amplitude, e.g. a GaussianPulse instance
    theta_deg: float = 0.0  # propagation direction, measured from +x axis, in the xy-plane
    E0: float = 1.0         # amplitude
    t0: float = 0.0         # extra time offset (rarely needed; profile already has tc)

    def __post_init__(self):
        self.theta = np.deg2rad(self.theta_deg)
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)

    def delay(self, X, Y):
        return (X * self.cos_t + Y * self.sin_t) / C0

    def Ex_inc(self, X, Y, t):
        return self.E0 * (-self.sin_t) * self.profile(t - self.delay(X, Y) - self.t0)

    def Ey_inc(self, X, Y, t):
        return self.E0 * self.cos_t * self.profile(t - self.delay(X, Y) - self.t0)

    def Hz_inc(self, X, Y, t):
        return (self.E0 / ETA0) * self.profile(t - self.delay(X, Y) - self.t0)


class TFSFBox:
    """Axis-aligned rectangular TFSF boundary specified by primary-grid index ranges
    i in [iL, iR], j in [jL, jR] (Ex/Ey total field spans these primary indices; the
    enclosed Hz total-field range is i in [iL, iR-1], j in [jL, jR-1])."""

    def __init__(self, grid, wave: PlaneWave, iL: int, iR: int, jL: int, jR: int):
        self.grid = grid
        self.wave = wave
        self.iL, self.iR, self.jL, self.jR = iL, iR, jL, jR
        from .constants import EPS0, MU0
        self.eps0, self.mu0 = EPS0, MU0

        g = grid
        self.x_iL, self.x_iR = g.x[iL], g.x[iR]
        self.y_jL, self.y_jR = g.y[jL], g.y[jR]

        # y-slices at Hz's yd coordinates (for the vertical boundaries)
        self.yd_v = g.yd[jL:jR]
        # x-slices at Hz's xd coordinates (for the horizontal boundaries)
        self.xd_h = g.xd[iL:iR]

        self.dx_iLm1 = g.ax.d[iL - 1] if iL > 0 else None
        self.dx_iR = g.ax.d[iR]
        self.dxd_iLm1 = g.ax.dual[iL] - g.ax.dual[iL - 1] if iL > 0 else None
        self.dxd_iRm1 = g.ax.dual[iR] - g.ax.dual[iR - 1]

        self.dy_jLm1 = g.ay.d[jL - 1] if jL > 0 else None
        self.dy_jR = g.ay.d[jR]
        self.dyd_jLm1 = g.ay.dual[jL] - g.ay.dual[jL - 1] if jL > 0 else None
        self.dyd_jRm1 = g.ay.dual[jR] - g.ay.dual[jR - 1]

    def correct_hz(self, Hz: np.ndarray, t: float):
        w, g, dt_mu = self.wave, self.grid, None
        iL, iR, jL, jR = self.iL, self.iR, self.jL, self.jR
        dt = self._dt
        # left / right vertical boundaries (Ey leakage into Hz)
        if iL > 0:
            Ey_inc = w.Ey_inc(self.x_iL, self.yd_v, t)
            Hz[iL - 1, jL:jR] += (dt / (self.mu0 * self.dx_iLm1)) * Ey_inc
        Ey_inc_r = w.Ey_inc(self.x_iR, self.yd_v, t)
        Hz[iR, jL:jR] -= (dt / (self.mu0 * self.dx_iR)) * Ey_inc_r
        # bottom / top horizontal boundaries (Ex leakage into Hz)
        if jL > 0:
            Ex_inc = w.Ex_inc(self.xd_h, self.y_jL, t)
            Hz[iL:iR, jL - 1] -= (dt / (self.mu0 * self.dy_jLm1)) * Ex_inc
        Ex_inc_t = w.Ex_inc(self.xd_h, self.y_jR, t)
        Hz[iL:iR, jR] += (dt / (self.mu0 * self.dy_jR)) * Ex_inc_t

    def correct_e(self, Ex: np.ndarray, Ey: np.ndarray, t: float):
        w = self.wave
        iL, iR, jL, jR = self.iL, self.iR, self.jL, self.jR
        dt = self._dt
        # Ey corrections at i=iL and i=iR (Hz leakage)
        if iL > 0:
            xd_l = self.grid.xd[iL - 1]
            Hz_inc_l = w.Hz_inc(xd_l, self.yd_v, t)
            Ey[iL, jL:jR] += (dt / (self.eps0 * self.dxd_iLm1)) * Hz_inc_l
        xd_r = self.grid.xd[iR - 1]
        Hz_inc_r = w.Hz_inc(xd_r, self.yd_v, t)
        Ey[iR, jL:jR] -= (dt / (self.eps0 * self.dxd_iRm1)) * Hz_inc_r
        # Ex corrections at j=jL and j=jR (Hz leakage)
        if jL > 0:
            yd_b = self.grid.yd[jL - 1]
            Hz_inc_b = w.Hz_inc(self.xd_h, yd_b, t)
            Ex[iL:iR, jL] -= (dt / (self.eps0 * self.dyd_jLm1)) * Hz_inc_b
        yd_t = self.grid.yd[jR - 1]
        Hz_inc_t = w.Hz_inc(self.xd_h, yd_t, t)
        Ex[iL:iR, jR] += (dt / (self.eps0 * self.dyd_jRm1)) * Hz_inc_t

    def set_dt(self, dt: float):
        self._dt = dt
