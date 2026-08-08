"""Analytic reference solutions used to validate the FDTD solver (Sec. 3.4 of Part 1).

- PMC cylinder: the 2-D TE problem of a plane wave scattering off an infinite PMC
  cylinder is the exact dual of the classical 2-D TM/PEC-cylinder problem (footnote 4 of
  the assignment): replace Ez -> Hz, and the boundary condition "tangential E = 0" on the
  PEC cylinder becomes "tangential H = 0" (i.e. Hz_total = 0) on the PMC cylinder. The
  cylindrical-harmonic series below is the standard PEC/TM-cylinder result with that
  substitution.
- Dielectric cylinder: the standard two-region cylindrical-wave-matching solution
  (continuity of Hz and of the tangential E, i.e. of (1/eps) dHz/drho, at rho=a).

Both return the *frequency-domain* total (or scattered-only) Hz at a fixed observation
point, as a function of an array of angular frequencies -- exactly what's needed to
compare against an FFT of a broadband time-domain FDTD recording (Sec. 3.3.2 of Part 1).
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp

from .constants import C0


def _global_phase(omega, cyl_x: float, cyl_y: float, theta_deg: float):
    """exp(-j k (cyl_x cos(theta) + cyl_y sin(theta))): the extra phase between a plane
    wave phasor referenced to the *cylinder centre* (rho=0 in the Bessel expansion, the
    natural reference for these formulas) and one referenced to the *global* coordinate
    origin (which is what an FDTD source spectrum, and hence a simulated response, is
    normally referenced to -- see `sources.GaussianPulse.spectrum_complex`). Needed
    whenever the scatterer is not placed exactly at (0, 0) in the simulation."""
    theta = np.deg2rad(theta_deg)
    k = np.atleast_1d(np.asarray(omega, dtype=float)) / C0
    return np.exp(-1j * k * (cyl_x * np.cos(theta) + cyl_y * np.sin(theta)))


def pmc_cylinder_hz(omega, rho: float, phi: float, a: float, nmax: int = 40, kind: str = "total",
                     cyl_x: float = 0.0, cyl_y: float = 0.0, theta_deg: float = 0.0):
    """Hz(omega) at polar point (rho,phi) relative to the cylinder centre (rho >= a),
    unit-amplitude incident plane wave (Hz_inc = exp(-j k x) = exp(-j k rho cos(phi)))
    travelling along +x, scattering off a PMC cylinder of radius a. If the cylinder is
    not at the global coordinate origin (cyl_x, cyl_y) != (0, 0), the result is rescaled
    by the appropriate global-origin-referenced phase (see `_global_phase`) so it can be
    compared directly against a simulated, source-spectrum-normalised FDTD response."""
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    k = omega / C0
    n = np.arange(-nmax, nmax + 1)
    K, N = np.meshgrid(k, n, indexing="ij")
    ka, krho = K * a, K * rho
    coef = np.exp(-1j * N * np.pi / 2)
    ratio = jv(N, ka) / hankel2(N, ka)
    scat = -coef * ratio * hankel2(N, krho) * np.exp(-1j * N * phi)
    if kind == "scattered":
        result = scat.sum(axis=1)
    else:
        inc = coef * jv(N, krho) * np.exp(-1j * N * phi)
        result = (inc + scat).sum(axis=1)
    return result * _global_phase(omega, cyl_x, cyl_y, theta_deg)


def dielectric_cylinder_hz(omega, rho: float, phi: float, a: float, eps_r: float, nmax: int = 40, kind: str = "total",
                            cyl_x: float = 0.0, cyl_y: float = 0.0, theta_deg: float = 0.0):
    """Hz(omega) at (rho,phi) for a plane wave scattering off a non-magnetic dielectric
    cylinder (relative permittivity eps_r, radius a). Valid for both rho>=a (exterior,
    total = incident + scattered) and rho<a (interior transmitted field)."""
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    k0 = omega / C0
    k1 = k0 * np.sqrt(eps_r)
    n = np.arange(-nmax, nmax + 1)
    K0, N = np.meshgrid(k0, n, indexing="ij")
    K1, _ = np.meshgrid(k1, n, indexing="ij")
    k0a, k1a = K0 * a, K1 * a

    J0a, J0ap = jv(N, k0a), jvp(N, k0a)
    H0a, H0ap = hankel2(N, k0a), h2vp(N, k0a)
    J1a, J1ap = jv(N, k1a), jvp(N, k1a)

    A11, A12 = H0a, -J1a
    A21, A22 = K0 * H0ap, -(K1 / eps_r) * J1ap
    b1, b2 = -J0a, -K0 * J0ap
    det = A11 * A22 - A12 * A21
    an = (b1 * A22 - A12 * b2) / det
    bn = (A11 * b2 - b1 * A21) / det

    coef = np.exp(-1j * N * np.pi / 2)
    if rho >= a:
        if kind == "scattered":
            Hrho = an * hankel2(N, K0 * rho)
        else:
            Hrho = jv(N, K0 * rho) + an * hankel2(N, K0 * rho)
    else:
        Hrho = bn * jv(N, K1 * rho)
    result = (coef * Hrho * np.exp(-1j * N * phi)).sum(axis=1)
    return result * _global_phase(omega, cyl_x, cyl_y, theta_deg)


def free_space_plane_wave_delay(x: float, y: float, theta_deg: float = 0.0) -> float:
    """t - t_source at which the incident pulse peak reaches (x,y): a trivial but useful
    time-domain sanity check (Sec. 3.4-1: visual/analytic inspection of the travelling
    wave)."""
    theta = np.deg2rad(theta_deg)
    return (x * np.cos(theta) + y * np.sin(theta)) / C0
