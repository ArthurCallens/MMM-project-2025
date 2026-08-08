"""Scatterer geometry (shape masks) and material parameters (Drude / PEC / PMC).

A non-dispersive dielectric object is simply a Drude scatterer with omega_p = 0 (no
plasma response, constant relative permittivity eps_inf) -- this keeps a single code
path for the Ampere-law auxiliary-current (ADE) update.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class Shape:
    def mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Circle(Shape):
    cx: float
    cy: float
    r: float

    def mask(self, X, Y):
        return (X - self.cx) ** 2 + (Y - self.cy) ** 2 <= self.r ** 2


@dataclass
class Rectangle(Shape):
    x0: float
    x1: float
    y0: float
    y1: float

    def mask(self, X, Y):
        return (X >= self.x0) & (X <= self.x1) & (Y >= self.y0) & (Y <= self.y1)


@dataclass
class DrudeScatterer:
    """Drude-dispersive (or, if omega_p=0, plain non-dispersive dielectric) object.

    eps(omega) = eps_inf*eps0 - eps0*omega_p^2 / (omega^2 + j*omega*gamma)
    """

    shape: Shape
    eps_inf: float = 1.0
    omega_p: float = 0.0   # plasma frequency [rad/s]
    gamma: float = 0.0     # collision frequency [rad/s]
    name: str = "drude"


@dataclass
class PECScatterer:
    shape: Shape
    name: str = "pec"


@dataclass
class PMCScatterer:
    shape: Shape
    name: str = "pmc"


@dataclass
class MaterialMaps:
    """Per-field material coefficient maps, precomputed on the Yee-staggered grids."""

    eps_inf_ex: np.ndarray
    eps_inf_ey: np.ndarray
    omega_p2_ex: np.ndarray
    omega_p2_ey: np.ndarray
    gamma_ex: np.ndarray
    gamma_ey: np.ndarray
    pec_ex: np.ndarray  # bool, forces Ex=0
    pec_ey: np.ndarray  # bool, forces Ey=0
    pmc_hz: np.ndarray  # bool, forces Hz=0


def build_material_maps(grid, scatterers) -> MaterialMaps:
    eps_inf_ex = np.ones_like(grid.X_ex)
    eps_inf_ey = np.ones_like(grid.X_ey)
    omega_p2_ex = np.zeros_like(grid.X_ex)
    omega_p2_ey = np.zeros_like(grid.X_ey)
    gamma_ex = np.zeros_like(grid.X_ex)
    gamma_ey = np.zeros_like(grid.X_ey)
    pec_ex = np.zeros_like(grid.X_ex, dtype=bool)
    pec_ey = np.zeros_like(grid.X_ey, dtype=bool)
    pmc_hz = np.zeros_like(grid.X_hz, dtype=bool)

    for s in scatterers:
        if isinstance(s, DrudeScatterer):
            mx = s.shape.mask(grid.X_ex, grid.Y_ex)
            my = s.shape.mask(grid.X_ey, grid.Y_ey)
            eps_inf_ex[mx] = s.eps_inf
            eps_inf_ey[my] = s.eps_inf
            omega_p2_ex[mx] = s.omega_p ** 2
            omega_p2_ey[my] = s.omega_p ** 2
            gamma_ex[mx] = s.gamma
            gamma_ey[my] = s.gamma
        elif isinstance(s, PECScatterer):
            pec_ex |= s.shape.mask(grid.X_ex, grid.Y_ex)
            pec_ey |= s.shape.mask(grid.X_ey, grid.Y_ey)
        elif isinstance(s, PMCScatterer):
            pmc_hz |= s.shape.mask(grid.X_hz, grid.Y_hz)
        else:
            raise TypeError(f"Unknown scatterer type: {type(s)}")

    return MaterialMaps(
        eps_inf_ex, eps_inf_ey, omega_p2_ex, omega_p2_ey, gamma_ex, gamma_ey,
        pec_ex, pec_ey, pmc_hz,
    )
