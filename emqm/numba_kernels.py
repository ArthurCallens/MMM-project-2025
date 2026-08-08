"""Optional Numba JIT acceleration for the hottest per-cell arithmetic.

The Drude auxiliary-differential-equation (ADE) update is the most FLOP-heavy elementwise
operation performed every single time step (over the full Ex / Ey interior arrays), so it
is the natural target for JIT compilation: a manually looped, @njit(parallel=True)
compiled version avoids NumPy's temporary-array allocations and is typically faster than
the equivalent vectorised NumPy expression on large grids. If Numba is not installed, the
plain NumPy vectorised implementation is used transparently -- results are identical
either way, only speed differs.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only when numba isn't installed
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # no-op fallback decorator
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

    def prange(n):
        return range(n)


def drude_update_numpy(E, J, RHS, C1, C2, beta, denom_inv, factor_Jn):
    E_new = (E * (1.0 - beta) + RHS - factor_Jn * J) * denom_inv
    J_new = C1 * J + C2 * (E_new + E)
    return E_new, J_new


@njit(cache=True, parallel=True)
def _drude_update_numba(E, J, RHS, C1, C2, beta, denom_inv, factor_Jn):
    Nx, Ny = E.shape
    E_new = np.empty_like(E)
    J_new = np.empty_like(J)
    for i in prange(Nx):
        for j in range(Ny):
            e = E[i, j]
            jc = J[i, j]
            en = (e * (1.0 - beta[i, j]) + RHS[i, j] - factor_Jn[i, j] * jc) * denom_inv[i, j]
            E_new[i, j] = en
            J_new[i, j] = C1[i, j] * jc + C2[i, j] * (en + e)
    return E_new, J_new


def drude_update(E, J, RHS, C1, C2, beta, denom_inv, factor_Jn, engine: str = "numpy"):
    if engine == "numba" and HAVE_NUMBA:
        return _drude_update_numba(
            np.ascontiguousarray(E), np.ascontiguousarray(J), np.ascontiguousarray(RHS),
            C1, C2, beta, denom_inv, factor_Jn,
        )
    return drude_update_numpy(E, J, RHS, C1, C2, beta, denom_inv, factor_Jn)
