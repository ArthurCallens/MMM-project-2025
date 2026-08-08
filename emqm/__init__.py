"""emqm: coupled 2-D EM (FDTD) / QM (Schrodinger) solver for electrons in nanoscale wells.

Part 1 - 2-D TE nonuniform Yee-FDTD solver with CPML/UPML absorbing boundaries,
         TFSF plane-wave injection (axis-aligned and oblique), and Drude / PEC / PMC
         scatterers.
Part 2 - 2-D time-dependent Schrodinger solver (ADI Crank-Nicolson) for an electron in
         a harmonic-oscillator well, forward-coupled to the local E-field (length gauge)
         and backward-coupled to the EM grid via the quantum current density.
"""

from . import constants  # noqa: F401

__version__ = "1.0.0"
