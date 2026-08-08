"""Physical constants (SI units) used throughout the solver."""

EPS0 = 8.8541878128e-12       # vacuum permittivity [F/m]
MU0 = 1.25663706212e-6        # vacuum permeability [H/m]
C0 = 1.0 / (EPS0 * MU0) ** 0.5  # speed of light in vacuum [m/s]
ETA0 = (MU0 / EPS0) ** 0.5    # vacuum impedance [Ohm]

HBAR = 1.054571817e-34        # reduced Planck constant [J s]
ME = 9.1093837015e-31         # electron rest mass [kg]
QE = 1.602176634e-19          # elementary charge [C] (electron charge = -QE)

NM = 1e-9                     # nanometre, for convenience
FS = 1e-15                    # femtosecond, for convenience
