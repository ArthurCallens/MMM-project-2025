"""Temporal profiles for the incident plane wave, and their (analytic) spectra.

All profiles are causal in the sense that tc is chosen large enough (tc >= 5*sigma by
default) that p(t) ~ 0 for t < 0 to good approximation, as recommended in the project
description, so that the closed-form Fourier magnitude below is accurate and can be used
to normalise frequency-domain results (division by source spectrum).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianPulse:
    """p(t) = A * exp(-(t-tc)^2 / (2*sigma^2))."""

    amplitude: float
    sigma: float
    tc: float | None = None  # default: 5*sigma (recommended minimum)

    def __post_init__(self):
        if self.tc is None:
            self.tc = 5.0 * self.sigma

    def omega_max(self, level: float = 3.0) -> float:
        """Angular frequency at which |spectrum| has dropped to exp(-level^2/2) of DC."""
        return level / self.sigma

    def __call__(self, t):
        t = np.asarray(t)
        return self.amplitude * np.exp(-((t - self.tc) ** 2) / (2.0 * self.sigma ** 2))

    def spectrum(self, omega):
        """Analytic |P(omega)| = A*sqrt(2*pi)*sigma*exp(-sigma^2 omega^2/2) (magnitude,
        DC-pulse centred at t=0; the tc-delay only adds a linear phase, exp(-j*omega*tc))."""
        omega = np.asarray(omega, dtype=float)
        return self.amplitude * np.sqrt(2 * np.pi) * self.sigma * np.exp(-(self.sigma ** 2) * omega ** 2 / 2.0)

    def spectrum_complex(self, omega):
        omega = np.asarray(omega, dtype=float)
        return self.spectrum(omega) * np.exp(-1j * omega * self.tc)


@dataclass
class ModulatedGaussianPulse:
    """Gaussian-modulated sinusoidal (RF) pulse: p(t)=A*exp(-(t-tc)^2/2sigma^2)*sin(wc*t)."""

    amplitude: float
    sigma: float
    omega_c: float
    tc: float | None = None

    def __post_init__(self):
        if self.tc is None:
            self.tc = 5.0 * self.sigma

    def omega_max(self, level: float = 3.0) -> float:
        return self.omega_c + level / self.sigma

    def __call__(self, t):
        t = np.asarray(t)
        return self.amplitude * np.exp(-((t - self.tc) ** 2) / (2.0 * self.sigma ** 2)) * np.sin(self.omega_c * t)

    def spectrum(self, omega):
        """Magnitude of the two Gaussian side-lobes centred at +/- omega_c."""
        omega = np.asarray(omega, dtype=float)
        g = lambda w: self.amplitude * np.sqrt(2 * np.pi) * self.sigma / 2.0 * np.exp(-(self.sigma ** 2) * w ** 2 / 2.0)
        return g(omega - self.omega_c) + g(omega + self.omega_c)


@dataclass
class RampedSine:
    """Monochromatic sine, smoothly ramped on over `ramp_cycles` periods.

    p(t) = A * ramp(t) * sin(omega_c * t), ramp = sin^2 smooth-step over [0, t_ramp].
    Used for the monochromatic-excitation validation of the QM part (Sec. 4 of Part 2).
    """

    amplitude: float
    omega_c: float
    ramp_cycles: float = 5.0

    def t_ramp(self) -> float:
        return self.ramp_cycles * 2 * np.pi / self.omega_c

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        tr = self.t_ramp()
        ramp = np.clip(t / tr, 0.0, 1.0)
        ramp = np.sin(0.5 * np.pi * ramp) ** 2
        return self.amplitude * ramp * np.sin(self.omega_c * t) * (t >= 0)


def fft_frequency_response(t: np.ndarray, signal: np.ndarray, band: tuple[float, float] | None = None):
    """FFT a recorded time-domain signal -> (omega, complex spectrum), numpy.fft convention
    exp(-j*omega*t) (numpy uses exp(-2j*pi*f*t) forward transform, consistent sign).

    If `band` = (omega_min, omega_max) is given, values outside the band are discarded
    (avoids dividing by ~0 source content outside the excitation bandwidth, as warned
    in the project description).
    """
    n = len(t)
    dt = t[1] - t[0]
    freq = np.fft.fftfreq(n, d=dt)
    omega = 2 * np.pi * freq
    spec = np.fft.fft(signal) * dt  # continuous-FT approximation
    order = np.argsort(omega)
    omega, spec = omega[order], spec[order]
    if band is not None:
        lo, hi = band
        mask = (np.abs(omega) >= lo) & (np.abs(omega) <= hi)
        omega, spec = omega[mask], spec[mask]
    return omega, spec
