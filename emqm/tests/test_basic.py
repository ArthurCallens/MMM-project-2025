"""Fast smoke/validation tests, runnable with `pytest emqm/tests`.

These are deliberately small/fast (a few seconds total) so they can be run after every
change; see report_content.py for the larger, more thorough validation runs shown in the
app's Report tab.
"""
import numpy as np

from emqm.constants import HBAR, ME, NM
from emqm.fdtd_te import FDTD2D
from emqm.grid import Grid2D, uniform_axis
from emqm.materials import build_material_maps
from emqm.pml import PMLParams
from emqm.quantum import Schrodinger2D
from emqm.sources import GaussianPulse
from emqm.tfsf import PlaneWave
from emqm.validation import free_space_plane_wave_delay


def test_plane_wave_arrival_time():
    Lx = Ly = 160e-9
    dx = 4e-9
    grid = Grid2D(uniform_axis(Lx, dx), uniform_axis(Ly, dx))
    mats = build_material_maps(grid, [])
    dt = grid.cfl_dt(0.9)
    pml = PMLParams(thickness=8 * dx)
    sim = FDTD2D(grid, dt, mats, pml_x=pml, pml_y=pml)
    pulse = GaussianPulse(amplitude=1.0, sigma=5e-16, tc=6 * 5e-16)
    sim.add_tfsf(PlaneWave(pulse, theta_deg=0.0), 12, grid.Nx - 12, 12, grid.Ny - 12)
    obs = sim.add_observer(Lx * 0.5, Ly * 0.5)
    for _ in range(500):
        sim.step()
    t_arr = np.array(obs.t)
    peak_t = t_arr[np.argmax(np.abs(np.array(obs.Ey)))]
    expected = free_space_plane_wave_delay(Lx * 0.5, Ly * 0.5, 0.0) + pulse.tc
    assert abs(peak_t - expected) < 2 * dt


def test_fdtd_stays_finite_and_pml_absorbs():
    Lx = Ly = 160e-9
    dx = 4e-9
    grid = Grid2D(uniform_axis(Lx, dx), uniform_axis(Ly, dx))
    mats = build_material_maps(grid, [])
    dt = grid.cfl_dt(0.9)
    pml = PMLParams(thickness=8 * dx)
    sim = FDTD2D(grid, dt, mats, pml_x=pml, pml_y=pml)
    pulse = GaussianPulse(amplitude=1.0, sigma=5e-16, tc=6 * 5e-16)
    sim.add_tfsf(PlaneWave(pulse, theta_deg=0.0), 12, grid.Nx - 12, 12, grid.Ny - 12)
    energies = []
    for n in range(700):
        sim.step()
        if n % 20 == 0:
            energies.append(np.sum(sim.Ex ** 2) + np.sum(sim.Ey ** 2) + np.sum(sim.Hz ** 2))
    assert np.isfinite(sim.Hz).all()
    assert energies[-1] < 0.01 * max(energies)


def test_qm_norm_conservation_and_ground_state_energy():
    omega = 50e14
    qm = Schrodinger2D(Lx=5 * NM, Ly=5 * NM, dx=0.05 * NM, m_eff=0.15 * ME, omega_ho=omega)
    qm.set_state(qm.ground_state())
    dt = (2 * np.pi / omega) / 400
    for _ in range(200):
        qm.step(dt)
    assert abs(qm.norm() - 1.0) < 1e-10
    Ekin = qm.expectation_kinetic_energy()
    # 2-D isotropic HO ground state: <T> = <V> = E_0/2 = hbar*omega/2 (virial theorem)
    assert abs(Ekin - HBAR * omega / 2) / (HBAR * omega / 2) < 0.05


def test_qm_ehrenfest_oscillation():
    omega = 50e14
    x0 = 0.4 * NM
    qm = Schrodinger2D(Lx=6 * NM, Ly=6 * NM, dx=0.05 * NM, m_eff=0.15 * ME, omega_ho=omega, boundary_radius=2.6 * NM)
    qm.set_state(qm.coherent_state(x0=x0))
    T = 2 * np.pi / omega
    dt = T / 300
    ts, xs = [], []
    for _ in range(300):
        qm.step(dt)
        ex, _ = qm.expectation_xy()
        ts.append(qm.t)
        xs.append(ex)
    ts, xs = np.array(ts), np.array(xs)
    err = np.max(np.abs(xs - x0 * np.cos(omega * ts))) / x0
    assert err < 0.05
    assert abs(qm.norm() - 1.0) < 1e-10
