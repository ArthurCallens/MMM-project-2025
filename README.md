# Hybrid QM/EM Modelling of Electrons in 2-D Wells

A nonuniform-grid 2-D TE Yee-FDTD electromagnetic solver, coupled to a 2-D
time-dependent Schrödinger solver, implementing both parts of the *Multiscale and
Multiphysics Modeling Techniques for Nanoelectronic Devices* project.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). The app has four
tabs:

- **Part 1 — EM (FDTD):** set up the domain, grid, PML, incident plane wave (Gaussian or
  RF-modulated pulse, any incidence angle) and scatterers (Drude-dispersive dielectric,
  PEC, or PMC), then run and inspect field snapshots, observation-point traces, spectra,
  and — for a single circular PMC/dielectric scatterer — an overlay against the analytic
  cylindrical-scattering solution.
- **Part 2 — QM (Schrödinger):** set up a harmonic-oscillator well, initial state (ground
  state or a displaced coherent state), and an optional driving field, then run and
  inspect the probability density, expectation values, norm conservation, and the
  continuity-equation residual.
- **Part 3 — Coupled EM/QM:** place one or more wells inside an EM domain illuminated by a
  plane wave, with forward (E-field → Hamiltonian) and backward (quantum current →
  Ampère's-law source) coupling, and watch the induced dipole oscillation / re-radiation.
- **Report:** a self-contained, reproducible write-up (methodology, validation results,
  discussion) rendered live in the app, with a **PDF download** button.

## Package layout

```
emqm/
  constants.py     physical constants (SI)
  grid.py          nonuniform Yee-grid construction
  sources.py       Gaussian / RF-modulated pulses, ramped sine, FFT helper
  materials.py     scatterer shapes (circle/rectangle) and Drude/PEC/PMC material maps
  pml.py           convolutional PML (ADE realisation of a uniaxial PML)
  tfsf.py          total-field/scattered-field plane-wave injection (any incidence angle)
  fdtd_te.py       the 2-D TE FDTD time-stepping engine
  numba_kernels.py optional Numba-JIT-accelerated Drude ADE update
  quantum.py       2-D time-dependent Schrödinger solver (ADI Crank–Nicolson)
  coupling.py      forward/backward EM<->QM coupling, multi-well support
  validation.py    analytic reference solutions (plane wave, PMC/dielectric cylinder)
app.py             Streamlit front-end
report_content.py  report text + figures, used by the app's "Report" tab
```

Every routine in `emqm/` is plain, importable Python — it can be scripted directly (see
`report_content.py` for complete runnable examples) without going through the app.

## Performance

All FDTD/QM updates are vectorised NumPy array operations (no Python loops over grid
points, per the project brief's own recommendation). The most FLOP-heavy per-cell
operation, the Drude auxiliary-differential-equation update, additionally has an optional
Numba-JIT-compiled path (`engine="numba"` in `FDTD2D`, selectable in the Part-1 form of
the app) for large grids / long runs.

## Validation summary (see the Report tab for full details and figures)

- Plane-wave propagation timing matches the analytic free-space prediction to
  sub-attosecond precision; the CPML absorbs the outgoing pulse by >50 dB.
- The QM solver conserves norm to machine precision (Crank–Nicolson is exactly unitary),
  reproduces the correct 2-D harmonic-oscillator ground-state energy to ~1%, and
  reproduces the exact Ehrenfest/classical trajectory for a driven coherent state, with
  the residual error shrinking under grid refinement.
- The required PMC-cylinder scattering benchmark (dual of the classical TM/PEC-cylinder
  problem) shows correct qualitative/order-of-magnitude agreement with the analytic
  Bessel-series solution; a residual quantitative phase discrepancy at the smallest
  tested `ka` is discussed openly in the report as a known limitation of the staircased
  boundary representation, together with a concrete proposed refinement.

## Extra features beyond the minimal requirements

- Oblique-incidence TFSF injection at any angle (via analytic free-space field values, as
  suggested in the assignment).
- Periodic (in addition to PEC) outer boundary condition.
- Non-dispersive dielectric and dielectric-cylinder (Mie-series) validation, in addition
  to the required PMC-cylinder case.
- Multiple, independently driven, radiatively-coupled quantum wells sharing one EM field.
- Optional Numba-JIT acceleration.
