# Dam-break modelling with shallow-water hydrodynamics, SPH and ML surrogates — Tehri Dam

Code, knowledge graph and simulation data for the paper

> G. S. Navale, A. Shirode, S. Deshmukh, A. Dewangan and P. Agrawal, *Coupling Shallow-Water Hydrodynamics, Smoothed Particle Hydrodynamics and Machine-Learning Surrogates for Rapid Dam-Break Inundation Modelling: Application to Tehri Dam, India.*

Corresponding author: Ayush Shirode (ayush.shirode242@vit.edu), Department of Computer Engineering, Vishwakarma Institute of Technology, Pune.

## What is here

| Folder | Contents |
|---|---|
| `benchmarks/` | Solver verification against Ritter and Stoker, the SPH water-column test, and the 300-scenario synthetic-valley ensemble with its surrogates (paper Figs. 2–4, Tables 1–2) |
| `tehri/` | The Tehri pipeline: knowledge graph → scenario sampling → breach model + 2D solver on the Copernicus DEM → surrogates (paper Table 3, Figs. 5–6) |

All simulation results used in the paper are included, so the surrogate and figure scripts run without re-running the simulations. The one exception is `benchmarks/preds.npz` (test-set predictions, 17.5 MB), which is left out to keep the archive small: run `python surrogate.py` in `benchmarks/` (about 45 s) to regenerate it before `figs_d.py`.

## Installation

```bash
pip install -r requirements.txt
```

Run every script from inside its own folder (`cd benchmarks` or `cd tehri`); paths are relative.

## Tehri pipeline (`tehri/`)

**DEM (not included).** Download the Copernicus GLO-30 tile N30E078 and save it in `tehri/` as `dem_N30E078.tif`:

https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N30_00_E078_00_DEM/Copernicus_DSM_COG_10_N30_00_E078_00_DEM.tif

Attribution: Copernicus DEM GLO-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.

| Step | Command | Output | Time |
|---|---|---|---|
| 0. Reference grid | `python make_grid.py` | `grid_60m_transform.txt` | seconds |
| 1–2. Knowledge graph | `python kg_build.py` | `tehri_kg.ttl` (255 triples, 6 sources) | seconds |
| 3–4. Sample + simulate | `python sampler_ensemble.py 128 32` | `tehri_ensemble.npz`, `tehri_ensemble_meta.json`, `ens_ranges.json` | ~40 s per scenario per core (~1 h on 2 cores) |
| 5. Surrogates | `python surrogate_tehri.py` | `res_tehri.json`, `preds_tehri.npz` | ~15 s |
| Sensitivity | `python sensitivity.py` | `spearman.json` | seconds |
| Figures | `python figs_tehri.py` | `fig6_tehri.png`, `fig7_tehri_ml.png` | ~1 min |

Steps 0 and 3–4 and the figure script need the DEM. Steps 1, 5 and the sensitivity script run from the included files alone.

### Knowledge graph

`tehri_kg.ttl` is an RDF/Turtle graph linking dam → reservoir → river reach → land-cover class → roughness. Every numerical node has a value or range, a unit, a `dbk:status` of `sourced`, `derived` or `assumption`, and a `dbk:source` link to a citation and URL. `sampler_ensemble.py` draws scenario ranges only from SPARQL queries on this graph. The one unsourced range, a ±30% breach-width factor, is stored with status `assumption`.

### Files in `tehri/`

| File | Description |
|---|---|
| `breach.py` | Level-pool reservoir routing, Froehlich (2008) / Von Thun–Gillette (1990) breach growth, critical-flow outflow |
| `swe_dem.py` | 2D shallow-water finite-volume solver (HLL, hydrostatic reconstruction, semi-implicit Manning friction) |
| `domain.py` | 90 m UTM-44N model domain from the DEM, river centreline, gauges |
| `tehri_ensemble.npz` | 160 scenarios: inputs `P` (level, mode, t_f fraction, width factor, Manning n), `split` (0 train, 1 test), max-depth fields `H`, gauge `peak` depths and `arr`ival times (s) at 5, 10, 17, 25, 33, 38.8 km, flooded `area` (km²), peak breach outflow `Qp` (m³/s) and its time `tQp`, solver runtime `rt` (s) |
| `tehri_ensemble_meta.json` | Per-scenario breach parameters, including the Froehlich (1995) peak estimate |
| `preds_tehri.npz` | Test-set surrogate predictions |

## Benchmarks (`benchmarks/`)

| Command | Output |
|---|---|
| `python fv1d.py` | Ritter/Stoker verification (`fv1d.json`) |
| `python sph2d.py 0.02` and `python sph2d.py 0.01` | SPH water-column collapse (`sph_0.02.json`, `sph_0.01.json`; the 0.01 m run takes ~2 min) |
| `python ensemble.py` | 300-scenario synthetic-valley ensemble (`ensemble.npz`, ~12 min) |
| `python surrogate.py` | Surrogate training and testing (`res_d.json`, `preds.npz`) |
| `python figs_ab.py`, `python figs_d.py`, `python fig1.py` | Paper figures 1–4 |

## Limitations

There is no observed Tehri breach, so surrogate accuracy is measured against the physics solver, not against reality. Both breach regressions are applied outside their calibration ranges (dams 3–93 m high). See the paper's limitations section for the full list.

## Licence

Code: MIT (see `LICENSE`). Data files and the knowledge graph: CC BY 4.0. The Copernicus DEM is not redistributed and remains under its own licence.
