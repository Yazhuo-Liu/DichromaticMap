# DichromaticMap

[中文文档](docs/zh/README.md) · [English guide](docs/en/README.md) · [Development](docs/en/development.md)

A standalone Qt viewer for layer-resolved FCC/BCC tilt grain-boundary
dichromatic patterns. Numerical calculations can be used independently of Qt;
the project does not depend on GBClaw.

## Quick start

Use Python 3.10+ in a conda environment with NumPy, PySide6 and PyQtGraph.
On the current development machine, these GUI dependencies are in `base`:

```bash
conda activate base
python main.py
python main.py --lattice BCC --axis 100 --workers 4
python main.py --axis "1 -1 3"
python main.py --help
```

Run these commands from the project root. If dependencies are missing, install
them in the active environment with `python -m pip install numpy PySide6 pyqtgraph`.
Launching from the checkout does not require installing or building this project.
`main.py` is the only root-level Python launcher; the old root compatibility
scripts have been removed.

To import the package from other projects, optionally run
`python -m pip install -e .` from the root, or use `".[gui]"` to include GUI
dependencies. Installed entry points are `python -m dichromatic_map` and
`dichromatic-map`.

## Features

- FCC/BCC geometry for preset and custom integer tilt axes, in normalized a₀ units.
- Independent visibility for each grain and axial layer, with matching marker icons.
- Exact same-layer CSL, local near-pair matching and homogeneous-strain cell search.
- GB-side filtering and P1→P2 vector measurements in both grain coordinate frames.
- Manual same-layer cells, separate G1/G2 atom counts and selected-cell bulk-strain fitting.
- Display-only rotation, background computation and PNG export.

A geometric selection is not automatically a periodic cell. Local matching does
not move atoms; applying bulk strain is an explicit geometric transformation,
not an energy relaxation. See the guides for counting and strain conventions.

## Documentation

| Topic | English | 中文 |
| --- | --- | --- |
| Installation, controls and scientific conventions | [User guide](docs/en/README.md) | [使用手册](docs/zh/README.md) |
| Structure, Python API, tests and packaging | [Development](docs/en/development.md) | [开发与维护](docs/zh/development.md) |

Documentation is plain Markdown under `docs/en/` and `docs/zh/`; no documentation
build step is required. The application lives in `src/dichromatic_map/`.
`legacy/` is retained only for historical reference and regression comparisons.

## Tests

The full suite additionally needs Matplotlib for the historical viewer comparisons.
Run from the project root in an environment with the test dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s test -v
```

Tests use an offscreen Qt backend. Normal Python execution may recreate
`__pycache__/`; this is bytecode caching, not a required project build.
