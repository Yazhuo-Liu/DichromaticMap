# DichromaticMap

[中文文档](docs/zh/README.md) · [English guide](docs/en/README.md)

A standalone Qt viewer for layer-resolved FCC/BCC tilt grain-boundary
dichromatic patterns. Numerical calculations can be used independently of Qt;
the project does not depend on GBClaw.

## Quick start

Use Python 3.10+ in your chosen conda environment. Activate that environment,
then install the GUI dependencies and launch from the project root:

```bash
python -m pip install numpy PySide6 pyqtgraph
python main.py
python main.py --lattice BCC --axis 100 --workers 4
python main.py --axis "1 -1 3"
python main.py --help
```

Launching from the source directory does not require installing or building this
project. No C compiler is needed.

To install the package for use outside the source directory, run
`python -m pip install ".[gui]"` from the project root. You can then launch with
`python -m dichromatic_map` or `dichromatic-map`.
For numerical use only, `python -m pip install .` installs the NumPy-based core
without GUI dependencies.

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

[English user guide](docs/en/README.md) · [中文使用手册](docs/zh/README.md)

The guides cover controls, counting conventions, strain calculations, PNG export
and troubleshooting. They are plain Markdown and require no documentation build.
