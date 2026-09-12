# Development and maintenance

[中文](../zh/development.md) · [User guide](README.md) · [Project home](../../README.md)

## Layout and responsibilities

```text
DichromaticMap/
├── main.py                     # Only root-level source launcher
├── pyproject.toml              # Metadata, dependencies, entry point and resources
├── README.md                   # English home page with a Chinese documentation link
├── docs/
│   ├── en/                     # English user and developer guides
│   └── zh/                     # Chinese user and developer guides
├── src/dichromatic_map/
│   ├── __init__.py             # Common numerical API
│   ├── __main__.py             # Argument parsing and application launch
│   ├── crystal.py              # Lattices, integer axes, projections, vector frames
│   ├── cells.py                # Cell geometry, result data and complete-region counts
│   ├── matching.py             # Exact CSL and local same-layer matching
│   ├── strain.py               # Uniform search and selected-cell strain fitting
│   ├── compute.py              # Workers, executors and asynchronous NearSearch
│   ├── state.py                # Parameters, selections, filters and result state
│   └── ui/
│       ├── __init__.py         # Display conventions; no Qt initialization
│       ├── controls.py         # Widgets, layout and application theme
│       ├── plot.py             # Graphics, display transforms, annotations and export
│       ├── window.py           # Interaction and workflow coordination
│       └── resources/          # theme.qss and arrow SVGs
├── test/                       # Numerical, GUI, worker and launcher regressions
└── legacy/                     # Historical Matplotlib comparison implementation
```

`src` is the source container; `dichromatic_map` is the importable package name.
Do not put `src.` in imports. The old root compatibility modules have been
removed. Import numerical functions from the package, not from `main.py`.
The source launcher only adds the checkout's `src` directory to the import path
and calls the package's `main()`; it does not maintain another application.
Installed commands call the same function. The main guard prevents a spawned
process from opening another window when it imports the launcher.

## Dependency and extension boundaries

Numerical modules depend on NumPy and the standard library, never on the UI.
`state.py` does not store Qt widgets, and `compute.py` does not draw graphics.
Worker functions are defined at module scope for `spawn`. Pass arrays and
numerical results across processes, not QWidget instances or window objects.
When `threadpoolctl` is available, the worker initializer limits nested BLAS
threads; this dependency is optional.

The window uses composition:

| Object | Ownership |
| --- | --- |
| `window.state` | Physical configuration, selections, display filters, results and cache identifiers |
| `window.compute` | Process/thread pools, pending work and task generations |
| `window.controls` | Widgets and layout |
| `window.plot` | Graphics items, display coordinate transforms, annotations and export |

The window coordinates user actions, state changes and timer-based polling.
Remaining forwarding properties expose the same component data; they do not
duplicate state. New code should access components directly rather than expand
the historical forwarding interface. There is no multiple-inheritance framework,
plugin registry or class-per-file hierarchy.

When extending the application:

1. Put a new numerical algorithm in the relevant module, such as `matching.py`
   or `strain.py`, and add independent numerical tests first.
2. If it needs background work, add a module-level worker to `compute.py` and
   coordinate submission/polling in the window. Invalidate obsolete requests so
   stale results cannot overwrite the current configuration. Release execution
   resources when the window closes.
3. Put controls in `ui/controls.py`, graphics and annotations in `ui/plot.py`,
   and the interaction workflow in `ui/window.py`.
4. Edit theme assets only in `ui/resources/`. New resource types also need
   entries in the package-data section of `pyproject.toml`.
5. Update both documentation languages when behavior changes. Introduce new
   abstractions when a second concrete implementation actually needs them.

Display rotation, layer visibility and viewport cropping must not rewrite
physical atom coordinates. Count the complete selected region rather than
visible scatter points. Never replace a manual cell's actual per-grain vertices
with local matching midpoints for counting.

## Python API

For imports from outside the repository, perform an editable installation in
the chosen conda environment, from the project root:

```bash
python -m pip install -e .
# Include GUI dependencies if needed:
python -m pip install -e ".[gui]"
```

Edits under `src/` then take effect without reinstalling. Reinstall after changing
package metadata, dependency declarations or entry-point configuration.

```python
import numpy as np
from dichromatic_map import get_geometry, projected_columns, count_cell_atoms
from dichromatic_map.crystal import csl_angle_deg
from dichromatic_map.matching import exact_csl_cell, local_near_pairs

geometry = get_geometry("FCC", "110")
angle = csl_angle_deg(4, 1, "110")
g1 = projected_columns(12, 9, angle / 2, lattice="FCC", axis="110")
g2 = projected_columns(12, 9, -angle / 2, lattice="FCC", axis="110")
pairs = local_near_pairs(g1, g2, 0.05)
cell = exact_csl_cell(angle, lattice="FCC", axis="110")
assert cell is not None
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ cell.cell.T
counts = count_cell_atoms(vertices, angle, (cell.f1, cell.f2), layer=1)
print(geometry.layer_count, counts.half_open[:, 1])
```

| API / data | Convention |
| --- | --- |
| `get_geometry(lattice, axis)` | Normalizes the axis and returns computed layer count, spacing, axial repeat and geometry |
| `projected_columns(...)` | Returns `ProjectedGrain`: `positions` is `(N,2)`, `layers` is `(N,)`, `half_indices` is `(N,3)` |
| `half_indices` | Integer coordinates in a₀/2, not normalized 2D plot positions |
| `same_layer_coincidence_sites(g1, g2, tolerance)` | Available in `matching.py`; returns exact coincidence positions by layer |
| `local_near_pairs(g1, g2, distance, layers=None)` | Same-layer mutual nearest pairs; does not move input atoms |
| `exact_csl_cell(angle, lattice, axis)` | A layer-preserving common translation cell, or `None`; not an arbitrary manual region |
| `count_cell_atoms(vertices, angle, deformations, ...)` | Shared `(4,2)` or independent `(2,4,2)` grain polygons; result arrays are indexed by grain and layer |
| `layer=-1` | The counting API's all-layer default; the GUI explicitly passes the manual cell's selected layer instead |
| `strain_selected_cell(...)` | From `strain.py`; accepts original paired `(2,4,2)` vertices and returns deformations, translations and strain data without modifying a window |

Lengths are in a₀, angle arguments are degrees, and in-plane deformations are
`(2,2)` matrices. Invalid input generally raises `ValueError`; interactive resource
limits raise its subclass `GeometryLimitError`. Function docstrings specify the
complete signatures and conventions.

## Tests and documentation maintenance

The full suite needs NumPy, PySide6, PyQtGraph and Matplotlib for historical
comparisons. With those dependencies present, no project installation is needed:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s test -v
```

Alternatively, `python -m pip install -e ".[test]"` installs the test dependencies.
Run from the project root. Tests configure the source import path and offscreen
Qt; temporary exports use temporary directories. Coverage includes lattice
enumeration, CSL/strain, complete cell counts, visibility/rotation, vector
coordinates, stale work, spawned results, source launching outside the checkout
directory and PNG export.

Documentation is plain Markdown, readable in an editor or repository browser.
There is no Sphinx/MkDocs dependency or requirement to generate `docs/_build/`.
Links are relative; check cross-language navigation after moving files.

## Why is there a build directory?

Running this project does not require a build. `python main.py` executes the
source directly. The previous `build/` directory was produced while validating
a package with `pip wheel`, not by a runtime requirement of the viewer.

This project has no custom C/C++ extensions. Building its wheel primarily
collects Python source, theme/icons and installation metadata into a distributable
package; it does not compile this project into a native executable.
The `[build-system]` section of `pyproject.toml` chooses the backend for
installation/packaging, not an operation required before every run. Keeping this
configuration makes the package installable and ensures UI resources are included.

| Artifact | Purpose and cleanup considerations |
| --- | --- |
| `build/` | Intermediate packaging copies; normally removable after a build |
| `dist/` | Explicitly generated wheel/sdist releases; keep any releases you still need |
| `__pycache__/`, `.pyc` | Python's import-time bytecode cache; removable and may be regenerated automatically |
| `*.egg-info/` | Build/installation metadata; stale build copies may be removed, but an active editable installation can depend on it |
| `.pytest_cache/` and similar | Regenerable test-tool caches |

If you delete metadata needed by an editable installation, run the appropriate
`pip install -e ...` command again. Cleaning a project's artifacts should never
mean deleting a conda environment or its `site-packages`. Source files,
`pyproject.toml`, QSS, SVGs and documentation are not temporary artifacts.

For a run that should not write bytecode caches, use `python -B main.py` or set
`PYTHONDONTWRITEBYTECODE=1`. This is an optional cache policy; the appearance of
`__pycache__/` does not mean the project requires a separate build.
