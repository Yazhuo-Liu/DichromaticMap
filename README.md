<h1 align="center"><img src="docs/images/dichromaticmap_logo_with_title.svg" alt="DichromaticMap — open-source tools for dichromatic pattern analysis" width="720"></h1>

[Project website](https://yazhuo-liu.github.io/DichromaticMap/) · [Use online](https://yazhuo-liu.github.io/DichromaticMap/use.html) · [English user guide](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/en/README.md) · [中文使用手册](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/zh/README.md)

If you find DichromaticMap helpful, please cite:

> K.Q. Ding, Y.Z. Liu, Y. Zhang, L.H. Wang, X.D. Han, T. Zhu,
> “Misfit-dislocation hierarchy governs sliding of asymmetric non-CSL grain
> boundaries,” *arXiv preprint* arXiv:2609.14673 (2026).
> [https://doi.org/10.48550/arXiv.2609.14673](https://doi.org/10.48550/arXiv.2609.14673)

DichromaticMap provides a Python library and an interactive viewer for
layer-resolved SC/FCC/BCC tilt grain-boundary dichromatic patterns. Use it to
inspect two overlaid grains, identify coincidence sites, measure crystal
vectors, count atoms in selected cells and explore common cells under uniform
strain.

## Example analysis plot

<a href="docs/images/dichromatic_pattern_example.png"><img src="docs/images/dichromatic_pattern_example.png" alt="FCC ⟨110⟩ dichromatic pattern at 22 degrees with axial layers, coincidence sites, a selected cell, atom counts, and a cross-grain vector" width="900"></a>

FCC ⟨110⟩ at a 22° reference misorientation. The plot shows both grains'
axial layers, coincidence markers, a selected four-vertex cell with separate
G1/G2 atom counts, and a cross-grain vector after selected-cell bulk strain.
[Open the full-resolution image](docs/images/dichromatic_pattern_example.png).

## Features

- Simple cubic (SC), face-centered cubic (FCC) and body-centered cubic (BCC)
  lattices with ⟨100⟩, ⟨110⟩, ⟨111⟩, ⟨112⟩ and custom integer tilt axes.
- Independent grain/layer visibility, grain-boundary side filtering,
  display rotation, floating grain reference axes, custom grain colors and unique
  layer symbols with adjustable sizes.
- Same-layer exact coincidence-site lattice (CSL) detection and local near-pair
  matching.
- Vector measurements in both grain coordinate frames, including axial
  periodic images.
- Manual common cells with symmetry completion after the first edge,
  parallelogram completion after three vertices, separate G1/G2 atom counts
  and optional uniform strain fitting.
- Automatic homogeneous-strain common-cell search and PNG export, with an
  atoms-only option.
- Save and restore sessions, with count, vector and strain CSV tables in one file.
- NumPy-based numerical functions usable independently of the viewer.

The three choices represent cubic Bravais point lattices with one atom per
primitive cell. Structures with an additional multi-atom basis, such as diamond,
are not represented. FCC remains the default.

Coordinates and distances use the reference lattice constant a₀ as their unit.
Local matching preserves the atom positions. Strain operations impose a
geometric deformation; they do not perform atomic relaxation or minimize
elastic energy. A manually selected polygon alone does not establish crystal
periodicity.

## Install and launch

Portable Windows, macOS (Apple Silicon / Intel) and Linux applications are
available as assets on [GitHub Releases](https://github.com/Yazhuo-Liu/DichromaticMap/releases).
Extract the complete archive for your system and open `DichromaticMap.exe`,
`DichromaticMap.app` or `DichromaticMap`. These include Python and the viewer
dependencies. See the [v0.2.2 release notes](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/releases/v0.2.2.md)
for platform and signing details.

**macOS first launch:** The portable `.app` builds are not signed or notarized,
so macOS may block them. If you downloaded the archive from this project's
GitHub Releases and trust it, try opening `DichromaticMap.app` once. Then open
**System Settings → Privacy & Security**, scroll to **Security**, click
**Open Anyway**, and confirm **Open**. macOS will remember the exception for
this app. If the warning says the app is damaged or will harm your Mac, do not
override it; report the issue instead. See [Apple's instructions](https://support.apple.com/en-us/102445).

For a Python installation, Python 3.10 or later is required. Install from PyPI:

```bash
python -m pip install "dichromatic-map[gui]"
python -m dichromatic_map
```

The viewer uses PySide6 and PyQtGraph. For numerical calculations only, install
with `python -m pip install dichromatic-map`; the numerical library requires
only NumPy.

```bash
python -m dichromatic_map --lattice SC --axis 100
python -m dichromatic_map --lattice BCC --axis 100
python -m dichromatic_map --axis "1 -1 3" --workers 4
python -m dichromatic_map --angle 22 --save pattern.png
python -m dichromatic_map --help
```

`dichromatic-map` is an equivalent launch command.

To install a downloaded source archive or a Git checkout instead, open its root
directory and run `python -m pip install ".[gui]"`. You can then use the same
launch commands above; `python main.py` also works from the source directory.

## GUI quick start

![DichromaticMap viewer with its plot and controls](https://raw.githubusercontent.com/Yazhuo-Liu/DichromaticMap/main/docs/images/gui-overview.png)

The example above shows FCC ⟨110⟩ at the Σ9 preset. The plot is on the left;
settings and results are in the scrollable **Controls** panel on the right.
Click section headers to expand them, and switch between the **ORIENTATION**
and **LAYERS** tabs at the top. By default, blue markers represent G1 and orange
outlines represent G2. Marker shapes distinguish axial layers, and gold outlines mark
same-layer exact coincidences.

1. **Choose the grains.** In **ORIENTATION**, select **Structure** (`FCC`, `BCC` or `SC`) and
   **View axis**, then choose a **CSL preset** or enter a
   **Misorientation**. For a custom axis, choose **Custom [h k l]**, enter an
   integer triple such as `1 -1 3`, and click **Apply axis** or press Enter.
   The angle range follows the axis: 0–45° for ⟨100⟩, 0–90° for ⟨110⟩,
   0–60° for ⟨111⟩, and 0–180° for other cubic axes such as ⟨112⟩.
   Custom indices use the same symmetry calculation; the angle field, slider
   and CSL presets update together. **Display rotation** turns the drawing
   without changing the grain geometry.
2. **Choose visible layers.** Open **LAYERS** and toggle individual **G1 A**,
   **G2 A**, etc. For a first selection, click **No layers**, then enable
   **G1 A** and **G2 A**. Coincidence and local-pair markers require the layer to
   be visible in both grains. Enable **Automatic common cell** to display an
   available exact or strain-search cell; **Fit cell** frames it in the view.
3. **Navigate.** Drag to pan and use the wheel to zoom. Under **VIEW /
   PERFORMANCE → VIEW**, **Field size** selects a wider or narrower region;
   **Center view** (`C`) returns to the origin at the current zoom.
   **Grain reference axes**, enabled by default, shows orientation arrows in the
   selected grain colors (blue G1 and orange G2 by default), sharing one fixed
   origin at the lower left. Color distinguishes the grains; no G1/G2 headings are drawn. Their panel keeps
   its size and position as the arrows rotate.
   They mark perpendicular reference directions and follow grain and display
   rotation; under strain they follow only the polar rigid rotation.
   The **PERFORMANCE** tab contains **CPU workers**.
4. **Choose colors, symbols and sizes.** Open **VIEW / PERFORMANCE → APPEARANCE**.
   Click the G1 or G2 color button to choose that grain's color, and use each
   layer's symbol menu to choose its shape for both grains. Symbols already used
   by other layers are disabled. Set a layer's **Size** from **25% to 400%** to
   scale its marker diameter in both grains; **100%** keeps the original size.
   **Reset appearance** restores the original colors, shapes and sizes;
   layers beyond the first twelve use distinct numbered
   circles. Changes preserve selections, counts and applied strain, and appear
   in PNG exports.
5. **Measure a vector.** In **GB / VECTOR**, click **Measure vector** (`V`),
   then click two distinct visible atom positions, P1 and P2. Read the vector
   annotation at the lower left of the plot, above the reference axes when
   they are enabled. Same-grain picks show that grain's
   coordinates; cross-grain picks show both G1 and G2 representations.
   **P2 axial periodic image** selects an axial repeat for the second endpoint
   without adding plotted atoms.
6. **Define a boundary.** Click **Pick GB** (`R`) and select B1, then B2.
   The side switches appear after the second pick; left/right are relative to
   B1 → B2. Use `1` or `2` for the two complementary grain-side arrangements,
   and `F` to show all sides again. Layer visibility settings still apply.
7. **Select and count a cell.** Expand **MANUAL COMMON CELL**, click
   **Pick 4 CSL vertices** (`M`), and select four gold sites of one layer in
   clockwise or counterclockwise order around a convex cell. Use **Undo
   vertex**, **Clear** or **Fit** as needed. The readout gives G1/G2 counts for
   the complete selected cell, including portions outside the current view.
   **Apply GB side visibility to counts** optionally restricts those counts
   to the displayed grain sides.
   After two vertices, **Complete by symmetry…** previews possible cells;
   after three, **Complete parallelogram…** generates the fourth vertex.
   Both work for exact CSL and local near pairs. Generated vertices are marked
   `(auto)` in the preview; accepting a candidate does not apply strain.
8. **Export the figure.** Scroll to **Export PNG…**, choose a file
   name and save. The PNG contains the plot with its current layers, rotation,
   reference axes when enabled, and annotations; the controls are excluded.
   For just the visible G1/G2 atom symbols, first check **Clean PNG (atoms only)**
   above the button. This keeps the current view and atom styles while omitting
   annotations, axes, grid and border. The live view stays unchanged.

9. **Save or restore a session.** Click **Save session…** beside **Export PNG…**
   to save one `.dmap` file containing the current structure, selections, applied
   strain/translations, display settings including colors, symbols and sizes, and numerical
   CSV tables. In **ORIENTATION**, use **Import session…** to restore it in the current window.
   The file is a standard ZIP archive; open it with a ZIP tool to read
   `counts.csv`, `vectors.csv` and `strain.csv`.

Press `Esc` to stop picking while retaining existing selections. Pressing `R`
or `V` starts a fresh boundary or vector selection. Click near a visible atom
or eligible cell marker; background clicks do not create arbitrary vertices.

For inexact orientations, expand **NEAR-CSL**, select a **Method**, and click
**Enable Near-CSL**:

| Method | How to use it | Effect |
| --- | --- | --- |
| **Local matching · no bulk strain** | Adjust **Local pair distance** (default 0.1 a₀); inspect purple midpoint markers and use them for manual cell picks | Finds nearby same-layer pairs while preserving the atom positions |
| **Homogeneous strain + periodic cell** | Set **Max principal strain** and **Search index bound**, then choose a returned candidate | Automatically applies the first result, or the candidate you select, to both grains |

A manually selected local cell containing at least one near pair can also be
fitted with **Apply bulk strain to selected cell** in **MANUAL COMMON CELL**.
Set its strain and rotation limits before applying; **Restore original local
structure** returns to the original geometry. This operation has a separate
apply step from the automatic search.

For control-by-control instructions, selection rules, result interpretation
and troubleshooting, see the [English GUI guide](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/en/README.md#gui-overview)
or [中文 GUI 使用指南](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/zh/README.md#gui-overview). Detailed workflows cover
[vector measurements](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/en/README.md#gui-vector),
[Near-CSL methods](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/en/README.md#gui-near-csl), and
[manual cells and counts](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/en/README.md#gui-manual-cell).

## Python usage

```python
from dichromatic_map import get_geometry, projected_columns, local_near_pairs

geometry = get_geometry("FCC", "110")
grain1 = projected_columns(12, 9, rotation_deg=11, lattice="FCC", axis="110")
grain2 = projected_columns(12, 9, rotation_deg=-11, lattice="FCC", axis="110")
pairs = local_near_pairs(grain1, grain2, distance=0.1)

print("Axial layers:", geometry.layer_count)
print("Near pairs:", len(pairs.layers))
print("Pair separations / a0:", pairs.distances)
```

The grains above have a reference misorientation of 22°. Matching uses the
supplied projected columns and preserves their layer labels.

| Function exported by `dichromatic_map` | Purpose |
| --- | --- |
| `get_geometry` | Obtain planar geometry, axial layers and repeat distance |
| `misorientation_range` | Obtain the fixed-axis angle limit and rotational symmetry period |
| `projected_columns` | Generate a grain's projected columns in a rectangular region |
| `same_layer_coincidence_sites` | Locate same-layer coincidences within a specified tolerance |
| `local_near_pairs` | Find same-layer mutual nearest pairs within a distance threshold |
| `exact_csl_cell` | Obtain a layer-preserving common translation cell for a recognized commensurate angle |
| `count_cell_atoms` | Count atoms in complete shared or per-grain polygons |

## Documentation

| Topic | English | 中文 |
| --- | --- | --- |
| Installation, viewer workflows and Python API | [User guide](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/en/README.md) | [使用手册](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/zh/README.md) |
| Algorithms, numerical conventions and implementation | [Implementation details](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/en/development.md) | [开发细节](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/zh/development.md) |

## Project website

The [project website](https://yazhuo-liu.github.io/DichromaticMap/) describes
the research context, analysis methods, scope and citation, with a separate
[online workspace](https://yazhuo-liu.github.io/DichromaticMap/use.html).
The online workspace follows the desktop viewer's plot-and-controls layout.
It runs the existing NumPy numerical core in a browser Web Worker through
Pyodide; the browser downloads Python and NumPy when the workspace opens, then
computes locally. No calculation server is used. It supports projected
grains, exact CSL, local near pairs, periodic-cell strain search, vector and
manual-cell measurements, selected-cell strain, PNG export, and `.dmap`
session import/export. The desktop app remains the reference interface.

To preview both pages locally, run:

```bash
conda run --no-capture-output -n lammps2026 python scripts/serve_site.py
```

Open `http://127.0.0.1:8000/` for the research software homepage. Check that
Overview, Analysis capabilities and Scope precede the usage sections, that the
application screenshot is unobscured, and that both citation copy buttons work.
Click **Use online** to open the workspace, or go directly to
`http://127.0.0.1:8000/use.html` and check the citation at the page bottom.
Opening the HTML file directly will not load its worker or package assets.
The first browser run needs internet access to fetch Pyodide and NumPy.
For a numerical adapter smoke check, run
`conda run -n lammps2026 python scripts/smoke_web.py` after building the site.
In the browser, wait for the starting overlay to disappear. At 22°, enable
**Near-CSL** with local matching and check that each near pair has a large
purple ring. Pick two atoms with **Pick GB** and check that the line reaches
both plot edges, including after pan or zoom. With grain reference axes shown,
measure a vector and check that its readout sits above the axes panel. After
applying strain to a selected cell, measure another vector and check the
`lattice [uvw]` readout for fractional directions. Save a `.dmap` file and
import it again to check the file-download path.
The Pages workflow publishes on pushes to `main` that change the site,
numerical source, screenshot or build script. Set the repository's Pages
build source to **GitHub Actions** to enable deployment.

## Contributing and support

Report problems through [GitHub Issues](https://github.com/Yazhuo-Liu/DichromaticMap/issues)
or [yliu3500@gatech.edu](mailto:yliu3500@gatech.edu). Contributions can be
submitted as a pull request or by email; see the [contribution guide](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/CONTRIBUTING.md).
This project is developed by volunteers and does not currently accept external
donations.

Maintainers can follow the [release build instructions](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/releases/README.md)
to build Python packages and native applications or prepare a draft release.

## License

DichromaticMap is distributed under the [MIT License](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/LICENSE).
