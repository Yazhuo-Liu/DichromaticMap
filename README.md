# DichromaticMap

[English user guide](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/en/README.md) · [中文使用手册](https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/docs/zh/README.md)

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
