# DichromaticMap user guide

[中文](../zh/README.md) · [Development](development.md) · [Project home](../../README.md)

DichromaticMap is a standalone FCC/BCC tilt grain-boundary viewer using Qt and
PyQtGraph. It does not depend on GBClaw. Implementation lives in
`src/dichromatic_map/`; numerical calculations can be imported without Qt.
`main.py` is the only root-level source launcher. The three old root compatibility
modules have been removed. The Matplotlib implementation in `legacy/` is retained
for historical reference and regression comparisons, not active development.

## Launching

Use Python 3.10+, NumPy, PySide6 and PyQtGraph in a conda environment. On the
current development machine, the GUI dependencies are installed in `base`:

```bash
conda activate base
python main.py --workers 4
python main.py --lattice BCC --axis 100
python main.py --axis "1 -1 3"
python main.py --help
```

Run these commands from the project root. With dependencies installed, no project
installation or build is needed. To install missing GUI dependencies in the
active environment, use `python -m pip install numpy PySide6 pyqtgraph`.
Do not assume that another conda environment contains the same dependencies.

To import the numerical package from other projects, optionally run
`python -m pip install -e .` from the project root. Use
`python -m pip install -e ".[gui]"` to include GUI dependencies. After installation,
`python -m dichromatic_map` and `dichromatic-map` are also available, without
editing `sys.path`. Numerical use requires only NumPy; `python main.py --help`
does not load Qt. See [Development](development.md) for details.

All plot coordinates, view dimensions and local matching distances are expressed
in a₀. A coordinate of 1 means one reference lattice constant. FCC/BCC and preset
⟨100⟩, ⟨110⟩, ⟨111⟩, ⟨112⟩ axes are supported, as are custom integer axes.
Axial layers and repeat distances are computed from the selected geometry.

### Command-line options

| Option | Meaning and default |
| --- | --- |
| `--lattice` | `FCC` (default) or `BCC` |
| `--axis` | Integer axis, default `110`; quote separated signed/multi-digit indices, e.g. `"1 -1 3"` |
| `--angle` | Reference misorientation, 0–90°; default Σ9 for `110`, Σ5 for `100`, lowest-Σ preset for other axes, or 0° if none exists |
| `--lattice-constant` | Reference a₀, default 3.52 Å; does not rescale normalized plot coordinates |
| `--width`, `--height` | Base view dimensions, default 12 and 9 in a₀ |
| `--marker-size` | Atom marker size parameter, default 32 |
| `--view-scale` | Initial field-size multiplier, 0.1–5, default 1 |
| `--workers` | Calculation processes, default up to 4, limited by available CPUs; 1 disables the process pool |
| `--save` | Export the initial plot to PNG and exit |

```bash
python main.py --angle 22 --save pattern.png
# Headless export on a machine without a display server:
QT_QPA_PLATFORM=offscreen python main.py --workers 1 --save pattern.png
```

The GUI's `Export plot as PNG…` button exports the current state, including
display rotation and annotations.

### Navigation and shortcuts

| Action | Effect |
| --- | --- |
| Drag / mouse wheel | Pan / zoom without clearing an unfinished selection |
| `R` | Pick B1 and B2 to define a GB reference line; left/right are relative to B1→B2 |
| `V` | Pick P1 and P2 to measure a vector |
| `M` | Pick four same-layer common sites for a manual cell |
| `C` | Center the view |
| `F` | Show both GB sides of both grains; Layers filtering still applies |
| `1` / `2` | G1 left/G2 right, or G1 right/G2 left |
| `Esc` | Return to idle mode and pause picking |

You can pan or zoom after the first point and then select the next point.
Hidden atoms cannot be selected.

## Vector measurements and grain coordinates

Choose `Measure vector` (`V`) and select P1 and P2. A single arrow represents
P1→P2. Picking within one grain reports that grain's coordinates; a cross-grain
selection reports both G1 and G2 representations. These are two coordinate
descriptions of the same physical displacement, not two physical arrows.

- `G1/G2 current (polar)` gives the actual displacement in each grain's
  orthonormal cubic reference frame, in units of the original a₀. The frame
  follows the grain's rigid rotation obtained by polar decomposition. This
  readout includes strain-induced changes in direction and length.
- With strain, `G1/G2 lattice [uvw]` also reports coefficients in each grain's
  deformed conventional lattice basis. For the same pair of atoms in one grain,
  these material coefficients can remain unchanged. They are not the same as
  current physical components. `[uvw]` denotes a direct-lattice direction, not
  reciprocal-plane indices `(hkl)`.
- Simple fractional/integer components retain forms such as `a₀/2[1 1 2]`,
  without discarding magnitude. General directions use real-valued `≈ a₀[...]`
  components rather than forcing strain or irrational directions into integers.
- The displacement is constructed from the endpoints' current atom positions,
  including each grain's deformation, rotation and relative translation.
- The 3D readout includes the selected layers' axial height difference and the
  `P2 axial periodic image` setting. That setting chooses which axial periodic
  image of the projected P2 column is represented; it does not add sample
  thickness or replicate the lattice. Its default is 0.
- The arrow is the 2D projection of the 3D vector. `Current |Δr|/a₀` is its 3D
  length; cross-grain measurements also show `View Δxy/a₀`. Navigation and
  display rotation do not change the grain-frame readouts.

Convention: let `Q_g` rotate undeformed cubic coordinates into analysis
coordinates, and let `F_g = R_g U_g` be the current homogeneous deformation.
For the actual displacement `d`,

```text
current = (R_g Q_g)ᵀ d / a₀
lattice = (F_g Q_g)⁻¹ d / a₀
```

The in-plane deformation is extended to 3D with the axial component unchanged.
No deformation is applied to `d` a second time.

## Control panels and layer visibility

`ORIENTATION` and `LAYERS` share horizontal tabs. Orientation is selected
initially; the panel height follows the active tab's content. Layers lists each
grain's axial phases with the corresponding marker shape and color, such as
`G1 A` with a filled blue circle and `G1 B` with a filled blue diamond.
Each grain/layer checkbox acts independently. A hidden grain/layer cannot be
picked; its corresponding exact CSL and Local Near-CSL overlay is shown only
when that layer is visible in both grains.

Once a manual cell has been selected, its atom counts remain restricted to its
vertex layer and do not change with subsequent layer display toggles.
`Show automatic cell` is off initially.

`GB / VECTOR` follows the top tabs. GB-side controls appear after both B1 and B2
are selected. The P2 axial image control appears during or after a vector
measurement. `VIEW` / `PERFORMANCE` share a collapsed section. `NEAR-CSL` and
then `MANUAL COMMON CELL` follow, both initially collapsed.

## Manual common cells

`Show automatic cell` independently controls the automatic periodic-cell overlay.
For a manual cell, use `MANUAL COMMON CELL`:

1. Click `Pick 4 CSL vertices` (`M`).
2. Select four vertices in clockwise or counterclockwise perimeter order, all
   in the same axial layer. Gold exact CSL markers and purple local near-pair
   midpoints are eligible; ordinary atoms are not. The first point fixes the
   layer. A different-layer click is rejected with a message rather than
   redirected to a nearby point. Hidden sites cannot be selected. Each grain's
   four actual vertices must form a non-self-intersecting, non-degenerate convex
   quadrilateral.
3. The fourth point closes the cell. Blue/red outlines connect the actual G1/G2
   atom vertices, respectively. A lower-right annotation reports counts for the
   chosen layer; the panel contains detailed statistics. The annotation is not
   placed over the cell center. `Undo vertex`, `Clear` and `Fit` edit or frame
   the selection; Esc pauses picking.

### Counting convention

- Only the vertex layer is counted. Diamond vertices count the diamond layer,
  not circular layers or just diamond-shaped CSL markers. G1 and G2 are counted
  separately; overlapping atoms are not merged.
- Each Local Near-CSL vertex stores both original atom positions. G1 uses its
  four blue atoms as its polygon; G2 uses its four red atoms. Purple midpoints
  are selection aids, not a shared counting boundary. Exact CSL selections also
  use the actual per-grain vertices. Areas and parallelogram tests are separate
  for each polygon; blue dashed and red dotted outlines distinguish the grains.
- `Interior` means strictly inside. `Boundary` includes edges and corners.
  `Closed` is their sum.
- For a parallelogram, the half-open convention additionally counts
  `C1 + u(C2−C1) + v(C4−C1)` with `0 ≤ u,v < 1`. The two upper edges are
  excluded, avoiding duplicate boundary counts when the polygon is tiled.
  Non-parallelograms report interior, boundary and closed counts only.
- The plot uses plain labels such as
  `40 atoms · 34 inside + 5 edge + 1 corner`. The five edge atoms and one corner
  are those retained by the half-open convention, not all boundary atoms in the
  closed polygon. `◇ layer` identifies the diamond layer.
- `Apply GB side visibility to counts` additionally filters by the selected GB
  sides. With or without this option, counting uses only the vertex layer,
  ignores layer display toggles and is not clipped to the visible viewport.
- Counting regenerates atoms for the complete selection in the background,
  using the shared process pool or a background thread with one worker. It
  does not depend on the plot cache. Pan, zoom and display rotation do not alter
  counts or clear partially selected vertices.
- A manual selection or a geometric parallelogram alone does not prove
  periodicity. In particular, a cell selected from Local Near-CSL midpoints can
  be only approximately repeating; it does not certify a strict CSL primitive
  cell or a Σ value.
- Changing lattice, tilt axis, reference misorientation or applicable physical
  strain invalidates the old cell. If it contains local near-pair vertices,
  changing the local distance threshold or leaving the local method also clears
  it. The selected-cell apply/restore workflow below preserves its own paired
  selection explicitly.

## Making a selected local cell exactly coincident

After choosing four same-layer vertices in Local matching,
`Apply bulk strain to selected cell` becomes available if at least one vertex
is a local near-pair. Exact CSL vertices of that layer may be mixed in.
Applying the transformation requires a separate click; selecting a cell does
not move atoms. This workflow is independent of automatic
`Homogeneous strain + periodic cell` search.

- The solver uses each grain's four actual atom vertices, referenced to their
  centroids, and minimizes `||F1 − I||²_F + ||F2 − I||²_F`. Each entire grain
  receives one uniform transformation; atoms are not snapped individually.
  Uniform translations place both centroids at their original midpoint, so
  non-A-layer cells away from the origin can align correctly.
- Small rigid rotations are allowed. Polar decomposition `F = R U` separates
  each grain's rotation, principal strains (eigenvalues of `U` minus 1) and
  Green–Lagrange strain tensor in the reference grain frame.
  The default strain limit is 2%; the rotation limit is 1° per grain.
  Setting the rotation limit to 0° selects the symmetric pure-strain solve.
- The solver finds a least-change solution and then checks the limits. It is
  not a global search over all transformations satisfying those limits.
  An incompatible fourth corner, a degenerate transformation or an exceeded
  limit produces an explanation without changing the original state.
- After success, all lattice atoms are regenerated and exact same-layer CSL
  sites are recomputed using the original tolerance. It does not simply mark
  the four corners as coincident. Large regeneration and coincidence tasks use
  the shared process pool; manual counts still use only the selected layer.
  The common translation cell is anchored at the aligned C1, not necessarily
  `(0,0)`, and is not assumed to be primitive.
- The title and panel distinguish strain from rotation. Orientation retains the
  reference misorientation. The panel additionally gives
  `polar-frame θ = reference θ + rotation(G1) − rotation(G2)`.
  Display rotation is a separate drawing-only transformation.
- After a successful application, a separate read-only details box appears in
  the manual-cell panel, with scrolling and text copying. It lists each grain's
  principal stretches, signed principal strains, polar rotation, `F/R/U`,
  Green–Lagrange `E` in analysis and reference cubic frames, uniform translation
  and area change. Limits, reference/polar-frame misorientation, four-pair
  residuals and common-cell vectors are also shown. Percent strains and
  dimensionless tensors are labeled separately; display rotation is excluded.
  The box is hidden and cleared before application, after a rejected fit,
  on restore or when leaving this feature, without reserving blank space.
  Navigation, display rotation and worker changes preserve its text and reading
  position.
- `Restore original local structure` restores the original atoms and four
  local vertices. Restore before editing the vertices or local threshold.
  Pan, zoom, display rotation and CPU-count changes do not undo the strain.
  Apply/restore clears old GB and vector selections; changing the structure,
  axis, reference angle or leaving the method clears the applied state.

This is an imposed geometric transformation, not a stress-free configuration,
an atomic relaxation or an elastic-energy minimum. For example, the FCC ⟨110⟩
22° diamond-layer vertices near `(0,±5.6013)` and `(±2.5248,0)` require a maximum
principal strain of approximately 0.444310%, with G1/G2 rotations of
approximately +0.168543° / −0.168543°. The half-open count is 40 atoms per grain
in that layer before and after the transformation.

## Display rotation

`Display rotation` in Orientation ranges from −180° to +180°; `0°` resets it.
It rotates both grains, CSL sites, automatic/manual cells, the GB line and the
measurement arrow. The grid and screen axes remain fixed, while the current
view center follows the rotated configuration.

This does not alter misorientation, physical strain, original coordinates or
reference Miller indices. Picking and GB-side tests still use model coordinates.
Cross-grain view components change with display orientation. Common-cell vectors
in the strain panel are explicitly given in unrotated analysis coordinates.
PNG export includes the current display rotation and manual-cell annotation.

## Two Near-CSL methods

Near-CSL is off initially. Select a method, then click `Enable Near-CSL`.

- `Local matching · no bulk strain` is the default. It does not move atoms.
  Same-layer, mutually nearest atoms are paired within a distance threshold,
  initially 0.05 a₀. Purple midpoints are candidate alignment positions;
  their symbols follow the underlying layer. Purple links join the original
  atoms, while gold is reserved for exact CSL. This method neither computes a
  periodic cell nor implies stress-free relaxation.
- `Homogeneous strain + periodic cell` searches for common translation cells
  under symmetric positive-definite in-plane deformations, without adding rigid
  rotation. Defaults are a 2% principal-strain limit and integer search extent
  ±12. The dropdown provides cell-size/strain trade-offs, not an elastic-energy
  minimum or a global-optimality guarantee.

Matching retains axial phase labels; local matching is not a cross-layer 3D
nearest-neighbor search. Multiple workers share a process pool, and local
queries do not form a full all-atom pairwise distance matrix.

Interactive limits are reduced axis indices with absolute value at most 64,
at most 256 axial layers and 250,000 candidate columns per grain. Oversized
manual selections report an error rather than a partial count. Reduce the
selection or view size, or use a lower-index axis.

## Tests and troubleshooting

Run from the project root in an environment with GUI dependencies and Matplotlib:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s test -v
```

Tests use offscreen Qt. Coverage includes independent 3D lattice enumeration,
exact and strained CSL, local matching, cancellation/stale results, hidden-point
picking, whole-selection counting, display rotation, selected-cell strain and
restore, and current/material vector components. Package tests cover GUI-free
imports, the source launcher and export, isolated state and spawned workers.
Matplotlib is required only for historical comparison tests, not the Qt viewer.

If PySide6 or PyQtGraph cannot be imported, check that the active conda
environment is the one where dependencies were installed. If importing
`dichromatic_map` fails, use the root launcher or perform an editable installation.
Use the offscreen export command without a display server, or `--workers 1` when
process creation is restricted. Do not treat a partial view as a complete count.

`build/` and `__pycache__/` are generated artifacts, not source requirements.
See [Development](development.md) for their roles and cleanup cautions. This
manual is Markdown and does not need an HTML build to be read.
