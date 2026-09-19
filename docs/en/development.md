# Algorithms and implementation

[中文](../zh/development.md) · [User guide](README.md) · [Project home](../../README.md)

This document describes the numerical algorithms, data conventions and execution model of
DichromaticMap. Mathematical vectors below are column vectors; NumPy point arrays store one
point per row, so corresponding matrix operations use a transpose.

## 1. Modules, data and coordinates

| Module | Responsibility |
| --- | --- |
| [crystal.py](../../src/dichromatic_map/crystal.py) | Integer axes, primitive layer geometry, projected columns and vector coordinates |
| [matching.py](../../src/dichromatic_map/matching.py) | Exact displayed coincidences, rational common cells and local pairs |
| [cells.py](../../src/dichromatic_map/cells.py) | Cell reduction, polygon membership and full-region atom counts |
| [strain.py](../../src/dichromatic_map/strain.py) | Automatic symmetric strain search and selected-cell fitting |
| [compute.py](../../src/dichromatic_map/compute.py) | Worker entry points, executors and asynchronous search |
| [state.py](../../src/dichromatic_map/state.py) | Parameters, geometry, selections, deformations and result state |
| [appearance.py](../../src/dichromatic_map/appearance.py) | Qt-free grain colors, unique marker identifiers, validation and resizing |
| [session.py](../../src/dichromatic_map/session.py) | Versioned session archive, validation and atomic persistence |
| [exports.py](../../src/dichromatic_map/exports.py) | Recomputed count, vector and strain CSV tables |
| [ui/session.py](../../src/dichromatic_map/ui/session.py) | File dialogs, control synchronization and session restoration |
| [ui/window.py](../../src/dichromatic_map/ui/window.py) | Interaction, validation, task dispatch and result application |
| [ui/plot.py](../../src/dichromatic_map/ui/plot.py) | Display transforms, buffered rendering, overlays and export |
| [ui/controls.py](../../src/dichromatic_map/ui/controls.py) | Controls and visual resources |
| [ui/markers.py](../../src/dichromatic_map/ui/markers.py) | Shared marker paths, layer icons and grain edge colors |

The numerical modules use NumPy without importing Qt. `PatternState`, `ComputeSession`,
`ControlDock` and `PatternPlot` hold separate parts of a viewer session; the window coordinates them.

- `CrystalGeometry` stores the reference frame, primitive planar basis, layer offsets and axial repeat.
- `ProjectedGrain` contains `positions: (N,2)`, `layers: (N,)`, `half_indices: (N,3)` and a layer count.
- `LocalPairs` retains both actual endpoint arrays and the layer of each pair. Midpoints are derived data.
- `StrainedCell` contains two integer cell matrices, two deformation gradients and common cell vectors.
- `SelectedCellStrain` also stores uniform translations, transformed vertices, the cell origin,
  alignment residual, polar rotation angles and principal stretches.
- `CellAtomCounts` stores counts by grain and layer, separate grain areas and per-grain half-open availability.

Positions, widths, translations and distances are normalized by the reference lattice constant a₀.
Areas use a₀². Integer `half_indices` instead represent coordinates in units of a₀/2.
Changing the displayed physical lattice constant does not rescale the normalized geometry.
Deformation gradients and strains are dimensionless; API angle arguments are in degrees.

`geometry.frame` has columns `(ex, ey, ez)`: the analysis axes expressed in reference cubic
coordinates, with `ez` along the tilt axis. Thus a cubic half-index row `h` projects as
`(h / 2) @ frame`. A layer identifies an axial phase within one repeat, rather than a second
atom species or an independently repeated slab.

## 2. Integer lattice and layer construction

Implementation: `get_geometry`, `plane_integer_basis` and `extended_gcd` in
[crystal.py](../../src/dichromatic_map/crystal.py).

The axis parser reduces an integer triple `d = (h,k,l)` by its greatest common divisor.
It rejects the zero vector and reduced components with absolute value above 64.
Let `D = d·d`. A transverse unit vector is constructed by crossing a reference direction
with `d / sqrt(D)`; the second transverse vector completes a right-handed orthonormal frame.

SC, FCC and BCC each describe a Bravais point lattice with one atom per primitive
cell. There is no additional atom-basis or chemical-species array. The columns of
the primitive translation matrices below are expressed in half-index units (a₀/2):

```text
          [0 1 1]             [2 0 1]             [2 0 0]
T_FCC =   [1 0 1]    T_BCC =  [0 2 1]    T_SC =   [0 2 0]
          [1 1 0]             [0 0 1]             [0 0 2]
```

For the integer row `r = dᵀ T`, extended Euclidean steps construct a unimodular integer matrix
`U` with `r U = (g,0,0)`, where `g = gcd(r)`. The last two columns of `T U` form an integer
basis `H` for translations perpendicular to the axis. Gauss reduction shortens this basis
using integer column operations; its sign is chosen so the projected basis has positive determinant.
With `C = geometry.frame`, the normalized planar basis is `P = C[:, :2]ᵀ H / 2`.

The shortest axial lattice translation has half-indices `κ d`, where:

- FCC uses `κ=1` when `h+k+l` is even, otherwise `κ=2`.
- BCC uses `κ=1` when all three components have the same parity, otherwise `κ=2`.
- SC uses `κ=2`: all reference half-indices are even, so the physical repeat is `a₀ d`.

The number of phases, axial repeat length and spacing are:

```text
L = κ D / g,       H_axial = κ sqrt(D) / 2,       s = g / (2 sqrt(D)).
```

For SC, the reduced axis has `gcd(d)=1`, hence `g=gcd(2d)=2`. Therefore
`L=D`, `H_axial=sqrt(D)` and `s=1/sqrt(D)` in normalized units. The [100], [110],
[111] and [112] layer counts are 1, 2, 3 and 6. SC uses the same integer-plane
construction, phase offsets and resource limits as FCC/BCC.

The first column of `T U` generates successive layer offsets. Integer planar translations
bring each offset back into the planar fundamental parallelogram without changing its axial height.
The implementation rejects geometries requiring more than 256 phases. Geometry arrays are
read-only and cached by normalized lattice and axis, with up to 64 cached geometries.

### Fixed-axis misorientation range

The reference angle input uses the symmetry of the undeformed cubic lattice.
SC, FCC and BCC have the same proper cubic point group: the 24 signed permutation
matrices with determinant +1. For the reduced integer axis `d`, count the matrices
`S` satisfying `S d = d` exactly. This directed-axis stabilizer has order `n`;
operations sending `d` to `-d` are not counted as rotations about `d`.

```text
rotation period = 360° / n
input interval  = [0°, 180° / n]
```

The second line also identifies `theta` and `-theta` by exchanging the grains.
The ⟨100⟩, ⟨110⟩ and ⟨111⟩ families have orders 4, 2 and 3, respectively, so the
input maxima are 45°, 90° and 60°. All other cubic axes, including ⟨112⟩, have
order 1 and use 180°. The cubic rotation orders are illustrated in the
[IUCr teaching pamphlet, *Projections of cubic crystals*](https://www.iucr.org/what-we-do/education/pamphlets/projections-of-cubic-crystals).
Sign changes, integer rescaling and index permutations are handled through the
axis parser and symmetry calculation, without relying on menu labels.

This is a reduction within the chosen fixed-axis rotation family, not a global
cubic disorientation minimization that can change the axis representation.
Every valid supported cubic axis has a computable order. The general 0–180°
interval is also the fallback when crystal symmetry information is unavailable.
The public `misorientation_range(axis="110", lattice="FCC")` helper returns an
immutable `AngleRange(maximum_deg, period_deg, symmetry_order)`; for an unknown
lattice it returns `(180.0, None, None)` without claiming a known rotation period.
Axis validation still applies, and geometry construction still rejects unsupported
lattices.

The GUI number field, slider, range hint and preset list use the same upper
bound. Command-line angle validation rejects values outside that bound rather
than folding them into the interval. The numerical geometry and rotation
functions continue to use their supplied angles without this UI restriction.

CSL menu presets use integer quaternions `(m, n d)` and
`theta = 2 atan2(sqrt(D) n, m)`. For primitive quaternions, preset Σ is the odd part of
`m² + D n²`. The [100]/[110] menus have selected entries; other axes use a bounded low-Σ menu.
All menus are limited to the corresponding fixed-axis input interval; for example,
the [100] Σ5 entry is approximately 36.87°, while its complementary 53.13° angle
is outside the 0–45° interval. The menu and its display-angle matching tolerance
do not establish exact cell commensurability.

## 3. Generating a finite projected region

Implementation: `projected_columns` in [crystal.py](../../src/dichromatic_map/crystal.py).

For grain rotation `phi`, planar deformation `F`, uniform translation `t` and layer half-index
offset `o_l`, define:

```text
B = F R(phi) P
b_l = F R(phi) (C[:, :2]ᵀ o_l / 2) + t
x(n,l) = B n + b_l,                   n in Z².
```

The two grains use `phi = +theta/2` and `-theta/2`. Display rotation is applied later and
is absent from this formula.

For each layer, the algorithm maps the four requested rectangle corners through `B⁻¹`,
subtracting `b_l` first. Floors/ceilings of these integer-coordinate bounds, padded by one
index, define the candidate mesh. The final `(ny,nx,2)` coordinate array is filled directly
by broadcasting one-dimensional index ranges. It preserves row-major enumeration while
avoiding two intermediate dense mesh arrays.

Candidates are projected, cropped against the requested rectangle and assigned reference
half-indices `H n + o_l`. Rotation, strain and translation affect positions, while these
reference indices remain unchanged. The edge tolerance is
`1e-10 * max(1, max(abs(rectangle_corners)))` in normalized coordinates.

The allocation guard counts candidate mesh entries across all layers, before cropping;
it allows at most 250,000 candidates per grain. Enumeration scales with the inverse-projected
planar area and layer count, rather than a surrounding three-dimensional crystal volume.

## 4. Exact displayed sites and exact common cells

Both algorithms are in [matching.py](../../src/dichromatic_map/matching.py), but answer
different questions.

### Displayed coincidence sites

`same_layer_coincidence_sites` separates the input arrays by layer. With spatial tolerance
`tau` (the viewer uses `1e-6`), it assigns positions to `floor(x/tau)` bins. Each pair of bin
coordinates is packed into one 64-bit key using the low 32 bits of each coordinate.
The second grain's keys are sorted. Unresolved first-grain sites query their own bin and
eight neighbors, then accept a candidate only if its squared Euclidean distance is at most `tau²`.
The displayed marker is the mean of the two matched positions.

This fast detector assumes tiny bins relative to same-layer lattice spacing. It checks the
first occupant found for each queried key; it is not an exhaustive multi-occupant radius search.
Packed keys can alias at very large bin separations. The distance check rejects false matches,
but another occupant may then be missed. The algorithm should not be treated as a general
large-radius matcher or a guarantee for arbitrary coordinate ranges.

These markers describe finite-tolerance coincidences in the supplied region. Their presence
alone does not certify a primitive periodic cell.

### Rational, layer-preserving common cell

`exact_csl_cell` approximates `tan(theta/2)/sqrt(D)` by a reduced rational `n/m`.
The `max_denominator` parameter bounds both primitive coefficients `m` and `n`
(default 128), preventing an arbitrarily large numerator near 180°. It accepts
the rational only if its reconstructed angle differs by at most `1e-9` degrees.
An unrecognized angle returns `None`; a mathematically commensurate rotation
can remain unrecognized when its coefficients exceed the bound. The full 0–180°
interval is supported: 0° and 180° use exact endpoint quaternions `(1, 0)` and
`(0, d)`, respectively, without evaluating the singular tangent at 180°.

The quaternion rotation is constructed as an integer numerator and denominator. Projecting
it into the primitive planar basis gives the rational matrix `B2⁻¹ B1 = A/q`, where
`B1 = R(theta/2) P` and `B2 = R(-theta/2) P`. The required integer translations satisfy:

```text
A z = 0 (mod q),     z in Z².
```

`congruence_kernel` intersects these congruences row by row using extended gcd and integer
basis changes. It does not require every coefficient to have a modular inverse. Large
intermediate products use Python integer arithmetic through object arrays.

For the kernel basis `M1`, compute `M2 = A M1 / q`. `reduce_cell` applies identical integer
column operations to `M1`, `M2` and the common vectors. Finally, `B1 M1` must equal `B2 M2`
with absolute tolerance `1e-8` and zero relative tolerance.

The result is primitive in the plane preserving the A phase. It is not necessarily a primitive
three-dimensional CSL cell; its atom counts are not a general definition of Σ. Results are
cached with a bounded 256-entry cache.

## 5. Local mutual-nearest matching

Implementation: `_nearest_in_radius` and `local_near_pairs` in
[matching.py](../../src/dichromatic_map/matching.py).

The cutoff `r` must satisfy `0 < r <= 0.5`; its default is 0.05 a₀. Each layer is processed
independently. Bin coordinates are measured relative to the common minimum of both point sets.
Within an overflow-checked padded rectangle, the bin `(ix,iy)` is encoded as one scalar integer:

```text
stride = max_iy + 3
key = ix * stride + iy + stride + 1.
```

Padding prevents neighboring rows from sharing a key. If the complete key range would exceed
signed 64-bit storage, lookup falls back to structured pairs of full 64-bit coordinates.
This fallback is specific to local matching, not the exact-site hash above.

Sorting records each unique bin's start and occupant count. Queries run in batches of 8,192
points and use one binary search per neighboring bin. Every occupant in all nine bins is
considered. The nearest candidate must lie within the cutoff; equal squared distances are
resolved by the candidate's lexicographic `(x,y)` order.

The query runs in both directions. A pair survives only when both endpoints select each other.
Exact coincidences participate in this assignment, then pairs with separation at or below
`exact_tolerance` (default `1e-6`) are removed from the local overlay. Thus an exact neighbor
cannot be bypassed to create another near pair. Inputs and atom positions are not modified.

The algorithm avoids allocating a complete pairwise distance matrix. Its cost still depends
on bin occupancy; dense bins require more candidate comparisons. Local pairs neither move
atoms nor establish periodic translations.

## 6. Automatic homogeneous-strain cell search

Implementation: `candidate_vectors`, `solve_cells_chunk`, `pareto_cells` in
[strain.py](../../src/dichromatic_map/strain.py), and `reduce_cell` in
[cells.py](../../src/dichromatic_map/cells.py).

### Candidate translations

The search index extent is between 2 and 40 and the strain limit `p` is in `(0,10]%`.
Let `e=p/100`. Integer grain-1 vectors use one half-plane (`nx>0`, or `nx=0, ny>0`) to remove
sign duplicates. Each `v=B1 n` is mapped to grain-2 coordinates and rounded to seed a bounded
offset search. A necessary compatibility condition is:

```text
||v-w|| <= e (||v|| + ||w||).
```

It implies `||v-w|| <= 2e ||v||/(1-e)`. Multiplying by the norms of the rows of `B2⁻¹`
gives conservative componentwise bounds in integer coordinates. These bounds, integer extent
limits and the per-vector search radius reject impossible offsets before distances are evaluated.
The bounds include slack for the final `1e-12` mismatch tolerance and floating-point transforms.

The retained set is the union of the first 160 shortest pairs and first 160 lowest-mismatch
pairs, using stable sorting. There are at most 320 candidates, often fewer because the lists
overlap. Therefore even the bounded search is sampled, rather than exhaustive over every cell.

### Symmetric least-change solve

Two candidate pairs form integer matrices `M1`, `M2`. Singular cells and incompatible determinant
signs are discarded using direct 2×2 integer determinants. Only pair-index rows belonging to
the current chunk are constructed. Both cell diagonals are also checked against the necessary
compatibility bound before solving; a `3e-8` margin accommodates the final residual tolerance.

With `A=B1 M1`, `B=B2 M2`, write `Fg=I+Sg` and constrain each `Sg` to be symmetric. Solve:

```text
minimize ||S1||_F² + ||S2||_F²
subject to S1 A - S2 B = B - A.
```

Each grain contributes unknowns `(Sxx, Syy, sqrt(2) Sxy)`. This makes the squared Euclidean norm of the
six unknowns equal to the Frobenius objective. The four edge-component equations form a 4×6
system, solved in batches by an SVD-based pseudoinverse with `rcond=1e-11`.

The result is accepted only when both `Fg` have positive eigenvalues, the largest absolute
principal strain `max|eig(Fg)-1|` is at most `e+1e-12`, and the largest component of
`F1 A - F2 B` is below `1e-8`. Accepted symmetric positive-definite gradients have no additional
polar rotation. Common vectors are Gauss-reduced with matching integer column operations.

This is a least-change solution followed by limit checks. Failure does not prove that every
other deformation satisfying the same edge constraints would violate the limits. No elastic
energy, force balance or relaxation is evaluated.

### Cell ranking

`StrainedCell.atoms` gives `L * abs(det(Mg))` for each grain: reference atom counts over one
full axial repeat. These differ from the GUI's selected-layer counts.
Within each chunk and in the final merge, cells are ordered by total reference atom count
and maximum principal strain. The final merge deduplicates by the grain counts and deformation
entries rounded to nine decimals, keeps successive strain improvements above `1e-10`, and
returns at most 12 tradeoff candidates. These are the retained set's tradeoffs, not a proof
of a globally optimal or three-dimensionally primitive CSL cell.

## 7. Fitting a selected four-pair cell

Implementation: `strain_selected_cell` and `strain_tensors` in
[strain.py](../../src/dichromatic_map/strain.py).

The input is two `(4,2)` polygons of actual grain endpoints, not their shared marker midpoints.
Each polygon must be convex and ordered around its perimeter. Transforming the vertices back
through each unstrained grain basis, then subtracting the selected layer's phase offset, must
recover integer indices within `1e-7` a₀. Cell edge determinants must have compatible signs.

For original grain centroids `cg`, let `ag,i = xg,i - cg`. The fit minimizes
`||F1-I||_F² + ||F2-I||_F²` under all four centered equalities `F1 a1,i = F2 a2,i`.
Normally all four entries of each gradient are free, giving eight equations in eight unknowns.
A rotation limit of exactly zero switches to symmetric gradients and six Frobenius-scaled
unknowns. Both variants use `numpy.linalg.lstsq` with `rcond=1e-11`.

The common target centroid is `c=(c1+c2)/2`, so each grain receives the uniform translation
`tg = c - Fg cg`. These translations are essential when the selected cell is away from the
origin or belongs to a shifted axial phase.

For general gradients, an SVD gives `F=R U`: singular values are principal stretches and the
orthogonal factor is the polar rotation. Principal strains are `stretch-1`, not eigenvalues
of a nonsymmetric `F`. The function rejects a nonpositive determinant, a minimum stretch at
or below `1e-10`, strain above `p/100+1e-12`, or rotation above the per-grain limit plus `1e-9` degrees.
The defaults are 2% strain and 1° rotation per grain; the maximum allowed rotation limit is 5°.

All four transformed vertex pairs must agree within `1e-8` a₀, and common cell edges must agree
componentwise within `1e-8`. The common origin is the transformed first vertex. The returned
cell need not be the smallest possible common cell. The UI applies the same gradients and
translations to whole grains, regenerates atoms and reruns exact detection.

Green–Lagrange strain is `E=(FᵀF-I)/2`. The readout also transforms this tensor into each
reference cubic grain frame. Polar rotation, engineering principal strain, Green–Lagrange
strain and area change `det(F)-1` are distinct reported quantities.

## 8. Full-region counting and three-dimensional vectors

### Polygon and half-open counts

Implementation: `validate_cell_vertices`, `cell_membership`, `count_cell_atoms` in
[cells.py](../../src/dichromatic_map/cells.py).

The counting input can be one common `(4,2)` polygon or separate `(2,4,2)` grain polygons.
Each grain is regenerated over its polygon's full bounding rectangle with a small margin,
using the current deformation and translation. A selected layer is filtered before polygon
and corner distance tests. `layer=-1` retains all layers; the GUI passes the selected layer.
The candidate allocation guard remains active, so an oversized region raises an error instead
of reporting an incomplete count.

Convexity requires consistently oriented consecutive edge turns. Signed distances to all
four edges classify interior points (`distance > eps`) and closed-region points
(`distance >= -eps`), where `eps=1e-8 * max(1, longest_edge)`. Boundary is closed minus interior.
Clockwise and counterclockwise vertex order are both supported.

For a parallelogram, solve `x=C1 + u(C2-C1) + v(C4-C1)`. The half-open cell is `0<=u,v<1`;
the implementation uses lower bounds `-eps_uv` and strict upper bounds `1-eps_uv`, with
`eps_uv` scaled by the inverse basis row norms. Corners and other retained boundary points
are counted separately. For a non-parallelogram, only interior and closed-boundary counts
are meaningful. Check `half_open_available` separately for each grain.

Optional GB filtering uses the sign of the directed B1→B2 cross product with tolerance `1e-9`.
It is independent of layer display switches and the viewport. `numpy.bincount` accumulates
per-layer totals. Areas use a translation-stable shoelace formula on vertices relative to C1;
unequal grain areas are kept separate. Polygon shape alone does not certify periodicity.

### Vector coordinate representations

Implementation: `crystal_vector_coordinates` and `format_direction_components` in
[crystal.py](../../src/dichromatic_map/crystal.py), with endpoint selection in
[ui/window.py](../../src/dichromatic_map/ui/window.py).

The displacement `d` is built from current atom endpoints and is already divided by a₀.
Its axial component includes the two layer heights and the selected periodic image of P2:
`dz = (l2-l1) s + k H_axial`. The image index chooses a representative of the same column;
it does not add sample thickness.

For a grain reference rotation `phi`, let `Q=R3(phi) Cᵀ` map cubic coordinates to the analysis
frame. Extend the planar gradient by leaving the axial direction unchanged to obtain `F3`.
Let `Rpolar` be its polar rotation. Then:

```text
O = Rpolar Q            current = Oᵀ d
B = F3 Q               lattice = solve(B, d).
```

`current` contains physical displacement components in transported orthonormal cubic axes;
`lattice` contains coefficients in the deformed conventional lattice basis. The displacement
must not be deformed or divided by a₀ again. Cross-grain endpoints include both grains'
transforms and translations before either coordinate representation is computed.

Direction formatting retains rational expressions only when all components match to `1e-10`,
the common denominator is at most 48 and reduced indices do not exceed 256. Otherwise it
prints real-valued approximate components instead of inventing Miller indices. Display
rotation affects the projected arrow but not these grain-frame quantities.

## 9. Rendering, picking and asynchronous execution

Implementation: [compute.py](../../src/dichromatic_map/compute.py),
[ui/plot.py](../../src/dichromatic_map/ui/plot.py) and
[ui/window.py](../../src/dichromatic_map/ui/window.py).

The plot retains atoms in a rectangle enlarged by a factor of 2.0 around the model-space
viewport. Near-pair mode ensures at least a `2r` neighborhood beyond visible points for both
directions of mutual matching; buffer dimensions are at least the visible dimension plus
`4.4r`. Pair midpoints within `2r` of a buffer edge are excluded from picking and overlays.
Moving outside the usable interior schedules a replacement buffer.

Display rotation applies a separate 2D transform. Mouse positions are mapped back to model
coordinates for picking. Independent grain/layer masks filter atoms; exact and local overlays
require the layer to be visible in both grains. Local visibility checks both original endpoints
against the GB sides, not just the midpoint. A manual cell stores its layer and actual paired
vertices, keeping its counting geometry independent of later display filters.

The floating grain reference axes use the two transverse unit directions in
`geometry.frame[:, :2]`, expressed as crystal direction labels `[uvw]`. They are
perpendicular to the viewing axis and to each other; for [110] they are parallel
to `[-1 1 0]` and `[0 0 1]`. `in_plane_reference_directions` derives their signed
integer labels by cross products followed by greatest-common-divisor reduction;
the labels describe directions, not primitive translation lengths.
`in_plane_reference_axes` uses the two unit basis vectors as their normalized
reference-plane components. For grain `g`, the displayed arrow directions are

```text
v_g,i = R_display R_polar(F_g) R(phi_g) e_i,
phi_1 = +theta/2,    phi_2 = -theta/2,    i = 1, 2.
```

Using only the polar rotation of `F_g` keeps both arrows orthogonal. Applying
`F_g` directly would draw the sheared/stretched lattice directions, which is a
different quantity from this orientation reference. The overlay is anchored to
the lower-left viewport with a fixed screen size, independently of the model
origin and atom positions. Both grain frames share one fixed origin, with the
selected grain colors (blue and orange by default) distinguishing them and no
G1/G2 headings. The layout reserves space
for the arrows and `[uvw]` labels independently of the current angle, so rotation
does not recenter or resize the panel. Pan and zoom preserve its placement and
size; geometry, reference angle, deformation and display rotation determine the
arrow directions. `Grain reference axes` controls visibility, and the vector readout
reserves space above it while visible. Enabled axes are included in plot export.

The plot reuses local-pair scatter data, links, transformed midpoints and distances when only
the viewport changes. The cache key includes pair-object identity, local-mode state, display
rotation, shared layer visibility, GB endpoints and sides, geometry validity, buffer bounds
and cutoff. Pair updates replace result objects. Mutating their arrays in place would bypass
the identity check and is not the update protocol used by the viewer.

Marker sizes update only when the calculated diameter changes; rebuilding layer items resets
that cache. Visible counts reuse already filtered, rotated scatter coordinates. Panning still
updates visible statistics and annotations; it does not eliminate pixel painting or buffer generation.

With more than one worker, a shared `ProcessPoolExecutor` uses the `spawn` context and
module-level numerical worker functions. Grains are separate jobs; exact and local matching
use bounded batches of layers. Manual counts and automatic strain search share the executor.
If available, `threadpoolctl` restricts each worker's internal BLAS pool to one thread.

With one worker, local matching, manual counting and automatic search use background
single-thread executors. Grain generation and exact-site detection run synchronously;
the small selected-cell fit is also synchronous. The entire application is not guaranteed
to perform every computation off the UI thread.

`NearSearch` prepares candidates, then schedules eight starting rows of the upper-triangular
vector-pair index set per job. At most the configured worker count is in flight. A generation
counter invalidates old requests, cancels jobs that have not started and discards stale results;
already running numerical work is allowed to finish. Completed chunks are merged in starting-row
order so process completion order cannot select a different tied candidate. Viewer geometry
signatures and count-request keys similarly prevent stale results from replacing current state.

These resource bounds, candidate sampling and numerical tolerances are part of the implemented
algorithm. They support interactive geometric analysis and do not establish relaxed structures,
elastic equilibria or global optimality.

## Appearance state and marker rendering

`PatternState.grain_colors` stores two `#RRGGBB` strings; `layer_symbols` stores
one unique marker identifier per axial layer, shared by both grains. The Qt-free
`appearance.py` validates the color format and normalizes it to lowercase. It
checks symbol count, membership and uniqueness, while allowing the two grain
colors to match. The original twelve PyQtGraph symbols remain the defaults for
the first twelve layers; later layers use `number:13` through `number:256`.
All numbered identifiers from `number:1` through `number:256` are valid regardless
of the current layer count, so an assigned numbered symbol survives a geometry
change to fewer layers. Resizing preserves valid retained assignments and fills
new layers from unused defaults.

`ui/markers.py` resolves ordinary identifiers to built-in symbols and numbered
identifiers to cached `QPainterPath` circles with numeral cutouts. It does not
modify PyQtGraph's global symbol registry. The numerals use built-in seven-segment
vector outlines rather than system fonts, so headless Qt environments cannot
replace distinct numbers with identical missing-glyph boxes. Plot markers and control icons share
this resolver. Grain colors also feed the reference axes, manual-cell outlines
and legend; G1 keeps a filled style with a darker edge and G2 keeps an outline.

The `APPEARANCE` controls disable symbols assigned to other layers and reject
duplicate assignments in callbacks. Appearance updates repaint existing data and
refresh affected icons and overlays. They do not invalidate physical geometry,
matching or count results, schedule numerical work, or clear selections, strain
or translations. `Reset appearance` changes only these style preferences.

## Session persistence and numerical export

`session.py` stores a `.dmap` ZIP archive with schema version 2. `session.json`
contains explicit physical and display inputs, not a dump of `PatternState`:
current lattice/axis/angle replace potentially stale launch parameters, arrays
become ordinary JSON lists, and selected atom indices retain their grain and
axial layer. Applied `StrainedCell` and `SelectedCellStrain` records include the
original vertices and cutoff needed to undo a selected-cell fit. Interaction
mode and partial selections are retained; rendering buffers, futures, worker
counts and local paths are excluded. The selected automatic strain candidate is
retained, while its other search candidates are omitted. Schema 2 adds
`grain_colors` and `layer_symbols` to the explicit state fields. Schema 1 files
still load with the default colors and unique layer symbols; saving them writes
schema 2. Unsupported future versions and unexpected fields are rejected.

`load_session` returns `SessionSnapshot(state, settings, view_range)`. The
loader checks schema/version, required archive members and bounded sizes,
finite numbers, array shapes, axis/layer/index validity, appearance formats and
symbol uniqueness, deformation orientation, common-cell consistency, and
selected-atom coordinates. It reads JSON directly
without extracting files or unpickling objects; CSV outputs are not inputs to
state restoration. `save_session` validates its own serialized payload and
builds the numerical tables before writing a temporary archive beside the
destination. A successful write is flushed and atomically replaces the target;
failure leaves the previous file intact and removes the temporary file.

`SessionController` checks viewport enumeration limits before replacing the
window state. It stops old timers, invalidates geometry/search work and drops
the previous count future's publication handle. Control signals are blocked
while applying the validated state, preventing geometry changes from clearing
imported selections. It then regenerates the plot, local pairs and manual counts
from the restored inputs. The current worker configuration is retained. View
bounds are given in display coordinates; aspect locking can enlarge a restored
view on a differently proportioned viewport while preserving its center.

`exports.build_export_tables` returns UTF-8 CSV text plus `README.txt`:

- Counts are recomputed with `count_cell_atoms` for each actual grain polygon
  and the picked layer, applying the saved GB filter only when enabled. No
  cached or viewport count is reused. Unavailable half-open values are blank.
- Vector xy components come from actual endpoint positions; the axial component
  uses reference half-indices plus the selected periodic image. The same
  physical displacement is converted to analysis/display, polar cubic and
  deformed lattice coordinates. Cross-grain vectors use both grain frames.
- Strain tables use the current per-grain deformation gradients and translations.
  SVD gives `F = R U`; `E = (F.T F - I)/2` is exported in the reference analysis
  plane and transformed to reference cubic coordinates. Rigid rotation and
  principal engineering strains remain distinct quantities.

Floating-point values use 17 significant digits. Headers and rows identify units
and frames; a₀/Å and a₀²/Å² conversions use the saved lattice constant. Incomplete
manual-cell or vector selections produce header-only tables, whereas an
unstrained grain has a real identity F and zero E. Numerical export errors fail
the save instead of silently writing partial or stale results.
