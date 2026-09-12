"""Uniform cell search and selected-cell strain fitting, independent of Qt."""

from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
from .crystal import get_geometry, rotation_matrix_2d
from .cells import StrainedCell, bases, determinant, reduce_cell, validate_cell_vertices

DEFAULT_STRAIN_PERCENT = 2.0
DEFAULT_SEARCH_INDEX = 12
MAX_VECTORS = 320


@dataclass(frozen=True)
class SelectedCellStrain:
    """Exact homogeneous alignment of four selected same-layer atom pairs."""

    cell: StrainedCell
    translations: np.ndarray  # two uniform shifts, x' = Fg x + tg
    vertices: np.ndarray  # (2,4,2), actual transformed endpoints
    origin: np.ndarray  # common C1, not necessarily the laboratory origin
    residual: float  # maximum selected-pair separation / a0
    rotations_deg: np.ndarray  # polar rigid rotation, separate from strain
    stretches: np.ndarray  # singular values, not eigenvalues of a nonsymmetric F


def strain_selected_cell(
    vertices,
    angle,
    percent=DEFAULT_STRAIN_PERCENT,
    lattice="FCC",
    axis="110",
    layer=0,
    max_rotation_deg=1.0,
):
    """Align all four pairs by bounded, homogeneous in-plane deformation.

    Minimize sum_g ||Fg-I||_F^2 under equality of the centered vertex sets.
    Small rotations are allowed and bounded separately using Fg = Rg Ug.
    A zero rotation limit selects the symmetric, pure-strain solve instead.
    No candidate search or individual atom snapping is used.
    The two centroids are placed at their original midpoint by uniform shifts.
    Integer same-layer edge vectors certify common translations; all FOUR pairs
    are validated independently, so an incompatible fourth corner is not ignored.
    This is geometric compatibility, not an elastic-energy minimum.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (2, 4, 2):
        raise ValueError("Select four paired atom vertices first")
    for polygon in vertices:
        validate_cell_vertices(polygon)
    if not np.isfinite(percent) or not 0 < percent <= 10:
        raise ValueError("Strain limit must be in (0,10]%")
    if not np.isfinite(angle) or not 0 <= angle <= 90:
        raise ValueError("Reference angle must be in [0,90] degrees")
    if not np.isfinite(max_rotation_deg) or not 0 <= max_rotation_deg <= 5:
        raise ValueError("Rotation limit must be in [0,5] degrees per grain")
    geometry = get_geometry(lattice, axis)
    if (
        not isinstance(layer, (int, np.integer))
        or not 0 <= layer < geometry.layer_count
    ):
        raise ValueError("Select one valid axial layer")
    grain_bases = bases(angle, lattice, axis)
    reference_offset = (
        geometry.layer_offsets_half_indices[layer] @ geometry.frame[:, :2] / 2
    )
    phase = np.linalg.solve(geometry.planar_basis, reference_offset)
    matrices = []
    for polygon, basis in zip(vertices, grain_bases):
        indices = polygon @ np.linalg.inv(basis).T - phase
        integer = np.rint(indices).astype(np.int64)
        if np.max(np.linalg.norm((indices - integer) @ basis.T, axis=1)) > 1e-7:
            raise ValueError(
                "Vertices must be original atoms in the same selected layer"
            )
        matrices.append((integer[[1, 3]] - integer[0]).T)
    if determinant(matrices[0]) * determinant(matrices[1]) <= 0:
        raise ValueError("The selected cells have incompatible orientation")

    centroids = vertices.mean(axis=1)
    a, b = vertices - centroids[:, None, :]
    pure_strain = max_rotation_deg == 0
    constraints = np.zeros((8, 6 if pure_strain else 8))
    root2 = np.sqrt(2)
    for i in range(4):
        x, y = a[i]
        u, v = b[i]
        if pure_strain:
            constraints[2 * i] = (x, 0, y / root2, -u, 0, -v / root2)
            constraints[2 * i + 1] = (0, y, x / root2, 0, -v, -u / root2)
        else:
            constraints[2 * i] = (x, y, 0, 0, -u, -v, 0, 0)
            constraints[2 * i + 1] = (0, 0, x, y, 0, 0, -u, -v)
    solution = np.linalg.lstsq(constraints, (b - a).ravel(), rcond=1e-11)[0]
    f = np.tile(np.eye(2), (2, 1, 1))
    if pure_strain:
        for g in (0, 1):
            f[g, 0, 0] += solution[3 * g]
            f[g, 1, 1] += solution[3 * g + 1]
            f[g, 0, 1] = f[g, 1, 0] = solution[3 * g + 2] / root2
    else:
        f += solution.reshape(2, 2, 2)
    left, stretches, right = np.linalg.svd(f)
    rotation = left @ right
    rotations_deg = np.degrees(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    if np.min(stretches) <= 1e-10 or np.any(np.linalg.det(f) <= 0):
        raise ValueError(
            "All four pairs cannot be aligned without collapse/reflection; choose another cell"
        )
    max_strain = float(np.max(np.abs(stretches - 1)))
    if max_strain > percent / 100 + 1e-12:
        raise ValueError(
            f"Least-change fit uses {100*max_strain:.6f}% principal strain, above the {percent:.4f}% limit. Nothing was changed."
        )
    if np.max(np.abs(rotations_deg)) > max_rotation_deg + 1e-9:
        raise ValueError(
            f"Least-change fit uses {np.max(np.abs(rotations_deg)):.6f}° rotation, above the {max_rotation_deg:.4f}° per-grain limit. Nothing was changed."
        )
    shifts = centroids.mean(axis=0) - np.einsum("gij,gj->gi", f, centroids)
    transformed = np.einsum("gij,gnj->gni", f, vertices) + shifts[:, None, :]
    residual = float(np.max(np.linalg.norm(transformed[0] - transformed[1], axis=1)))
    common = f[0] @ grain_bases[0] @ matrices[0]
    periodic_error = np.max(np.abs(common - f[1] @ grain_bases[1] @ matrices[1]))
    if residual > 1e-8 or periodic_error > 1e-8:
        raise ValueError(
            "No exact homogeneous alignment of all four pairs; choose another cell"
        )
    cell = StrainedCell(
        *matrices, f[0], f[1], common, max_strain, geometry.lattice, geometry.axis
    )
    return SelectedCellStrain(
        cell,
        shifts,
        transformed,
        transformed[:, 0].mean(axis=0),
        residual,
        rotations_deg,
        stretches,
    )


def candidate_vectors(angle, percent, extent, lattice="FCC", axis="110"):
    """Approximately compatible integer translations, without NxN distances."""
    if not 0 < percent <= 10 or not 2 <= extent <= 40:
        raise ValueError("strain must be (0,10]% and search index 2..40")
    e = percent / 100
    b1, b2 = bases(angle, lattice, axis)
    integers = np.array(
        [
            (x, y)
            for x in range(-extent, extent + 1)
            for y in range(-extent, extent + 1)
            if x > 0 or (x == 0 and y > 0)
        ],
        dtype=int,
    )
    v = integers @ b1.T
    lengths = np.linalg.norm(v, axis=1)
    inverse_b2 = np.linalg.inv(b2)
    nearest = np.rint(v @ inverse_b2.T).astype(int)
    # ||v-w|| <= e (||v||+||w||) is necessary for two bounded strains.
    # Convert the physical separation bound to an integer-coordinate bound;
    # the fixed FCC [110] factor sqrt(2) is invalid for other layer bases.
    inverse_norm = np.linalg.norm(inverse_b2, ord=2)
    radii = np.ceil((2 * e / (1 - e)) * lengths * inverse_norm).astype(int) + 1
    n1, n2, errors, sizes = [], [], [], []
    for dx in range(-int(radii.max()), int(radii.max()) + 1):
        for dy in range(-int(radii.max()), int(radii.max()) + 1):
            other = nearest + (dx, dy)
            w = other @ b2.T
            l2 = np.linalg.norm(w, axis=1)
            error = np.linalg.norm(v - w, axis=1) / (lengths + l2)
            mask = (
                (error <= e + 1e-12)
                & (np.max(np.abs(other), axis=1) <= extent)
                & (l2 > 0)
                & (radii >= max(abs(dx), abs(dy)))
            )
            n1.append(integers[mask])
            n2.append(other[mask])
            errors.append(error[mask])
            sizes.append((lengths + l2)[mask])
    i, j = np.concatenate(n1), np.concatenate(n2)
    err, size = np.concatenate(errors), np.concatenate(sizes)
    # Retain short translations and accurate larger translations.
    chosen = np.unique(
        np.r_[
            np.argsort(size, kind="stable")[: MAX_VECTORS // 2],
            np.argsort(err, kind="stable")[: MAX_VECTORS // 2],
        ]
    )
    return i[chosen], j[chosen]


def pareto_cells(cells, limit=12):
    """The tradeoff between reference atom count and required strain."""
    unique = {}
    for cell in cells:
        key = (cell.atoms, tuple(np.round(np.r_[cell.f1.ravel(), cell.f2.ravel()], 9)))
        unique.setdefault(key, cell)
    ordered = sorted(unique.values(), key=lambda c: (sum(c.atoms), c.max_strain))
    result, best = [], np.inf
    for cell in ordered:
        if cell.max_strain < best - 1e-10:
            result.append(cell)
            best = cell.max_strain
    return result[:limit]


def solve_cells_chunk(
    angle, percent, integers1, integers2, start, stop, lattice="FCC", axis="110"
):
    """Batched least-norm symmetric deformation solve for vector pairs."""
    geometry = get_geometry(lattice, axis)
    lattice, axis = geometry.lattice, geometry.axis
    b1, b2 = bases(angle, lattice, axis)
    first, second = np.triu_indices(len(integers1), 1)
    use = (first >= start) & (first < stop)
    first, second = first[use], second[use]
    m1 = np.stack([integers1[first], integers1[second]], axis=2)
    m2 = np.stack([integers2[first], integers2[second]], axis=2)
    d1, d2 = np.linalg.det(m1), np.linalg.det(m2)
    mask = (np.abs(d1) >= 0.5) & (d1 * d2 > 0)
    m1, m2 = m1[mask], m2[mask]
    if not len(m1):
        return os.getpid(), []
    a, b = b1 @ m1, b2 @ m2
    # Frobenius-orthonormal [Sxx,Syy,sqrt(2)Sxy] coordinates per grain.
    c = np.zeros((len(a), 4, 6))
    for col in (0, 1):
        c[:, 2 * col, 0] = a[:, 0, col]
        c[:, 2 * col, 2] = a[:, 1, col] / np.sqrt(2)
        c[:, 2 * col, 3] = -b[:, 0, col]
        c[:, 2 * col, 5] = -b[:, 1, col] / np.sqrt(2)
        c[:, 2 * col + 1, 1] = a[:, 1, col]
        c[:, 2 * col + 1, 2] = a[:, 0, col] / np.sqrt(2)
        c[:, 2 * col + 1, 4] = -b[:, 1, col]
        c[:, 2 * col + 1, 5] = -b[:, 0, col] / np.sqrt(2)
    rhs = (b - a).transpose(0, 2, 1).reshape(-1, 4)
    sol = (np.linalg.pinv(c, rcond=1e-11) @ rhs[..., None])[..., 0]
    f = np.tile(np.eye(2), (len(a), 2, 1, 1))
    for g in (0, 1):
        f[:, g, 0, 0] += sol[:, 3 * g]
        f[:, g, 1, 1] += sol[:, 3 * g + 1]
        f[:, g, 0, 1] = f[:, g, 1, 0] = sol[:, 3 * g + 2] / np.sqrt(2)
    eig = np.linalg.eigvalsh(f)
    strain = np.max(np.abs(eig - 1), axis=(1, 2))
    common = f[:, 0] @ a
    residual = np.max(np.abs(common - f[:, 1] @ b), axis=(1, 2))
    valid = (
        (strain <= percent / 100 + 1e-12)
        & (np.min(eig, axis=(1, 2)) > 0)
        & (residual < 1e-8)
    )
    cells = []
    ordered = sorted(
        np.flatnonzero(valid),
        key=lambda k: (abs(determinant(m1[k])) + abs(determinant(m2[k])), strain[k]),
    )
    best = np.inf
    for k in ordered:
        if strain[k] >= best - 1e-10:
            continue
        best = strain[k]
        u, v, cell = reduce_cell(m1[k], m2[k], common[k])
        cells.append(
            StrainedCell(u, v, f[k, 0], f[k, 1], cell, float(strain[k]), lattice, axis)
        )
    return os.getpid(), cells


def strain_tensors(cell, angle):
    """Return dimensionless 3D Green-Lagrange E in each reference cubic frame."""
    projection = get_geometry(cell.lattice, cell.axis).frame
    tensors = []
    for g, f in enumerate((cell.f1, cell.f2)):
        t = np.deg2rad((1 if g == 0 else -1) * angle / 2)
        r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        f3 = np.eye(3)
        f3[:2, :2] = r.T @ f @ r
        tensors.append(projection @ ((f3.T @ f3 - np.eye(3)) / 2) @ projection.T)
    return tuple(tensors)


def selected_cell_strain_readout(
    fit: SelectedCellStrain,
    angle: float,
    *,
    strain_limit_percent: float,
    rotation_limit_deg: float,
) -> str:
    """Describe an applied fit without changing it or depending on the view.

    F, R, U and the 2D E tensor use unrotated analysis coordinates. The 3D E
    tensor uses each original grain's cubic frame. Principal strains are
    engineering stretches minus one, not the eigenvalues of E.
    """
    cell = fit.cell
    geometry = get_geometry(cell.lattice, cell.axis)

    def array(values):
        return np.array2string(
            np.asarray(values, dtype=float),
            formatter={"float_kind": lambda value: f"{value:+.8f}"},
            max_line_width=100,
        )

    lines = [
        "SELECTED-CELL STRAIN DETAILS",
        f"{geometry.lattice}; tilt axis {geometry.axis_label}",
        f"Reference θ = {angle:.8f}°",
        f"Polar-frame θ = {angle + fit.rotations_deg[0] - fit.rotations_deg[1]:.8f}°",
        f"Max |principal strain| = {100 * cell.max_strain:.8f}%",
    ]
    # Put both grains' main results first, before the longer tensor report.
    for grain in (0, 1):
        lines.extend(
            [
                f"G{grain + 1} rotation = {fit.rotations_deg[grain]:+.8f}°",
                "principal strains (%):",
                array(100 * (fit.stretches[grain] - 1)),
            ]
        )
    lines.extend(
        [
            "",
            "Limits used for this fit",
            f"Strain limit = {strain_limit_percent:.6f}%",
            f"Rotation limit / grain = {rotation_limit_deg:.6f}°",
        ]
    )
    for grain, (f, e_cubic) in enumerate(
        zip((cell.f1, cell.f2), strain_tensors(cell, angle))
    ):
        rotation = rotation_matrix_2d(fit.rotations_deg[grain])
        stretch = rotation.T @ f
        e_analysis = (f.T @ f - np.eye(2)) / 2
        lines.extend(
            [
                "",
                f"G{grain + 1} ({'blue' if grain == 0 else 'red'})",
                f"Polar rotation = {fit.rotations_deg[grain]:+.8f}°",
                "Principal stretch factors:",
                array(fit.stretches[grain]),
                "principal strains (%):",
                array(100 * (fit.stretches[grain] - 1)),
                f"Area ratio det(F) = {np.linalg.det(f):.10f}",
                f"Area change = {100 * (np.linalg.det(f) - 1):+.8f}%",
                "Uniform shift t / a₀ (analysis x,y):",
                array(fit.translations[grain]),
                "F (analysis x,y; dimensionless):",
                array(f),
                "R (analysis x,y; dimensionless):",
                array(rotation),
                "U (analysis x,y; dimensionless):",
                array(stretch),
                "E = (FᵀF - I)/2 (analysis x,y; dimensionless):",
                array(e_analysis),
                "E (reference grain cubic [100],[010],[001]; dimensionless):",
                array(e_cubic),
            ]
        )
    residuals = np.linalg.norm(fit.vertices[0] - fit.vertices[1], axis=1)
    lines.extend(
        [
            "",
            "Four-pair alignment",
            f"Maximum residual = {fit.residual:.6e} a₀",
            *[
                f"C{index + 1} pair residual = {value:.6e} a₀"
                for index, value in enumerate(residuals)
            ],
            "Common C1 origin / a₀ (analysis x,y):",
            array(fit.origin),
            "Common translation vectors a,b as columns / a₀ (analysis x,y):",
            array(cell.cell),
            f"Common in-plane area = {abs(np.linalg.det(cell.cell)):.8f} a₀²",
            "Common translations verified; the cell may be nonprimitive.",
            "Full-lattice exact CSL is recomputed separately, layer by layer.",
            "",
            "Coordinate convention",
            "x' = F x + t; x is the original atom position in analysis x,y.",
            "F = R U: rigid rotation R and symmetric stretch U.",
            "Positive rotation is counterclockwise in analysis x,y.",
            "Display rotation is excluded; axial stretch = 1 (axial strain = 0).",
            "principal strains = stretch - 1; positive = tension, negative = compression.",
            "One uniform transform per grain, applied to ALL axial layers.",
            "Not stress-free; no atomic relaxation or energy minimization.",
        ]
    )
    return "\n".join(lines)


def tensor_readout(cell, angle, selected=False):
    """3D strain and common translations, with units and frame explicit."""
    geometry = get_geometry(cell.lattice, cell.axis)
    lines = [
        ("Full axial repeat (all layers): " if selected else "") + cell.label(),
        f"{geometry.lattice}; {'selected-cell F1/F2' if selected else 'symmetric F1/F2'}; {geometry.axis_label} axial strain = 0",
        "Green-Lagrange E (grain cubic [100],[010],[001] frame):",
    ]
    for g, e3 in enumerate(strain_tensors(cell, angle)):
        lines.append(
            f"G{g+1}:\n" + np.array2string(e3, precision=6, suppress_small=True)
        )
    lines.append(
        "Common C1,C2 columns / a0 (unrotated analysis x,y):\n"
        + np.array2string(cell.cell, precision=6)
    )
    axial = geometry.axial_repeat_half_indices
    coefficient = "a0/2"
    if np.all(axial % 2 == 0):
        axial = axial // 2
        coefficient = "a0"
    axial_indices = " ".join(str(int(x)) for x in axial)
    lines.append(
        f"Common C3 = ({coefficient})[{axial_indices}]; M1/M2 in A-layer basis:"
    )
    lines.extend(np.array2string(m) for m in (cell.m1, cell.m2))
    lines.append(
        "Cell may be nonprimitive; counts are not Sigma. "
        + (
            "Selected-vertex least-change fit; no energy fit."
            if selected
            else "Bounded search, no energy fit."
        )
    )
    return "\n".join(lines)
