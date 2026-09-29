"""Same-layer cell completion from paired atoms, independent of the viewer.

Symmetries act on translation vectors, with grain exchange when required.
Generated vertices are certified by integer lattice coordinates, not by a
nearest-neighbor cutoff. No deformation is applied by these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product

import numpy as np

from .cells import bases, determinant, validate_cell_vertices
from .crystal import get_geometry, rotation_matrix_2d
from .strain import SelectedCellStrain, strain_selected_cell


@dataclass(frozen=True)
class CellCompletion:
    description: str
    source: str
    vertices: np.ndarray  # grain x four perimeter vertices x xy; original atoms
    atoms: tuple[int, int]  # reference atoms per selected layer, per grain
    areas: tuple[float, float]  # original cell areas / a0^2
    fit: SelectedCellStrain | None
    error: str | None


def _integer_vertices(vertices, grain_bases, geometry, layer):
    offset = geometry.layer_offsets_half_indices[layer] @ geometry.frame[:, :2] / 2
    phase = np.linalg.solve(geometry.planar_basis, offset)
    result = []
    for points, basis in zip(vertices, grain_bases):
        coordinates = np.linalg.solve(basis, points.T).T - phase
        integers = np.rint(coordinates)
        if np.max(np.linalg.norm((coordinates - integers) @ basis.T, axis=1)) > 1e-7:
            raise ValueError("Vertices must be original atoms in the same selected layer")
        result.append(integers.astype(np.int64))
    return np.array(result), phase


@lru_cache(maxsize=128)
def _plane_operations(lattice, axis):
    """Restrictions of cubic point operations preserving the axial line."""
    geometry = get_geometry(lattice, axis)
    direction = geometry.axis_indices
    plane = geometry.frame[:, :2]
    operations = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            cubic = np.eye(3, dtype=int)[list(permutation)] * np.array(signs)[:, None]
            transformed = cubic @ direction
            if not (np.array_equal(transformed, direction)
                    or np.array_equal(transformed, -direction)):
                continue
            operation = plane.T @ cubic @ plane
            if not any(np.allclose(operation, old, atol=1e-10, rtol=0) for old in operations):
                operation.setflags(write=False)
                operations.append(operation)
    return tuple(operations)


def _translation_symmetries(angle, geometry, grain_bases):
    """Certify each operation in BOTH same-layer translation lattices."""
    rotations = [rotation_matrix_2d(sign * angle / 2) for sign in (1, -1)]
    for exchange in (False, True):
        source = (1, 0) if exchange else (0, 1)
        for local in _plane_operations(geometry.lattice, geometry.axis):
            operation = rotations[0] @ local @ rotations[source[0]].T
            valid = True
            for target, origin in enumerate(source):
                mapping = np.linalg.solve(grain_bases[target], operation @ grain_bases[origin])
                integer = np.rint(mapping).astype(np.int64)
                if (not np.allclose(mapping, integer, atol=1e-9, rtol=0)
                        or abs(determinant(integer)) != 1):
                    valid = False
                    break
            if valid:
                if np.linalg.det(operation) < 0:
                    degrees = np.degrees(np.arctan2(operation[1, 0], operation[0, 0])) / 2 % 180
                    description = f"Mirror at {degrees:.3f}°"
                else:
                    degrees = np.degrees(np.arctan2(operation[1, 0], operation[0, 0]))
                    description = f"Rotation {degrees:+.3f}°"
                yield operation, source, description + (" · exchange G1/G2" if exchange else "")


def cell_completion_candidates(
    vertices, angle, *, lattice="FCC", axis="110", layer=0,
    percent=2.0, max_rotation_deg=1.0,
):
    """Complete two paired vertices by symmetry, or three by parallelogram closure.

    Inputs are unstrained analysis coordinates in perimeter order. Every
    candidate preserves the picked endpoints and contains certified same-layer
    atoms. Fit failures remain visible as candidates with an explanatory error.
    These bounded cubic symmetry candidates need not be primitive or exhaustive.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape not in ((2, 2, 2), (2, 3, 2)) or not np.all(np.isfinite(vertices)):
        raise ValueError("Select two or three finite paired vertices first")
    if not np.isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError("Reference angle must be in [0,180] degrees")
    if not np.isfinite(percent) or not 0 < percent <= 10:
        raise ValueError("Strain limit must be in (0,10]%")
    if not np.isfinite(max_rotation_deg) or not 0 <= max_rotation_deg <= 5:
        raise ValueError("Rotation limit must be in [0,5] degrees per grain")
    geometry = get_geometry(lattice, axis)
    if not isinstance(layer, (int, np.integer)) or not 0 <= layer < geometry.layer_count:
        raise ValueError("Select one valid axial layer")
    grain_bases = bases(angle, geometry.lattice, geometry.axis)
    indices, phase = _integer_vertices(vertices, grain_bases, geometry, layer)
    count = vertices.shape[1]
    proposed = []
    if count == 3:
        fourth = indices[:, 0] + indices[:, 2] - indices[:, 1]
        integer_cells = np.concatenate((indices, fourth[:, None]), axis=1)
        proposed.append(("Parallelogram closure", "closure", integer_cells))
    else:
        edge = vertices[:, 1] - vertices[:, 0]
        for operation, source, description in _translation_symmetries(angle, geometry, grain_bases):
            second_edge = edge[list(source)] @ operation.T
            steps = np.array([
                np.rint(np.linalg.solve(basis, vector)).astype(np.int64)
                for basis, vector in zip(grain_bases, second_edge)
            ])
            integer_cells = np.stack(
                (indices[:, 0], indices[:, 1], indices[:, 1] + steps,
                 indices[:, 0] + steps), axis=1,
            )
            proposed.append((description, "symmetry", integer_cells))

    results, seen = [], set()
    for description, source, integer_cells in proposed:
        key = tuple(integer_cells.ravel())
        if key in seen:
            continue
        seen.add(key)
        matrices = (integer_cells[:, [1, 3]] - integer_cells[:, :1]).transpose(0, 2, 1)
        determinants = [determinant(matrix) for matrix in matrices]
        if determinants[0] * determinants[1] <= 0:
            continue
        polygons = np.array([
            (integer + phase) @ basis.T for integer, basis in zip(integer_cells, grain_bases)
        ])
        polygons[:, :count] = vertices  # preserve every originally selected atom
        try:
            for polygon in (*polygons, polygons.mean(axis=0)):
                validate_cell_vertices(polygon)
        except ValueError:
            continue
        areas = tuple(float(abs(np.linalg.det(basis @ matrix)))
                      for basis, matrix in zip(grain_bases, matrices))
        try:
            fit = strain_selected_cell(
                polygons, angle, percent, geometry.lattice, geometry.axis, layer,
                max_rotation_deg=max_rotation_deg,
            )
            error = None
        except ValueError as exc:
            fit, error = None, str(exc)
        results.append(CellCompletion(
            description, source, polygons, tuple(abs(d) for d in determinants), areas, fit, error,
        ))
    return sorted(results, key=lambda candidate: (
        candidate.fit is None, sum(candidate.atoms),
        candidate.fit.cell.max_strain if candidate.fit is not None else float("inf"),
        candidate.description,
    ))
