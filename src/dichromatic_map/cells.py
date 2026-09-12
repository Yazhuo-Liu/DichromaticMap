"""Cell geometry, common-cell results and whole-region atom counting."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .crystal import get_geometry, projected_columns, GeometryLimitError

SIDE_TOLERANCE = 1.0e-9


def validate_cell_vertices(vertices):
    """Four distinct convex vertices in perimeter order, in model coordinates."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (4, 2) or not np.all(np.isfinite(vertices)):
        raise ValueError("Select four finite vertices in perimeter order")
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    tolerance = 1e-8 * max(1.0, float(lengths.max()))
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(
        edges[:, 0], -1
    )
    if np.any(lengths <= tolerance) or not (
        np.all(turns > tolerance * lengths) or np.all(turns < -tolerance * lengths)
    ):
        raise ValueError(
            "Vertices must form a convex, non-crossing cell; pick them around its perimeter"
        )
    return vertices


def cell_membership(points, vertices):
    """Interior, boundary and (if parallelogram) half-open membership masks.

    Half-open means p0 + u*(p1-p0) + v*(p3-p0), 0 <= u,v < 1.
    It avoids counting the upper two edges twice when cells are repeated.
    Geometric parallelogram shape does not prove crystal periodicity.
    """
    vertices = validate_cell_vertices(vertices)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    edges = np.roll(vertices, -1, axis=0) - vertices
    lengths = np.linalg.norm(edges, axis=1)
    tolerance = 1e-8 * max(1.0, float(lengths.max()))
    orientation = np.sign(np.linalg.det(np.column_stack((edges[0], -edges[-1]))))
    delta = points[:, None, :] - vertices
    distances = (
        orientation
        * (edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0])
        / lengths
    )
    closed = np.all(distances >= -tolerance, axis=1)
    interior = np.all(distances > tolerance, axis=1)
    half_open = None
    if (
        np.linalg.norm((vertices[2] - vertices[1]) - (vertices[3] - vertices[0]))
        <= tolerance
    ):
        basis = np.column_stack((vertices[1] - vertices[0], vertices[3] - vertices[0]))
        inverse = np.linalg.inv(basis)
        uv = (points - vertices[0]) @ inverse.T
        eps = tolerance * np.linalg.norm(inverse, axis=1)
        half_open = np.all((uv >= -eps) & (uv < 1 - eps), axis=1)
    return interior, closed & ~interior, half_open


@dataclass(frozen=True)
class CellAtomCounts:
    interior: np.ndarray  # grain x axial layer, one full axial repeat
    boundary: np.ndarray
    half_open: np.ndarray | None
    areas: np.ndarray  # one actual polygon area per grain, a0 squared
    half_open_available: np.ndarray  # parallelogram test per grain
    half_open_edges: np.ndarray | None  # unique edge atoms retained
    half_open_corners: np.ndarray | None  # unique corner atoms retained

    @property
    def area(self):
        """Compatibility for a shared-area cell; never average different cells."""
        if not np.allclose(self.areas, self.areas[0], atol=1e-10, rtol=1e-10):
            raise ValueError("Grain polygon areas differ; use areas[grain_index]")
        return float(self.areas[0])


def count_cell_atoms(
    vertices,
    angle,
    deformations,
    lattice="FCC",
    axis="110",
    boundary_points=None,
    region_states=(True, True, True, True),
    layer=-1,
    translations=None,
):
    """Count the entire selected cell, independently of viewport and rendering.

    Vertices may be one shared (4,2) polygon or two independent (2,4,2)
    polygons built from each grain's actual atom positions. Each grain is
    counted inside its own polygon; the viewer passes the picked layer.
    layer=-1 retains all-layer counting for non-GUI callers.
    Generation uses the same interactive allocation guard as normal rendering.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape == (4, 2):
        vertices = np.stack((vertices, vertices))
    if vertices.shape != (2, 4, 2):
        raise ValueError("Provide a (4,2) common polygon or (2,4,2) grain polygons")
    for polygon in vertices:
        validate_cell_vertices(polygon)
    geometry = get_geometry(lattice, axis)
    if (
        not isinstance(layer, (int, np.integer))
        or not -1 <= layer < geometry.layer_count
    ):
        raise ValueError("Count layer must be -1 or a valid axial layer index")
    inside_counts = np.zeros((2, geometry.layer_count), dtype=int)
    translations = (
        np.zeros((2, 2))
        if translations is None
        else np.asarray(translations, dtype=float)
    )
    if translations.shape != (2, 2) or not np.all(np.isfinite(translations)):
        raise ValueError("Provide two finite grain translations")
    edge_counts = np.zeros_like(inside_counts)
    half_counts = np.zeros_like(inside_counts)
    half_edge_counts = np.zeros_like(inside_counts)
    half_corner_counts = np.zeros_like(inside_counts)
    is_parallelogram = np.zeros(2, dtype=bool)
    areas = np.zeros(2)
    for grain_index, sign in enumerate((1, -1)):
        polygon = vertices[grain_index]
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        margin = 1e-7 * max(1.0, float(np.max(high - low)))
        width, height = high - low + 2 * margin
        center = (low + high) / 2
        try:
            grain = projected_columns(
                width,
                height,
                sign * angle / 2,
                center,
                deformations[grain_index],
                lattice,
                axis,
                translations[grain_index],
            )
        except GeometryLimitError as error:
            raise GeometryLimitError(
                "Manual cell exceeds the atom enumeration limit; select a smaller cell. "
                "View zoom does not affect counting."
            ) from error
        inside, boundary, half_open = cell_membership(grain.positions, polygon)
        half_open_edges = None
        half_open_corners = None
        if half_open is not None:
            corner_tolerance = 1e-8 * max(1.0, float(np.max(high - low)))
            at_corner = np.any(
                np.linalg.norm(
                    grain.positions[:, None, :] - polygon[None, :, :], axis=2
                )
                <= corner_tolerance,
                axis=1,
            )
            half_open_corners = half_open & at_corner
            half_open_edges = half_open & boundary & ~at_corner
        visible = np.ones(len(grain.positions), dtype=bool)
        if layer >= 0:
            visible &= grain.layers == layer
        if boundary_points is not None and len(boundary_points) == 2:
            start, end = np.asarray(boundary_points)
            direction = end - start
            cross = direction[0] * (grain.positions[:, 1] - start[1]) - direction[1] * (
                grain.positions[:, 0] - start[0]
            )
            visible &= (region_states[2 * grain_index] & (cross >= -1e-9)) | (
                region_states[2 * grain_index + 1] & (cross <= 1e-9)
            )
        for mask, counts in (
            (inside, inside_counts),
            (boundary, edge_counts),
            (half_open, half_counts),
            (half_open_edges, half_edge_counts),
            (half_open_corners, half_corner_counts),
        ):
            if mask is not None:
                counts[grain_index] = np.bincount(
                    grain.layers[mask & visible], minlength=geometry.layer_count
                )
        is_parallelogram[grain_index] = half_open is not None
        # Translation-stable area of this grain's actual polygon.
        relative = polygon - polygon[0]
        areas[grain_index] = (
            abs(
                np.sum(
                    relative[:, 0] * np.roll(relative[:, 1], -1)
                    - relative[:, 1] * np.roll(relative[:, 0], -1)
                )
            )
            / 2
        )
    half_open_result = np.any(is_parallelogram)
    return CellAtomCounts(
        interior=inside_counts,
        boundary=edge_counts,
        half_open=half_counts if half_open_result else None,
        areas=areas,
        half_open_available=is_parallelogram,
        half_open_edges=half_edge_counts if half_open_result else None,
        half_open_corners=half_corner_counts if half_open_result else None,
    )


def selected_region_mask(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    keep_left: bool,
    keep_right: bool,
) -> np.ndarray:
    direction = second - first
    signed_cross_product = direction[0] * (points[:, 1] - first[1]) - direction[1] * (
        points[:, 0] - first[0]
    )
    keep = np.zeros(len(points), dtype=bool)
    if keep_left:
        keep |= signed_cross_product >= -SIDE_TOLERANCE
    if keep_right:
        keep |= signed_cross_product <= SIDE_TOLERANCE
    return keep


def bases(angle, lattice="FCC", axis="110"):
    t = np.deg2rad(angle / 2)
    c, s = np.cos(t), np.sin(t)
    r = np.array([[c, -s], [s, c]])
    b = get_geometry(lattice, axis).planar_basis
    return r @ b, r.T @ b


def determinant(m):
    return int(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])


@dataclass(frozen=True)
class StrainedCell:
    m1: np.ndarray
    m2: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    cell: np.ndarray  # columns, laboratory projection, in units of a0
    max_strain: float
    lattice: str = "FCC"
    axis: str = "110"

    @property
    def atoms(self):
        layers = get_geometry(self.lattice, self.axis).layer_count
        return tuple(layers * abs(determinant(m)) for m in (self.m1, self.m2))

    def label(self):
        n1, n2 = self.atoms
        return f"{n1}/{n2} atoms | max strain {100*self.max_strain:.3f}%"


def reduce_cell(m1, m2, cell):
    """Gauss-reduce common vectors with identical integer column operations."""
    m1, m2, cell = m1.copy(), m2.copy(), cell.copy()
    for _ in range(64):
        if np.dot(cell[:, 1], cell[:, 1]) < np.dot(cell[:, 0], cell[:, 0]):
            cell = cell[:, ::-1]
            m1 = m1[:, ::-1]
            m2 = m2[:, ::-1]
        q = int(
            np.rint(np.dot(cell[:, 0], cell[:, 1]) / np.dot(cell[:, 0], cell[:, 0]))
        )
        if q == 0:
            break
        cell[:, 1] -= q * cell[:, 0]
        m1[:, 1] -= q * m1[:, 0]
        m2[:, 1] -= q * m2[:, 0]
    if np.linalg.det(cell) < 0:
        cell[:, 1] *= -1
        m1[:, 1] *= -1
        m2[:, 1] *= -1
    return m1, m2, cell
