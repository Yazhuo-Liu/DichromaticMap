"""Local near pairs and exact/strained cells for standalone cubic tilt viewers.

Local matching keeps both input lattices unchanged and pairs mutual nearest
neighbors within a distance cutoff, separately in every axial phase. It is a
geometric candidate analysis, not a relaxation or a stress-free equilibrium.

Exact CSL cells use integer congruences without invoking the strain search.

Selected local cells have a separate direct fit: align all four actual pairs
with homogeneous Fg and uniform shifts about their centroids. Small polar
rotations and principal strains have independent limits; there is no search.

For integer M1/M2, minimize ||F1-I||_F^2 + ||F2-I||_F^2 subject to
F1 B1 M1 = F2 B2 M2, symmetric positive-definite F1/F2, and a bound on
principal stretch minus one. No additional polar rotation is introduced.
The chosen axial direction is unchanged (plane strain). This is geometric
compatibility, not elastic-energy minimization, and a bounded candidate
search, not a proof of the smallest cell. Green-Lagrange tensors are also
reported in each grain's cubic frame.

The A-layer lattice and the axial repeat depend on crystal structure and
tilt axis. Common in-plane vectors preserve every axial phase. A full axial lattice
translation completes a common 3D cell, possibly nonprimitive.
Reference atom counts are layer_count*abs(det(Mg)), NOT Sigma values.
"""

from __future__ import annotations
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import gcd
import multiprocessing
import os
import numpy as np

from tilt_gb_crystallography import (
    get_geometry,
    congruence_kernel,
    validate_cell_vertices,
)

DEFAULT_STRAIN_PERCENT = 2.0
DEFAULT_SEARCH_INDEX = 12
NEAR_COLOR = "#009b87"
MAX_VECTORS = 320
DEFAULT_LOCAL_DISTANCE = 0.05  # pair separation in a0, not a strain percentage
LOCAL_COLOR = "#aa45bb"


@dataclass(frozen=True)
class LocalPairs:
    """Original endpoints of one-to-one same-layer near pairs (units a0)."""

    first: np.ndarray
    second: np.ndarray
    layers: np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=np.int16))

    @classmethod
    def concatenate(cls, batches):
        batches = list(batches)
        if not batches:
            return cls.empty()
        return cls(
            *(
                np.concatenate([getattr(batch, name) for batch in batches])
                for name in ("first", "second", "layers")
            )
        )

    @property
    def midpoints(self):
        return (self.first + self.second) / 2

    @property
    def distances(self):
        return np.linalg.norm(self.first - self.second, axis=1)


def _nearest_in_radius(first, second, radius):
    """Spatial bins + bounded vector batches, without a dense distance matrix.

    Search *all* occupants of the nine neighboring bins. Unlike the tiny-
    tolerance exact CSL hash, a local-radius bin may contain multiple atoms.
    Equal-distance ties use the endpoint's lexicographic coordinate order so
    pairing does not depend on worker count or input enumeration order.
    """
    nearest = np.full(len(first), -1, dtype=np.int64)
    if not len(first) or not len(second):
        return nearest
    # Work relative to a common origin, avoiding packed-key wraparound on pan.
    origin = np.minimum(first.min(axis=0), second.min(axis=0))
    first_cells = np.floor((first - origin) / radius).astype(np.int64)
    second_cells = np.floor((second - origin) / radius).astype(np.int64)
    dtype = np.dtype([("x", np.int64), ("y", np.int64)])

    def keys(cells):
        return np.ascontiguousarray(cells).view(dtype).reshape(-1)

    # Sort each bin geometrically as well as by cell address.
    order = np.lexsort(
        (second[:, 1], second[:, 0], second_cells[:, 1], second_cells[:, 0])
    )
    sorted_keys = keys(second_cells[order])
    coordinate_order = np.lexsort((second[:, 1], second[:, 0]))
    rank = np.empty(len(second), dtype=int)
    rank[coordinate_order] = np.arange(len(second))
    for start in range(0, len(first), 8192):
        stop = min(start + 8192, len(first))
        best = np.full(stop - start, np.inf)
        best_rank = np.full(stop - start, len(second), dtype=int)
        chosen = np.full(stop - start, -1, dtype=int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                addresses = keys(first_cells[start:stop] + (dx, dy))
                lower = np.searchsorted(sorted_keys, addresses, side="left")
                upper = np.searchsorted(sorted_keys, addresses, side="right")
                for occupant in range(int(np.max(upper - lower, initial=0))):
                    rows = np.flatnonzero(lower + occupant < upper)
                    targets = order[lower[rows] + occupant]
                    delta = first[start + rows] - second[targets]
                    squared = np.einsum("ij,ij->i", delta, delta)
                    better = (squared <= radius * radius) & (
                        (squared < best[rows])
                        | ((squared == best[rows]) & (rank[targets] < best_rank[rows]))
                    )
                    chosen[rows[better]] = targets[better]
                    best[rows[better]] = squared[better]
                    best_rank[rows[better]] = rank[targets[better]]
        nearest[start:stop] = chosen
    return nearest


def local_near_pairs(
    grain1, grain2, distance=DEFAULT_LOCAL_DISTANCE, layers=None, exact_tolerance=1e-6
):
    """Mutual nearest same-height pairs with exact_tol < d <= distance.

    Exact pairs participate in nearest-neighbor assignment but are omitted
    from this overlay: they are already highlighted by the exact CSL detector.
    Each endpoint can be used at most once. Inputs are never modified.
    """
    if not np.isfinite(distance) or not 0 < distance <= 0.5:
        raise ValueError("Local pair distance must be in (0, 0.5] a0")
    if not np.isfinite(exact_tolerance) or exact_tolerance < 0:
        raise ValueError("Exact tolerance must be finite and nonnegative")
    if layers is None:
        layers = np.union1d(grain1.layers, grain2.layers)
    batches = []
    for layer in layers:
        first = grain1.positions[grain1.layers == layer]
        second = grain2.positions[grain2.layers == layer]
        forward = _nearest_in_radius(first, second, distance)
        reverse = _nearest_in_radius(second, first, distance)
        i = np.flatnonzero(forward >= 0)
        j = forward[i]
        mutual = reverse[j] == i
        i, j = i[mutual], j[mutual]
        keep = np.linalg.norm(first[i] - second[j], axis=1) > exact_tolerance
        batches.append(
            LocalPairs(
                first[i[keep]],
                second[j[keep]],
                np.full(np.count_nonzero(keep), layer, dtype=np.int16),
            )
        )
    return LocalPairs.concatenate(batches)


def local_match_layers_worker(grain1, grain2, distance, layers):
    return os.getpid(), local_near_pairs(grain1, grain2, distance, layers)


def worker_initializer():
    """Avoid each process starting its own BLAS thread team when available."""
    global _blas_limit
    try:
        from threadpoolctl import threadpool_limits

        _blas_limit = threadpool_limits(limits=1)
    except ImportError:
        pass  # NumPy-only installations remain supported.


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


@lru_cache(maxsize=256)
def exact_csl_cell(angle, max_denominator=128, lattice="FCC", axis="110"):
    """Exact, layer-preserving common cell; no strain search or atom matching.

    Recognize tan(theta/2)/sqrt(axis.axis) = n/m to 1e-9 degrees.
    In the selected A-layer basis, B2^-1 B1 = P/d with integer P.
    Build the rational 3D quaternion rotation and express it in the computed
    planar lattice basis. A general integer-congruence kernel handles axes
    whose coefficients are not invertible modulo the denominator.
    This is primitive in the A-preserving plane, not necessarily in 3D.
    Noncommensurate/unrecognized angles return None, never a strained fit.
    """
    geometry = get_geometry(lattice, axis)
    lattice, axis = geometry.lattice, geometry.axis
    norm_squared = geometry.axis_norm_squared
    if not np.isfinite(angle) or not 0 <= angle <= 90:
        return None
    ratio = Fraction(float(np.tan(np.deg2rad(angle / 2)) / np.sqrt(norm_squared)))
    ratio = ratio.limit_denominator(max_denominator)
    n, m = ratio.numerator, ratio.denominator
    recognized = np.degrees(2 * np.arctan2(np.sqrt(norm_squared) * n, m))
    if abs(recognized - angle) > 1e-9:
        return None
    direction = geometry.axis_indices.astype(object)
    x, y, z = direction
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=object)
    rotation_numerator = (
        (m * m - n * n * norm_squared) * np.eye(3, dtype=object)
        + 2 * n * n * np.outer(direction, direction)
        + 2 * m * n * cross
    )
    h = geometry.basis_half_indices.astype(object)
    rows = max(((0, 1), (0, 2), (1, 2)), key=lambda ij: abs(determinant(h[list(ij)])))
    sub = h[list(rows)]
    minor = determinant(sub)
    adjugate = np.array(
        [[sub[1, 1], -sub[0, 1]], [-sub[1, 0], sub[0, 0]]], dtype=object
    )
    p = adjugate @ (rotation_numerator @ h)[list(rows)]
    d = minor * (m * m + norm_squared * n * n)
    if d < 0:
        d, p = -d, -p
    factor = gcd(d, gcd(*(int(x) for x in p.ravel())))
    d, p = d // factor, p // factor
    m1 = congruence_kernel(p, d)
    m2 = np.asarray((p @ m1) // d, dtype=np.int64)
    b1, b2 = bases(angle, lattice, axis)
    m1, m2, cell = reduce_cell(m1, m2, b1 @ m1)
    # The recognition tolerance is not permission to create non-common corners.
    if not np.allclose(cell, b2 @ m2, atol=1e-8, rtol=0):
        return None
    return StrainedCell(m1, m2, np.eye(2), np.eye(2), cell, 0.0, lattice, axis)


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


class NearSearch:
    """Bounded asynchronous search; stale requests never publish results.

    Preparation and pair solves run off the UI thread. At most workers jobs
    are in flight. Invalidation drops pending jobs and lets bounded running
    chunks finish before submitting the next request.
    """

    def __init__(self, workers=1, executor=None):
        self.workers = max(1, int(workers))
        self.executor = executor
        self.owns_executor = executor is None
        self.generation = 0
        self.jobs = deque()
        self.running = {}
        self.parts = []
        self.total = self.completed = 0
        self.busy = False
        self.error = None
        self.process_ids = set()

    def cancel(self):
        self.generation += 1
        self.jobs.clear()
        self.parts.clear()
        self.busy = False
        self.error = None
        self.total = self.completed = 0
        for future in self.running:
            future.cancel()

    def request(self, angle, percent, extent, lattice="FCC", axis="110"):
        self.cancel()
        geometry = get_geometry(lattice, axis)
        if self.executor is None:
            self.executor = (
                ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=worker_initializer,
                )
                if self.workers > 1
                else ThreadPoolExecutor(max_workers=1)
            )
        self.args = (angle, percent, extent, geometry.lattice, geometry.axis)
        self.jobs.append(("prepare", candidate_vectors, self.args))
        self.busy = True

    def poll(self):
        try:
            for future in list(self.running):
                if not future.done():
                    continue
                generation, stage = self.running.pop(future)
                if generation != self.generation:
                    continue
                result = future.result()
                if stage == "prepare":
                    i, j = result
                    for start in range(0, len(i), 8):
                        self.jobs.append(
                            (
                                "solve",
                                solve_cells_chunk,
                                (
                                    self.args[0],
                                    self.args[1],
                                    i,
                                    j,
                                    start,
                                    start + 8,
                                    self.args[3],
                                    self.args[4],
                                ),
                            )
                        )
                    self.total = len(self.jobs)
                else:
                    pid, cells = result
                    self.process_ids.add(pid)
                    self.parts.extend(cells)
                    self.completed += 1
            while self.jobs and len(self.running) < self.workers:
                stage, function, args = self.jobs.popleft()
                future = self.executor.submit(function, *args)
                self.running[future] = (self.generation, stage)
            if self.busy and not self.jobs and not self.running:
                result = pareto_cells(self.parts)
                self.parts.clear()
                self.busy = False
                return result
        except Exception as error:
            self.cancel()
            self.error = str(error)
        return None

    def close(self):
        self.cancel()
        if self.owns_executor and self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = None


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
