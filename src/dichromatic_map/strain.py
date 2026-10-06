"""Uniform cell search and selected-cell strain fitting, independent of Qt."""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import os
from threading import RLock
import numpy as np
from .crystal import get_geometry, rotation_matrix_2d
from .cells import StrainedCell, bases, determinant, reduce_cell, validate_cell_vertices

DEFAULT_STRAIN_PERCENT = 2.0
DEFAULT_SEARCH_INDEX = 12
MAX_VECTORS = 320
_CANDIDATE_CACHE_LIMIT = 32
_CELL_SEARCH_CACHE_LIMIT = 16
_candidate_cache = OrderedDict()
_cell_search_cache = OrderedDict()
_cache_lock = RLock()


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
    if not np.isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError("Reference angle must be in [0,180] degrees")
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


def _candidate_parameters(angle, percent, extent, lattice, axis):
    angle, percent = float(angle), float(percent)
    if (not np.isfinite(angle) or not np.isfinite(percent)
            or not 0 < percent <= 10 or not 2 <= extent <= 40 or int(extent) != extent):
        raise ValueError("finite angle, strain (0,10]% and search index 2..40 required")
    geometry = get_geometry(lattice, axis)
    return angle, percent, int(extent), geometry.lattice, geometry.axis


@lru_cache(maxsize=39)
def _integer_translation_grid(extent):
    integers = np.column_stack(
        (
            np.r_[
                np.zeros(extent, dtype=int),
                np.repeat(np.arange(1, extent + 1), 2 * extent + 1),
            ],
            np.r_[
                np.arange(1, extent + 1),
                np.tile(np.arange(-extent, extent + 1), extent),
            ],
        )
    )
    # Bytes-backed storage cannot be made writable by a preparation caller.
    return np.frombuffer(integers.tobytes(), dtype=integers.dtype).reshape(integers.shape)


def _cache_get(cache, key):
    with _cache_lock:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value


def _cache_put(cache, key, value, limit):
    with _cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)


class CandidatePreparation:
    """Bounded, resumable candidate enumeration, in historical search order.

    Construction does only the vectorized O(extent**2) setup. Each step visits
    a bounded number of integer offsets so a caller can yield/cancel between
    steps. Only completed preparations populate the bounded shared cache.
    """

    def __init__(self, angle, percent, extent, lattice="FCC", axis="110"):
        self.parameters = _candidate_parameters(angle, percent, extent, lattice, axis)
        self._result = _cache_get(_candidate_cache, self.parameters)
        self.completed_offsets = self.total_offsets = 0
        if self._result is not None:
            return
        angle, percent, self.extent, lattice, axis = self.parameters
        self.e = percent / 100
        b1, self.b2 = bases(angle, lattice, axis)
        self.integers = _integer_translation_grid(self.extent)
        self.v = self.integers @ b1.T
        self.lengths = np.linalg.norm(self.v, axis=1)
        inverse_b2 = np.linalg.inv(self.b2)
        coordinates = self.v @ inverse_b2.T
        self.nearest = np.rint(coordinates).astype(int)
        # Same necessary geometric bounds and slack as the synchronous search.
        inverse_norm = np.linalg.norm(inverse_b2, ord=2)
        self.radii = np.ceil(
            (2 * self.e / (1 - self.e)) * self.lengths * inverse_norm
        ).astype(int) + 1
        bound_e = self.e + 1e-12
        self.coordinate_bounds = (
            (2 * bound_e / (1 - bound_e)) * self.lengths[:, None]
            * np.linalg.norm(inverse_b2, axis=1) + 1e-10
        )
        self.displacement = self.nearest - coordinates
        self.radius = int(self.radii.max())
        self.side = 2 * self.radius + 1
        self.total_offsets = self.side * self.side
        self._dx = None
        self._use_x = np.empty(0, dtype=int)
        self._n1, self._n2, self._errors, self._sizes = [], [], [], []

    @property
    def done(self):
        return self._result is not None

    def _finish(self):
        if not self._n1:
            first = np.empty((0, 2), dtype=int)
            second = np.empty((0, 2), dtype=int)
        else:
            first, second = np.concatenate(self._n1), np.concatenate(self._n2)
            error, size = np.concatenate(self._errors), np.concatenate(self._sizes)
            chosen = np.unique(np.r_[
                np.argsort(size, kind="stable")[:MAX_VECTORS // 2],
                np.argsort(error, kind="stable")[:MAX_VECTORS // 2],
            ])
            first, second = first[chosen], second[chosen]
        first.setflags(write=False)
        second.setflags(write=False)
        self._result = first, second
        _cache_put(_candidate_cache, self.parameters, self._result, _CANDIDATE_CACHE_LIMIT)
        self._n1.clear()
        self._n2.clear()
        self._errors.clear()
        self._sizes.clear()

    def step(self, max_offsets=32):
        """Return candidate array copies on completion, otherwise None."""
        if int(max_offsets) != max_offsets or max_offsets < 1:
            raise ValueError("max_offsets must be a positive integer")
        if self.done:
            return tuple(array.copy() for array in self._result)
        stop = min(self.total_offsets, self.completed_offsets + int(max_offsets))
        while self.completed_offsets < stop:
            dx = self.completed_offsets // self.side - self.radius
            dy = self.completed_offsets % self.side - self.radius
            self.completed_offsets += 1
            if dx != self._dx:
                self._dx = dx
                # The x tests are identical for every dy. Filter once per dx,
                # then do y tests only on these survivors, preserving integer
                # order and the original radii >= max(abs(dx), abs(dy)) test.
                self._use_x = np.flatnonzero(
                    (self.radii >= abs(dx))
                    & (np.abs(self.nearest[:, 0] + dx) <= self.extent)
                    & (np.abs(self.displacement[:, 0] + dx)
                       <= self.coordinate_bounds[:, 0])
                )
            if not len(self._use_x):
                continue
            rows = self._use_x
            use = rows[np.flatnonzero(
                (self.radii[rows] >= abs(dy))
                & (np.abs(self.nearest[rows, 1] + dy) <= self.extent)
                & (np.abs(self.displacement[rows, 1] + dy)
                   <= self.coordinate_bounds[rows, 1])
            )]
            if not len(use):
                continue
            other = self.nearest[use] + (dx, dy)
            w = other @ self.b2.T
            l2 = np.linalg.norm(w, axis=1)
            size = self.lengths[use] + l2
            error = np.linalg.norm(self.v[use] - w, axis=1) / size
            mask = (error <= self.e + 1e-12) & (l2 > 0)
            if not np.any(mask):
                continue
            self._n1.append(self.integers[use[mask]])
            self._n2.append(other[mask])
            self._errors.append(error[mask])
            self._sizes.append(size[mask])
        if self.completed_offsets == self.total_offsets:
            self._finish()
            return tuple(array.copy() for array in self._result)
        return None


def candidate_vectors(angle, percent, extent, lattice="FCC", axis="110"):
    """Approximately compatible translations; bounded cache, fresh array copies."""
    preparation = CandidatePreparation(angle, percent, extent, lattice, axis)
    result = preparation.step(max(1, preparation.total_offsets))
    return result


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


def _cell_search_key(angle, percent, extent, lattice, axis, rows_per_chunk):
    if int(rows_per_chunk) != rows_per_chunk or rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be a positive integer")
    return (*_candidate_parameters(angle, percent, extent, lattice, axis),
            int(rows_per_chunk))


def _copy_search_cell(cell, readonly=False):
    arrays = [np.array(getattr(cell, field), copy=True)
              for field in ("m1", "m2", "f1", "f2", "cell")]
    if readonly:
        for array in arrays:
            array.setflags(write=False)
    return StrainedCell(*arrays, cell.max_strain, cell.lattice, cell.axis)


def get_cached_cell_search(
    angle, percent, extent, lattice="FCC", axis="110", *, rows_per_chunk=8
):
    """Return fresh result copies, or None; use in the coordinating process.

    Keys preserve the exact floating parameters and normalized crystal. The
    numerical chunk policy is included because roundoff may select a different
    equivalent representative under a different partition.
    """
    key = _cell_search_key(angle, percent, extent, lattice, axis, rows_per_chunk)
    cached = _cache_get(_cell_search_cache, key)
    return None if cached is None else [_copy_search_cell(cell) for cell in cached]


def cache_cell_search(
    cells, angle, percent, extent, lattice="FCC", axis="110", *, rows_per_chunk=8
):
    """Remember a complete successful Pareto result, with private array copies.

    Asynchronous callers must only publish a current, complete generation;
    cancelled/partial/failed work must not enter this cache. Empty successful
    results are cached too. Entries and result sizes are bounded.
    """
    key = _cell_search_key(angle, percent, extent, lattice, axis, rows_per_chunk)
    cells = tuple(cells)
    if len(cells) > 12:
        raise ValueError("Cache only a complete Pareto result of at most 12 cells")
    for cell in cells:
        if not isinstance(cell, StrainedCell):
            raise TypeError("Search cache requires StrainedCell results")
        for field in ("m1", "m2", "f1", "f2", "cell"):
            array = np.asarray(getattr(cell, field))
            if array.shape != (2, 2) or array.dtype.kind not in "biuf":
                raise ValueError("Search cache requires real numeric 2-by-2 cell arrays")
        if not np.isfinite(cell.max_strain):
            raise ValueError("Search cache requires a finite strain")
        geometry = get_geometry(cell.lattice, cell.axis)
        if (geometry.lattice, geometry.axis) != key[3:5]:
            raise ValueError("Search result crystal does not match the cache key")
    stored = tuple(_copy_search_cell(cell, readonly=True) for cell in cells)
    _cache_put(_cell_search_cache, key, stored, _CELL_SEARCH_CACHE_LIMIT)


def find_strained_cells(
    angle, percent, extent, lattice="FCC", axis="110", *, rows_per_chunk=8
):
    """Synchronous bounded search with the same cache used by async callers."""
    key = _cell_search_key(angle, percent, extent, lattice, axis, rows_per_chunk)
    cached = get_cached_cell_search(*key[:5], rows_per_chunk=key[5])
    if cached is not None:
        return cached
    first, second = candidate_vectors(*key[:5])
    parts = []
    for start in range(0, len(first), key[5]):
        parts.extend(solve_cells_chunk(
            key[0], key[1], first, second, start, start + key[5], key[3], key[4]
        )[1])
    result = pareto_cells(parts)
    cache_cell_search(result, *key[:5], rows_per_chunk=key[5])
    return result


def clear_strain_caches():
    """Release completed candidate/search entries and reusable integer grids."""
    with _cache_lock:
        _candidate_cache.clear()
        _cell_search_cache.clear()
        _integer_translation_grid.cache_clear()


def _least_norm_cell_solution(constraints, rhs):
    """Solve the 4-by-6 strain constraints without squaring their condition.

    For full row rank, C.T = Q R gives the minimum-norm solution
    Q solve(R.T, rhs). Householder QR is cheaper than a batched SVD. Only
    conservatively well-conditioned systems take that path; all others keep
    the original pinv cutoff, including exactly common/rank-deficient cells.
    """
    solution = np.zeros((len(constraints), 6))
    active = np.flatnonzero(np.any(rhs != 0, axis=1))
    if not len(active):
        return solution  # Zero is the exact minimum norm at any matrix rank.
    c, target = constraints[active], rhs[active]
    q, r = np.linalg.qr(c.transpose(0, 2, 1), mode="reduced")
    scale = np.linalg.norm(r, axis=(1, 2))
    diagonal = np.divide(
        np.diagonal(r, axis1=1, axis2=2), scale[:, None],
        out=np.zeros((len(r), 4)), where=scale[:, None] > 0,
    )
    # |det(R/||R||F)| is the product of its singular values. Every singular
    # value is <= 1, so this product is a lower bound on sigma_min/sigma_max.
    # The guard is far above pinv's 1e-11 rank cutoff and bounds cond(C) < 1e5.
    regular = np.abs(np.prod(diagonal, axis=1)) > 1e-5
    if np.any(regular):
        rows = np.flatnonzero(regular)
        try:
            values = (q[rows] @ np.linalg.solve(
                r[rows].transpose(0, 2, 1), target[rows, :, None]
            ))[..., 0]
        except np.linalg.LinAlgError:
            regular[:] = False
        else:
            # Reject an unexpectedly poor backward error rather than weakening
            # the physical residual/strain checks in solve_cells_chunk.
            residual = np.max(np.abs(
                (c[rows] @ values[..., None])[..., 0] - target[rows]
            ), axis=1)
            bound = 128 * np.finfo(float).eps * (
                scale[rows] * np.linalg.norm(values, axis=1)
                + np.linalg.norm(target[rows], axis=1)
            )
            accepted = residual <= bound
            solution[active[rows[accepted]]] = values[accepted]
            regular[rows[~accepted]] = False
    fallback = ~regular
    if np.any(fallback):
        solution[active[fallback]] = (
            np.linalg.pinv(c[fallback], rcond=1e-11)
            @ target[fallback, :, None]
        )[..., 0]
    return solution


def _symmetric_deformations(solution):
    f = np.tile(np.eye(2), (len(solution), 2, 1, 1))
    for g in (0, 1):
        f[:, g, 0, 0] += solution[:, 3 * g]
        f[:, g, 1, 1] += solution[:, 3 * g + 1]
        f[:, g, 0, 1] = f[:, g, 1, 0] = solution[:, 3 * g + 2] / np.sqrt(2)
    return f


def solve_cells_chunk(
    angle, percent, integers1, integers2, start, stop, lattice="FCC", axis="110"
):
    """Batched least-norm symmetric deformation solve for vector pairs."""
    geometry = get_geometry(lattice, axis)
    lattice, axis = geometry.lattice, geometry.axis
    b1, b2 = bases(angle, lattice, axis)
    # Build only this chunk's rows of the upper triangle. UI workers request
    # small batches; constructing the entire triangle repeats O(N²) work.
    n = len(integers1)
    start, stop = max(0, start), min(stop, n - 1)
    if start >= stop:
        return os.getpid(), []
    rows = np.arange(start, stop)
    counts = n - rows - 1
    first = np.repeat(rows, counts)
    second = np.concatenate([np.arange(row + 1, n) for row in rows])
    m1 = np.stack([integers1[first], integers1[second]], axis=2)
    m2 = np.stack([integers2[first], integers2[second]], axis=2)
    d1 = m1[:, 0, 0] * m1[:, 1, 1] - m1[:, 0, 1] * m1[:, 1, 0]
    d2 = m2[:, 0, 0] * m2[:, 1, 1] - m2[:, 0, 1] * m2[:, 1, 0]
    mask = (np.abs(d1) >= 0.5) & (d1 * d2 > 0)
    m1, m2 = m1[mask], m2[mask]
    if not len(m1):
        return os.getpid(), []
    a, b = b1 @ m1, b2 @ m2
    # Bounded symmetric strains obey ||a-b|| <= e (||a||+||b||) for
    # EVERY linear combination of the two cell edges. Check both diagonals
    # before the more expensive SVD, with slack for the final residual limit.
    compatible = np.ones(len(a), dtype=bool)
    for sign in (-1, 1):
        diagonal_a = a[:, :, 0] + sign * a[:, :, 1]
        diagonal_b = b[:, :, 0] + sign * b[:, :, 1]
        compatible &= np.linalg.norm(diagonal_a - diagonal_b, axis=1) <= (
            (percent / 100 + 1e-12)
            * (np.linalg.norm(diagonal_a, axis=1) + np.linalg.norm(diagonal_b, axis=1))
            + 3e-8
        )
    m1, m2, a, b = m1[compatible], m2[compatible], a[compatible], b[compatible]
    if not len(a):
        return os.getpid(), []
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
    sol = _least_norm_cell_solution(c, rhs)
    f = _symmetric_deformations(sol)
    eig = np.linalg.eigvalsh(f)
    strain = np.max(np.abs(eig - 1), axis=(1, 2))
    common = f[:, 0] @ a
    residual = np.max(np.abs(common - f[:, 1] @ b), axis=(1, 2))
    # Preserve the original SVD's representative for machine-precision ties
    # (e.g. grain-exchanged cells). Only minima at potentially nondominated
    # atom counts need refinement, not the thousands of dominated cells.
    margin = 1e-11  # Smaller than the 1e-10 Pareto improvement threshold.
    eligible = np.flatnonzero(
        (strain <= percent / 100 + margin)
        & (np.min(eig, axis=(1, 2)) > 0)
        & (residual < 1e-8)
    )
    if len(eligible):
        atom_counts = sum(
            np.abs(m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0])
            for m in (m1, m2)
        )
        _, groups = np.unique(atom_counts[eligible], return_inverse=True)
        minima = np.full(int(groups.max()) + 1, np.inf)
        np.minimum.at(minima, groups, strain[eligible])
        previous = np.r_[np.inf, np.minimum.accumulate(minima)[:-1]]
        competitive = minima < previous - 1e-10 + margin
        refine = eligible[
            competitive[groups] & (strain[eligible] <= minima[groups] + margin)
            & np.any(rhs[eligible] != 0, axis=1)
        ]
        if len(refine):
            refined = (np.linalg.pinv(c[refine], rcond=1e-11)
                       @ rhs[refine, :, None])[..., 0]
            f[refine] = _symmetric_deformations(refined)
            eig[refine] = np.linalg.eigvalsh(f[refine])
            strain[refine] = np.max(np.abs(eig[refine] - 1), axis=(1, 2))
            common[refine] = f[refine, 0] @ a[refine]
            residual[refine] = np.max(np.abs(
                common[refine] - f[refine, 1] @ b[refine]
            ), axis=(1, 2))
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
                f"G{grain + 1}",
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
