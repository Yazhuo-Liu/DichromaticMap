"""Exact layer-preserving CSL and local same-layer atom matching."""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import gcd
import numpy as np
from .crystal import ProjectedGrain, get_geometry, congruence_kernel
from .cells import StrainedCell, bases, determinant, reduce_cell

DEFAULT_LOCAL_DISTANCE = 0.1  # pair separation in a0, not a strain percentage
COINCIDENCE_TOLERANCE_FACTOR = 1.0e-6


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
        return _point_distances(self.first, self.second)


def _point_distances(first, second):
    """Retain normal-range norms and avoid squared underflow in tiny vectors."""
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distances = np.linalg.norm(delta, axis=1)
    small = np.sqrt(np.finfo(float).tiny)
    if distances.min(initial=np.inf) < small or not np.isfinite(distances.max(initial=0)):
        unstable = ((distances < small) & np.any(delta != 0, axis=1)) | ~np.isfinite(distances)
        if np.any(unstable):
            distances[unstable] = np.hypot(delta[unstable, 0], delta[unstable, 1])
    return distances


def _spatial_cells(first, second, radius):
    """Use Float64 bins with rounding slack, without changing the radius.

    Bins are slightly wider than the physical matching radius. Their margin
    covers rounding in common-origin subtraction and division at bin borders;
    distance tests still use the caller's original radius. Reject spans whose
    floating resolution would require a margin larger than one eighth of it.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if any(points.ndim != 2 or points.shape[1] != 2
           or not np.all(np.isfinite(points)) for points in (first, second)):
        raise ValueError("Matching requires finite two-dimensional point arrays")
    origin = np.minimum(first.min(axis=0), second.min(axis=0))
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        relative_first = first - origin
        relative_second = second - origin
        first_scaled = relative_first / radius
        second_scaled = relative_second / radius
    limit = float(np.iinfo(np.int64).max)
    if any(not np.all(np.isfinite(cells)) or np.any(cells < 0)
           or np.any(cells >= limit) for cells in (first_scaled, second_scaled)):
        raise ValueError("Matching coordinate/tolerance range exceeds integer bin limits")
    extent = max(float(np.max(relative_first)), float(np.max(relative_second)))
    if extent == 0:
        factor = 1.0
    elif radius < np.finfo(float).tiny:
        # Subnormal coordinates lie on an integer grid of minimum Float64
        # steps. A two-radius bin also supports this range without squaring or
        # constructing a doubled subnormal radius in the original units.
        factor = 2.0
        largest_bin = max(float(np.max(first_scaled)), float(np.max(second_scaled)))
        if np.spacing(largest_bin) / 2 > 1 / 16:
            raise ValueError("Matching span cannot resolve the radius with Float64 precision")
    else:
        with np.errstate(under="ignore"):
            resolution = np.spacing(extent) / radius
        if resolution > 1 / 128:
            raise ValueError("Matching span cannot resolve the radius with Float64 precision")
        factor = 1.0 + 16 * resolution
    first_cells = np.floor(first_scaled / factor)
    second_cells = np.floor(second_scaled / factor)
    return first_cells.astype(np.int64), second_cells.astype(np.int64)


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
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    radius = float(radius)
    radius_squared = radius * radius
    min_normal = np.finfo(float).tiny
    norm_mode = not np.isfinite(radius_squared) or radius_squared < min_normal
    # Work relative to a common origin, avoiding packed-key wraparound on pan.
    first_cells, second_cells = _spatial_cells(first, second, radius)
    # Scalar integer searches are substantially faster than structured-array
    # comparisons. Pad the shared rectangle by one cell on every side, so
    # neighboring rows cannot alias even at the boundary. Keep the full-width
    # coordinate fallback when the rectangle would overflow an int64 key.
    maximum = np.maximum(first_cells.max(axis=0), second_cells.max(axis=0))
    stride = int(maximum[1]) + 3
    scalar_keys = (
        first_cells.min() >= 0
        and second_cells.min() >= 0
        and (int(maximum[0]) + 3) * stride - 1 <= np.iinfo(np.int64).max
    )
    if scalar_keys:
        def keys(cells):
            return cells[:, 0] * stride + cells[:, 1] + stride + 1
    else:
        dtype = np.dtype([("x", np.int64), ("y", np.int64)])

        def keys(cells):
            return np.ascontiguousarray(cells).view(dtype).reshape(-1)

    # Sort each bin geometrically as well as by cell address.
    order = np.lexsort(
        (second[:, 1], second[:, 0], second_cells[:, 1], second_cells[:, 0])
    )
    sorted_keys = keys(second_cells[order])
    bin_starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    bin_keys = sorted_keys[bin_starts]
    bin_counts = np.diff(np.r_[bin_starts, len(second)])
    first_keys = keys(first_cells) if scalar_keys else None
    coordinate_order = np.lexsort((second[:, 1], second[:, 0]))
    rank = np.empty(len(second), dtype=int)
    rank[coordinate_order] = np.arange(len(second))
    for start in range(0, len(first), 8192):
        stop = min(start + 8192, len(first))
        best = np.full(stop - start, np.inf)
        best_rank = np.full(stop - start, len(second), dtype=int)
        chosen = np.full(stop - start, -1, dtype=int)
        stable_norm = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                addresses = (
                    first_keys[start:stop] + dx * stride + dy
                    if scalar_keys
                    else keys(first_cells[start:stop] + (dx, dy))
                )
                locations = np.searchsorted(bin_keys, addresses)
                safe = np.minimum(locations, len(bin_keys) - 1)
                found = (locations < len(bin_keys)) & (bin_keys[safe] == addresses)
                rows = np.flatnonzero(found)
                starts = bin_starts[safe[rows]]
                counts = bin_counts[safe[rows]]
                for occupant in range(int(np.max(counts, initial=0))):
                    active = counts > occupant
                    active_rows = rows[active]
                    targets = order[starts[active] + occupant]
                    delta = first[start + active_rows] - second[targets]
                    if norm_mode:
                        metric = np.hypot(delta[:, 0], delta[:, 1])
                        threshold = radius
                    else:
                        metric = np.einsum("ij,ij->i", delta, delta)
                        if metric.min(initial=np.inf) < min_normal:
                            tiny = (metric < min_normal) & np.any(delta != 0, axis=1)
                            if np.any(tiny):
                                if stable_norm is None:
                                    stable_norm = np.zeros(stop - start, dtype=bool)
                                switch = tiny & ~stable_norm[active_rows]
                                best[active_rows[switch]] = np.sqrt(best[active_rows[switch]])
                                stable_norm[active_rows[switch]] = True
                        threshold = radius_squared
                        if stable_norm is not None:
                            use_norm = stable_norm[active_rows]
                            if np.any(use_norm):
                                metric[use_norm] = np.hypot(delta[use_norm, 0], delta[use_norm, 1])
                                threshold = np.where(use_norm, radius, radius_squared)
                    better = (metric <= threshold) & (
                        (metric < best[active_rows])
                        | (
                            (metric == best[active_rows])
                            & (rank[targets] < best_rank[active_rows])
                        )
                    )
                    chosen[active_rows[better]] = targets[better]
                    best[active_rows[better]] = metric[better]
                    best_rank[active_rows[better]] = rank[targets[better]]
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
    first_layers = grain1.layer_selections()
    second_layers = grain2.layer_selections()
    if layers is None:
        layers = sorted(first_layers.keys() | second_layers.keys())
    batches = []
    for layer in layers:
        first = grain1.positions[first_layers.get(layer, slice(0, 0))]
        second = grain2.positions[second_layers.get(layer, slice(0, 0))]
        if not len(first) or not len(second):
            continue
        forward = _nearest_in_radius(first, second, distance)
        reverse = _nearest_in_radius(second, first, distance)
        i = np.flatnonzero(forward >= 0)
        j = forward[i]
        mutual = reverse[j] == i
        i, j = i[mutual], j[mutual]
        keep = _point_distances(first[i], second[j]) > exact_tolerance
        batches.append(
            LocalPairs(
                first[i[keep]],
                second[j[keep]],
                np.full(np.count_nonzero(keep), layer, dtype=np.int16),
            )
        )
    return LocalPairs.concatenate(batches)


def exact_csl_cell(angle, max_denominator=128, lattice="FCC", axis="110"):
    """Exact, layer-preserving common cell; no strain search or atom matching.

    Recognize tan(theta/2)/sqrt(axis.axis) = n/m to 1e-9 degrees, with
    primitive m,n bounded by max_denominator. The raw angle may span [0,180]
    independently of the viewer's symmetry-reduced range. At 180 degrees use
    the exact quaternion (0, axis) without evaluating the tangent singularity.
    In the selected A-layer basis, B2^-1 B1 = P/d with integer P.
    Build the rational 3D quaternion rotation and express it in the computed
    planar lattice basis. A general integer-congruence kernel handles axes
    whose coefficients are not invertible modulo the denominator.
    This is primitive in the A-preserving plane, not necessarily in 3D.
    Noncommensurate/unrecognized angles return None, never a strained fit.
    Returned arrays are independent writable copies of the private cache.
    """
    geometry = get_geometry(lattice, axis)
    if not isinstance(max_denominator, (int, np.integer)) or max_denominator < 1:
        raise ValueError("max_denominator must be a positive integer")
    if not np.isfinite(angle) or not 0 <= angle <= 180:
        return None
    cached = _exact_csl_cell(float(angle), int(max_denominator),
                             geometry.lattice, geometry.axis)
    if cached is None:
        return None
    return StrainedCell(*(getattr(cached, field).copy()
                         for field in ("m1", "m2", "f1", "f2", "cell")),
                       cached.max_strain, cached.lattice, cached.axis)


@lru_cache(maxsize=256)
def _exact_csl_cell(angle, max_denominator, lattice, axis):
    """Cached exact calculation; callers never receive these private arrays."""
    geometry = get_geometry(lattice, axis)
    lattice, axis = geometry.lattice, geometry.axis
    norm_squared = geometry.axis_norm_squared
    if not np.isfinite(angle) or not 0 <= angle <= 180:
        return None
    if angle == 180:
        m, n = 0, 1
    else:
        ratio = Fraction(float(np.tan(np.deg2rad(angle / 2)) / np.sqrt(norm_squared)))
        ratio = ratio.limit_denominator(max_denominator)
        n, m = ratio.numerator, ratio.denominator
        # Below 90 degrees n <= m already held. Bound both coefficients now
        # that n/m can grow without limit near a half-turn; an enormous cell
        # should remain unrecognized rather than overflow its integer basis.
        if n > max_denominator:
            return None
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
    arrays = (m1, m2, np.eye(2), np.eye(2), cell)
    for array in arrays:
        array.setflags(write=False)
    return StrainedCell(*arrays, 0.0, lattice, axis)


# Preserve cache inspection/control used by Python clients of the public API.
exact_csl_cell.cache_clear = _exact_csl_cell.cache_clear
exact_csl_cell.cache_info = _exact_csl_cell.cache_info
exact_csl_cell.cache_parameters = _exact_csl_cell.cache_parameters


def same_layer_coincidence_sites(
    grain_1: ProjectedGrain,
    grain_2: ProjectedGrain,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    """Find coincidences separately in every computed axial phase."""

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("coincidence tolerance must be finite and positive")
    sites_by_layer: list[np.ndarray] = []
    with np.errstate(over="ignore", under="ignore"):
        tolerance_squared = np.float64(tolerance) ** 2

    first_layers = grain_1.layer_selections()
    second_layers = grain_2.layer_selections()
    count = max(
        getattr(grain_1, "layer_count", 2),
        getattr(grain_2, "layer_count", 2),
        max(first_layers, default=-1) + 1,
        max(second_layers, default=-1) + 1,
    )
    for layer in range(count):
        first = np.asarray(grain_1.positions[first_layers.get(layer, slice(0, 0))], dtype=np.float64)
        second = np.asarray(grain_2.positions[second_layers.get(layer, slice(0, 0))], dtype=np.float64)
        if len(first) == 0 or len(second) == 0:
            sites_by_layer.append(np.empty((0, 2)))
            continue

        first_cells, second_cells = _spatial_cells(first, second, tolerance)
        # Small relative rectangles keep the fast packed-key path. Larger
        # rectangles retain both full int64 coordinates, without 32-bit wrap.
        packed = all(np.max(cells) < np.iinfo(np.uint32).max
                     for cells in (first_cells, second_cells))
        if packed:
            def cell_keys(cells):
                unsigned = cells.astype(np.uint64)
                return (unsigned[:, 0] << np.uint64(32)) | unsigned[:, 1]
        else:
            dtype = np.dtype([("x", np.int64), ("y", np.int64)])

            def cell_keys(cells):
                return np.ascontiguousarray(cells).view(dtype).reshape(-1)

        second_keys = cell_keys(second_cells)
        order = np.argsort(second_keys, kind="stable")
        sorted_keys = second_keys[order]
        bin_starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
        bin_keys = sorted_keys[bin_starts]
        bin_counts = np.diff(np.r_[bin_starts, len(second)])
        matched_second = np.full(len(first), -1, dtype=np.int64)

        # A genuine match can lie in the same cell or one of eight neighbors.
        # Searching all first sites in a batch keeps large CSL views responsive.
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                unresolved = matched_second < 0
                if not np.any(unresolved):
                    break
                first_indices = np.flatnonzero(unresolved)
                neighbor_cells = first_cells[first_indices] + (delta_x, delta_y)
                neighbor_keys = cell_keys(neighbor_cells)
                locations = np.searchsorted(bin_keys, neighbor_keys)
                within = locations < len(bin_keys)
                safe_locations = np.minimum(locations, len(bin_keys) - 1)
                key_matches = within & (bin_keys[safe_locations] == neighbor_keys)
                if not np.any(key_matches):
                    continue
                trial_first = first_indices[key_matches]
                starts = bin_starts[safe_locations[key_matches]]
                counts = bin_counts[safe_locations[key_matches]]
                # Usually a tiny exact bin has one atom. When it has several,
                # every occupant must be considered before declaring no match.
                for occupant in range(int(np.max(counts))):
                    active = (counts > occupant) & (matched_second[trial_first] < 0)
                    if not np.any(active):
                        break
                    current_first = trial_first[active]
                    current_second = order[starts[active] + occupant]
                    delta = first[current_first] - second[current_second]
                    if np.isfinite(tolerance_squared) and tolerance_squared >= np.finfo(float).tiny:
                        with np.errstate(over="ignore", under="ignore"):
                            accepted = np.einsum("ij,ij->i", delta, delta) <= tolerance_squared
                    else:
                        accepted = np.hypot(delta[:, 0], delta[:, 1]) <= tolerance
                    matched_second[current_first[accepted]] = current_second[accepted]
            if not np.any(matched_second < 0):
                break

        matched_first = np.flatnonzero(matched_second >= 0)
        sites_by_layer.append(
            first[matched_first] / 2 + second[matched_second[matched_first]] / 2
        )
    return tuple(sites_by_layer)
