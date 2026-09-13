"""Exact layer-preserving CSL and local same-layer atom matching."""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import gcd
import numpy as np
from .crystal import ProjectedGrain, get_geometry, congruence_kernel
from .cells import StrainedCell, bases, determinant, reduce_cell

DEFAULT_LOCAL_DISTANCE = 0.05  # pair separation in a0, not a strain percentage
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
                    squared = np.einsum("ij,ij->i", delta, delta)
                    better = (squared <= radius * radius) & (
                        (squared < best[active_rows])
                        | (
                            (squared == best[active_rows])
                            & (rank[targets] < best_rank[active_rows])
                        )
                    )
                    chosen[active_rows[better]] = targets[better]
                    best[active_rows[better]] = squared[better]
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


def same_layer_coincidence_sites(
    grain_1: ProjectedGrain,
    grain_2: ProjectedGrain,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    """Find coincidences separately in every computed axial phase."""

    if tolerance <= 0.0:
        raise ValueError("coincidence tolerance must be positive")
    sites_by_layer: list[np.ndarray] = []
    tolerance_squared = tolerance * tolerance

    def packed_cell_keys(cells: np.ndarray) -> np.ndarray:
        """Pack two signed 32-bit cell coordinates into one sortable key."""

        mask = np.uint64(0xFFFFFFFF)
        x_bits = cells[:, 0].astype(np.uint64) & mask
        y_bits = cells[:, 1].astype(np.uint64) & mask
        return (x_bits << np.uint64(32)) | y_bits

    count = max(
        getattr(grain_1, "layer_count", 2),
        getattr(grain_2, "layer_count", 2),
        int(np.max(grain_1.layers, initial=-1)) + 1,
        int(np.max(grain_2.layers, initial=-1)) + 1,
    )
    for layer in range(count):
        first = grain_1.positions[grain_1.layers == layer]
        second = grain_2.positions[grain_2.layers == layer]
        if len(first) == 0 or len(second) == 0:
            sites_by_layer.append(np.empty((0, 2)))
            continue

        second_cells = np.floor(second / tolerance).astype(np.int64)
        second_keys = packed_cell_keys(second_cells)
        order = np.argsort(second_keys)
        sorted_keys = second_keys[order]
        first_cells = np.floor(first / tolerance).astype(np.int64)
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
                neighbor_keys = packed_cell_keys(neighbor_cells)
                locations = np.searchsorted(sorted_keys, neighbor_keys)
                within = locations < len(sorted_keys)
                safe_locations = np.minimum(locations, len(sorted_keys) - 1)
                key_matches = within & (sorted_keys[safe_locations] == neighbor_keys)
                if not np.any(key_matches):
                    continue
                trial_first = first_indices[key_matches]
                trial_second = order[safe_locations[key_matches]]
                distances_squared = np.sum(
                    (first[trial_first] - second[trial_second]) ** 2, axis=1
                )
                accepted = distances_squared <= tolerance_squared
                matched_second[trial_first[accepted]] = trial_second[accepted]
            if not np.any(matched_second < 0):
                break

        matched_first = np.flatnonzero(matched_second >= 0)
        sites_by_layer.append(
            0.5 * (first[matched_first] + second[matched_second[matched_first]])
        )
    return tuple(sites_by_layer)
