"""Ordered whole-grain and rectangular atom translations in analysis x,y / a0.

Selections are evaluated before each operation, on the positions produced by
the preceding operations. Reference atom identities and axial phases never
change. This module deliberately has no GUI or crystal geometry dependency.
"""

from collections.abc import Mapping

import numpy as np


MAX_DISPLACEMENT_OPERATIONS = 64
MAX_DISPLACEMENT_SOURCE_RECTANGLES = 512
MAX_DISPLACEMENT_COORDINATE = 1e8


class DisplacementLimitError(ValueError):
    """A displacement history exceeds the supported interactive work budget."""


def _finite_vector(values, length, name):
    try:
        raw = np.asarray(values)
        if (raw.shape != (length,) or raw.dtype.kind not in "iuf"
                or any(isinstance(value, (bool, np.bool_)) for value in values)):
            raise ValueError
        result = raw.astype(float)
        if not np.all(np.isfinite(result)):
            raise ValueError
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Translation {name} must contain {length} finite numeric coordinates / a0") from error
    if np.any(np.abs(result) > MAX_DISPLACEMENT_COORDINATE):
        raise ValueError(f"Translation {name} coordinates must be within ±{MAX_DISPLACEMENT_COORDINATE:g} a0")
    return result


def normalize_displacements(operations=None):
    """Validate operations and return independent JSON-safe dictionaries.

    Each operation has ``delta=[dx,dy]`` and ``bounds=None`` for a whole-grain
    shift or ``bounds=[xmin,xmax,ymin,ymax]`` for a closed rectangular region.
    Zero displacement is valid, so an imported history keeps its operation
    numbering. A malformed history is rejected rather than partially applied.
    """
    if operations is None:
        return []
    if not isinstance(operations, (list, tuple)):
        raise ValueError("Displacements must be a list of translation operations")
    if len(operations) > MAX_DISPLACEMENT_OPERATIONS:
        raise DisplacementLimitError(
            f"At most {MAX_DISPLACEMENT_OPERATIONS} translation operations are supported per grain"
        )
    result = []
    for operation in operations:
        if not isinstance(operation, Mapping) or set(operation) - {"delta", "bounds"}:
            raise ValueError("Each translation must contain delta and optional bounds")
        delta = _finite_vector(operation.get("delta"), 2, "delta")
        raw_bounds = operation.get("bounds")
        bounds = None if raw_bounds is None else _finite_vector(raw_bounds, 4, "bounds")
        if bounds is not None:
            if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
                raise ValueError("Translation bounds must satisfy xmin < xmax and ymin < ymax")
        result.append({"delta": delta.tolist(), "bounds": None if bounds is None else bounds.tolist()})
    return result


def normalize_grain_displacements(histories=None):
    """Validate and independently copy the two grains' operation histories."""
    if histories is None:
        return [[], []]
    if not isinstance(histories, (list, tuple)) or len(histories) != 2:
        raise ValueError("Provide translation histories for both grains")
    return [normalize_displacements(operations) for operations in histories]


def _region_bounds(bounds):
    """Keep closed-edge membership stable under equivalent Float64 transforms.

    Bulk lattice projection and independent reference-index reconstruction
    differ by a few rounding bits. A boundary selected through either path
    must retain the same atom. The margin follows each coordinate's ULP, not
    region size, so large rectangles do not swallow nearby outside atoms.
    """
    bounds = np.asarray(bounds, dtype=float)
    margin = 8 * np.spacing(np.maximum(np.abs(bounds), 1.0))
    return bounds[[0, 2]] - margin[[0, 2]], bounds[[1, 3]] + margin[[1, 3]]


def apply_displacements(points, operations=None):
    """Return translated points without mutating points or operation history."""
    result = np.asarray(points, dtype=float).copy()
    if result.ndim != 2 or result.shape[1] != 2 or not np.all(np.isfinite(result)):
        raise ValueError("Translation points must be a finite N-by-2 array")
    for operation in normalize_displacements(operations):
        bounds = operation["bounds"]
        if bounds is None:
            selected = slice(None)
        else:
            lower, upper = _region_bounds(bounds)
            selected = (
                (result[:, 0] >= lower[0]) & (result[:, 0] <= upper[0])
                & (result[:, 1] >= lower[1]) & (result[:, 1] <= upper[1])
            )
        with np.errstate(over="ignore", invalid="ignore"):
            result[selected] += operation["delta"]
        if not np.all(np.isfinite(result)):
            raise ValueError("Translation exceeds the supported numerical coordinate range")
    return result


def _intersection(low, high, other_low, other_high):
    lower, upper = np.maximum(low, other_low), np.minimum(high, other_high)
    return None if np.any(lower > upper) else (lower, upper)


def _outside_rectangles(low, high, bounds_low, bounds_high):
    """Cover a rectangle outside the selection, conservatively sharing edges."""
    overlap = _intersection(low, high, bounds_low, bounds_high)
    if overlap is None:
        return [(low, high)]
    inside_low, inside_high = overlap
    parts = []
    if low[0] < inside_low[0]:
        parts.append((low, np.array([inside_low[0], high[1]])))
    if high[0] > inside_high[0]:
        parts.append((np.array([inside_high[0], low[1]]), high))
    if low[1] < inside_low[1]:
        parts.append((np.array([inside_low[0], low[1]]),
                      np.array([inside_high[0], inside_low[1]])))
    if high[1] > inside_high[1]:
        parts.append((np.array([inside_low[0], inside_high[1]]),
                      np.array([inside_high[0], high[1]])))
    return parts


def displacement_source_rectangles(low, high, operations=None):
    """Return conservative source rectangles whose translated atoms enter a crop.

    Invert the history in reverse order: unchanged points come from outside
    each selection; moved points come from the crop shifted by minus delta,
    intersected with the selection. Separated sources stay separate, so moving
    a small region a long distance never allocates the space between regions.
    Shared boundary points are intentionally allowed; projection deduplicates
    by reference atom identity and then crops the final translated positions.
    """
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if (low.shape != (2,) or high.shape != (2,)
            or not np.all(np.isfinite([low, high])) or np.any(low > high)):
        raise ValueError("Translation crop must contain ordered finite x,y bounds")
    rectangles = [(low.copy(), high.copy())]
    for operation in reversed(normalize_displacements(operations)):
        delta = np.asarray(operation["delta"])
        if not np.any(delta):
            continue
        bounds = operation["bounds"]
        parts = []
        for lower, upper in rectangles:
            with np.errstate(over="ignore", invalid="ignore"):
                shifted = (lower - delta, upper - delta)
            if not np.all(np.isfinite(shifted)):
                raise DisplacementLimitError("Translation exceeds the supported numerical coordinate range")
            if bounds is None:
                parts.append(shifted)
                continue
            bounds_low, bounds_high = _region_bounds(bounds)
            parts.extend(_outside_rectangles(lower, upper, bounds_low, bounds_high))
            moved = _intersection(*shifted, bounds_low, bounds_high)
            if moved is not None:
                parts.append(moved)
        # Adjacent rectangles may share boundaries; identical rectangles do
        # not need another lattice mesh. Avoid an unbounded inverse history.
        unique = {tuple(np.r_[lower, upper]): (lower, upper) for lower, upper in parts}
        if len(unique) > MAX_DISPLACEMENT_SOURCE_RECTANGLES:
            raise DisplacementLimitError(
                "Translation selections require too many source regions; undo some translations or zoom in"
            )
        rectangles = list(unique.values())
    return tuple(rectangles)
