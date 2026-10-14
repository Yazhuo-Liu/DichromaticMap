"""Float64 affine evaluation shared by saved fits and atom regeneration.

Compensation retains the existing x' = F x + t model. It does not move atoms
to a common average, introduce an origin parameter, or rely on longdouble
(whose precision differs between native NumPy and WebAssembly).
"""

from __future__ import annotations

import numpy as np


# A conservative switch for bulk generation: at this operation magnitude,
# several rounding errors are still far below the 1e-8 alignment tolerance.
# Small ordinary views retain their existing, cheaper vectorized path.
AFFINE_COMPENSATION_THRESHOLD = float(2**18)


def affine_needs_compensation(point_bound, matrix, translation=None):
    """Whether a conservative input magnitude warrants bulk compensation.

    point_bound is an upper bound on absolute point components (a scalar or
    array of bounds). Use original reference coordinates, not the final small
    viewport center, when large F*x and t can cancel.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    translation = np.zeros(2) if translation is None else np.asarray(translation, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        bound = (float(np.max(np.abs(point_bound)))
                 * float(np.max(np.sum(np.abs(matrix), axis=-1)))
                 + float(np.max(np.abs(translation))))
    return not np.isfinite(bound) or bound > AFFINE_COMPENSATION_THRESHOLD


def _two_sum(first, second):
    total = first + second
    virtual_second = total - first
    error = (first - (total - virtual_second)) + (second - virtual_second)
    return total, error


def _split(value):
    # Dekker's balanced 26/27-bit split, rather than truncating mantissa bits.
    splitter_product = (2**27 + 1) * value
    high = splitter_product - (splitter_product - value)
    return high, value - high


def _two_product(first, second):
    product = first * second
    # Keep the splitter finite for extreme but valid Float64 inputs. Scaling
    # one operand down and the other up by an exact power of two preserves
    # their product. Normal crystal/F inputs take neither rare branch.
    split_limit = float(2**996)
    large_first = np.abs(first) > split_limit
    if np.any(large_first):
        second = np.where(large_first, second * 2**28, second)
        first = np.where(large_first, first * 2.0**-28, first)
    large_second = np.abs(second) > split_limit
    if np.any(large_second):
        first = np.where(large_second, first * 2**28, first)
        second = np.where(large_second, second * 2.0**-28, second)
    first_high, first_low = _split(first)
    second_high, second_low = _split(second)
    error1 = product - first_high * second_high
    error2 = error1 - first_low * second_high
    error3 = error2 - first_high * second_low
    return product, first_low * second_low - error3


def affine_points(points, matrix, translation=None):
    """Evaluate (...,2) points under a (2,2) F and (2,) t in Float64.

    Error-free product/sum residuals compensate the two-term dot and uniform
    translation, avoiding the large-origin cancellation of a plain F*x+t.
    Returned coordinates are individually evaluated actual grain endpoints.
    Unsupported overflowing intermediate arithmetic raises a range error.
    """
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    translation = np.zeros(2) if translation is None else np.asarray(translation, dtype=np.float64)
    if (points.ndim < 1 or points.shape[-1] != 2 or matrix.shape != (2, 2)
            or translation.shape != (2,)):
        raise ValueError("Provide (...,2) points, a 2-by-2 matrix and a two-component translation")
    if not all(np.all(np.isfinite(value)) for value in (points, matrix, translation)):
        raise ValueError("Affine inputs must be finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        first, first_error = _two_product(points[..., 0, None], matrix[:, 0])
        second, second_error = _two_product(points[..., 1, None], matrix[:, 1])
        high, low = _two_sum(first, second)
        high, low = _two_sum(high, low + first_error + second_error)
        high, shift_error = _two_sum(high, translation)
        result = high + (low + shift_error)
    if not np.all(np.isfinite(result)):
        raise ValueError("Affine evaluation exceeds the supported numerical range")
    return result
