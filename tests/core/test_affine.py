"""Decimal oracle for the actual saved Float64 affine inputs."""

from decimal import Decimal, localcontext
import warnings

import numpy as np
import pytest

from dichromatic_map.affine import affine_needs_compensation, affine_points


def exact_affine(points, matrix, shift):
    with localcontext() as context:
        # Largest/subnormal binary64 values can have over 1000 exact decimal
        # digits. Keep the oracle exact even before a huge cancellation.
        context.prec = 2000
        decimal = lambda value: Decimal.from_float(float(value))
        return np.array([
            [float(sum(decimal(matrix[row, column]) * decimal(point[column])
                       for column in (0, 1)) + decimal(shift[row])) for row in (0, 1)]
            for point in np.asarray(points).reshape(-1, 2)
        ]).reshape(np.asarray(points).shape)


@pytest.mark.parametrize("scale", [1, 60_000_000, 1e16, 1e100])
def test_compensated_affine_matches_independent_decimal_oracle(scale):
    rng = np.random.default_rng(1234)
    points = rng.normal(size=(30, 2)) * scale
    matrix = rng.normal(size=(2, 2))
    shift = rng.normal(size=2) * scale
    expected = exact_affine(points, matrix, shift)
    np.testing.assert_array_equal(affine_points(points, matrix, shift), expected)
    np.testing.assert_array_equal(affine_points(points.reshape(2, 15, 2), matrix, shift),
                                  expected.reshape(2, 15, 2))


def test_compensated_affine_handles_cancellation_and_extreme_finite_products():
    cases = [
        (np.array([[1e308, 1e308]]), np.eye(2), np.array([-1e308, -1e308])),
        (np.array([[1e-308, np.nextafter(0., 1.)]]), np.eye(2) * 1e308, np.zeros(2)),
        (np.array([[60_000_049., 60_000_000.]]), np.eye(2) * 1.020000000000615,
         np.array([-1199999.9899615918, -1199999.9899615918])),
    ]
    for points, matrix, shift in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = affine_points(points, matrix, shift)
        np.testing.assert_array_equal(result, exact_affine(points, matrix, shift))


def test_affine_empty_inputs_and_compensation_trigger_keep_the_bulk_contract():
    assert affine_points(np.empty((0, 2)), np.eye(2)).shape == (0, 2)
    assert not affine_needs_compensation(100, np.eye(2), [0, 0])
    assert affine_needs_compensation(60_000_000, np.eye(2), [-60_000_000, 0])
    assert affine_needs_compensation(100, np.eye(2) * 1e6, [0, 0])


@pytest.mark.parametrize("points,matrix,shift", [
    ([[np.nan, 0]], np.eye(2), [0, 0]),
    ([[1, 2]], [[np.inf, 0], [0, 1]], [0, 0]),
    ([[1, 2]], np.eye(2), [0, np.inf]),
    ([[2, 2]], np.eye(2) * 1e308, [0, 0]),
    ([[1, 2, 3]], np.eye(2), [0, 0]),
])
def test_affine_rejects_nonfinite_or_unsupported_arithmetic(points, matrix, shift):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError):
            affine_points(points, matrix, shift)
