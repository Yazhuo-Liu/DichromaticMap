"""A nonfinite point cannot silently mean an empty physical cell."""

import warnings

import numpy as np
import pytest

from dichromatic_map.cells import cell_membership


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_membership_rejects_nonfinite_points_before_geometric_arithmetic(invalid):
    polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="finite"):
            cell_membership(np.array([[1, 1], [invalid, 0]]), polygon)


def test_membership_valid_grid_and_empty_input_keep_physical_masks():
    polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([[1, 1], [0, 0], [2, 1], [3, 1]])
    interior, boundary, half_open = cell_membership(points, polygon)
    np.testing.assert_array_equal(interior, [True, False, False, False])
    np.testing.assert_array_equal(boundary, [False, True, True, False])
    np.testing.assert_array_equal(half_open, [True, True, False, False])
    assert all(mask.shape == (0,) for mask in cell_membership(np.empty((0, 2)), polygon))
