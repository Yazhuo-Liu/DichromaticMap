"""Exact lattice counts and explicit storage limits, independently of NumPy ints."""

import warnings

import numpy as np
import pytest

from dichromatic_map.cells import bases, checked_integer_array, determinant, reduce_cell
from dichromatic_map.completion import cell_completion_candidates
from dichromatic_map.strain import solve_cells_chunk, strain_selected_cell


def square_vertices(side):
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    return np.stack([uv @ (basis * side).T for basis in bases(0, "SC", "100")])


@pytest.mark.parametrize("matrix", [
    np.array([[3_100_000_000, 0], [0, 3_100_000_000]], dtype=np.int64),
    np.array([[0, 3_100_000_000], [3_100_000_000, 0]], dtype=np.int64),
    np.array([[2**63 + 1, 0], [0, 3]], dtype=np.uint64),
    np.array([[10**30, 7], [11, 10**30]], dtype=object),
])
def test_integer_determinant_matches_python_arithmetic_before_multiplication(matrix):
    a, b, c, d = (int(value) for value in matrix.flat)
    expected = a * d - b * c
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = determinant(matrix)
    assert type(actual) is int and actual == expected


def test_large_selected_cell_and_completion_keep_exact_python_atom_counts():
    side = 3_100_000_000
    vertices = square_vertices(side)
    expected = side * side  # Python integers, not np.linalg.det or np.int64.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fit = strain_selected_cell(vertices, 0, 2, "SC", "100", max_rotation_deg=0)
        candidates = cell_completion_candidates(
            vertices[:, :3], 0, lattice="SC", axis="100", max_rotation_deg=0
        )
    assert fit.cell.atoms == (expected, expected)
    assert str(expected) in fit.cell.label()
    np.testing.assert_array_equal(fit.cell.f1, np.eye(2))
    np.testing.assert_array_equal(fit.cell.f2, np.eye(2))
    assert len(candidates) == 1 and candidates[0].fit is not None
    assert candidates[0].atoms == (expected, expected)


@pytest.mark.parametrize("side", [60_000, 3_100_000_000])
def test_public_chunk_determinant_and_orientation_never_use_overflowed_products(side):
    # Even before determinant itself overflows, multiplying two valid positive
    # determinants can wrap negative and incorrectly reject their orientation.
    vectors = np.array([[side, 0], [0, side]], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, cells = solve_cells_chunk(0, 2, vectors, vectors, 0, 2, "SC", "100")
    assert len(cells) == 1
    assert cells[0].atoms == (side * side, side * side)


@pytest.mark.parametrize("side", [2**63, 2**64])
def test_integer_vertex_storage_limit_is_explicit_before_any_unsafe_cast(side):
    vertices = square_vertices(side)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="integer.*range|range.*integer"):
            strain_selected_cell(vertices, 0, 2, "SC", "100", max_rotation_deg=0)
        with pytest.raises(ValueError, match="integer.*range|range.*integer"):
            cell_completion_candidates(
                vertices[:, :3], 0, lattice="SC", axis="100", max_rotation_deg=0
            )


def test_completion_checks_exact_intermediate_closure_before_storing_int64():
    # All three input positions fit int64; C1+C3-C2 does not. The exact
    # parallelogram has C4=[12e18,4e18], not a wrapped negative coordinate.
    points = np.array([[4e18, 0], [-4e18, 0], [4e18, 4e18]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="integer.*range|range.*integer"):
            cell_completion_candidates(
                np.stack((points, points)), 0, lattice="SC", axis="100", max_rotation_deg=0
            )


@pytest.mark.parametrize("common_dtype", [np.float64, np.uint64])
def test_unsigned_cell_reduction_preserves_exact_shared_column_operations(common_dtype):
    size = 10**10
    edges = np.array([[size, size-1], [size+1, size]], dtype=np.uint64)
    first, second, common = reduce_cell(edges, edges, edges.astype(common_dtype))
    assert first.dtype == second.dtype == np.dtype(np.int64)
    assert abs(determinant(first)) == abs(determinant(second)) == 1
    np.testing.assert_array_equal(first, common)
    np.testing.assert_array_equal(second, common)


def test_cell_reduction_checks_integer_range_before_column_updates():
    # A valid positive diagonal F (10% strain), with representable original
    # integer columns. Its reduced second column would exceed int64.
    large = 8_000_000_000_000_000_000
    edges = np.array([[large, 1], [-large, 1]], dtype=np.int64)
    common = np.diag([.9, 1.1]) @ edges
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="integer.*range|range.*integer"):
            reduce_cell(edges, edges, common)


@pytest.mark.parametrize("values", [
    [True, 1], [[1, 0], [False, 1]], [np.bool_(True), np.int64(1)],
])
def test_integer_storage_keeps_raw_boolean_types_until_validation(values):
    with pytest.raises(ValueError, match="integers"):
        checked_integer_array(values)
