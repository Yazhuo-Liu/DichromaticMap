"""Exact match occupancy, coordinate range and cache isolation regressions."""

from dataclasses import replace
import numpy as np
import pytest
from dichromatic_map import crystal, matching, strain
from dichromatic_map.state import PatternParameters


def grain(points):
    positions = np.asarray(points, dtype=float).reshape(-1, 2)
    return crystal.ProjectedGrain(positions, np.zeros(len(positions), dtype=np.int16),
                                  np.zeros((len(positions), 3), dtype=int), 1)


@pytest.mark.parametrize("field", ("width", "height", "lattice_constant", "marker_size"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_pattern_parameters_reject_nonfinite_positive_fields(field, value):
    with pytest.raises(ValueError, match="finite"):
        PatternParameters(**{field: value}).validate()


@pytest.mark.parametrize("tolerance", (np.nan, np.inf, -np.inf, 0, -1))
def test_exact_tolerance_is_finite_and_positive_even_for_empty_inputs(tolerance):
    with pytest.raises(ValueError, match="finite and positive"):
        matching.same_layer_coincidence_sites(grain([]), grain([]), tolerance)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_matching_rejects_nonfinite_positions(value):
    first, second = grain([[value, 0]]), grain([[0, 0]])
    with pytest.raises(ValueError, match="finite"):
        matching.same_layer_coincidence_sites(first, second, 1e-6)
    with pytest.raises(ValueError, match="finite"):
        matching.local_near_pairs(first, second)


def test_exact_matching_checks_every_occupant_in_one_bin():
    first = grain([[0.99e-6, 0.99e-6]])
    candidates = np.array([[0, 0], [0.99e-6, 0.99e-6]])
    for points in (candidates, candidates[::-1]):
        sites = matching.same_layer_coincidence_sites(first, grain(points), 1e-6)[0]
        np.testing.assert_array_equal(sites, first.positions)


def test_exact_matching_preserves_full_coordinates_beyond_32_bits():
    first = grain([[0, 0]])
    candidates = np.array([[2**32 * 1e-6, 0], [0, 0]])
    for points in (candidates, candidates[::-1]):
        sites = matching.same_layer_coincidence_sites(first, grain(points), 1e-6)[0]
        np.testing.assert_array_equal(sites, [[0, 0]])


def test_large_common_pan_does_not_overflow_absolute_bin_addresses():
    points = np.array([[2**40, -2**40], [2**40 + 0.125, -2**40 + 0.125]])
    sites = matching.same_layer_coincidence_sites(grain(points), grain(points[::-1]), 1e-6)[0]
    np.testing.assert_array_equal(sites, points)


def test_matching_rejects_unrepresentable_relative_bin_addresses():
    with pytest.raises(ValueError, match="integer bin limits"):
        matching.same_layer_coincidence_sites(grain([[0, 0]]), grain([[1, 0]]), 1e-20)


@pytest.mark.parametrize("tolerance", (1e-160, 1e160))
def test_exact_distance_comparison_handles_squared_tolerance_range(tolerance):
    first = grain([[0, 0]])
    second = grain([[1.1 * tolerance, 0]])
    assert len(matching.same_layer_coincidence_sites(first, second, tolerance)[0]) == 0


def test_exact_matching_with_multiple_occupants_matches_distance_matrix_oracle():
    rng = np.random.default_rng(812)
    first, second = grain(rng.uniform(-1, 1, (50, 2))), grain(rng.uniform(-1, 1, (60, 2)))
    tolerance = 0.2
    distances = np.linalg.norm(first.positions[:, None] - second.positions[None], axis=2)
    expected_first = first.positions[np.any(distances <= tolerance, axis=1)]
    sites = matching.same_layer_coincidence_sites(first, second, tolerance)[0]
    assert len(sites) == len(expected_first)
    for point, midpoint in zip(expected_first, sites):
        endpoint = 2 * midpoint - point
        assert np.linalg.norm(endpoint - point) <= tolerance + 1e-14
        assert np.min(np.linalg.norm(second.positions - endpoint, axis=1)) < 1e-14


def test_exact_cell_cache_does_not_share_mutable_arrays_with_callers():
    matching.exact_csl_cell.cache_clear()
    original = matching.exact_csl_cell(0)
    expected = {field: getattr(original, field).copy()
                for field in ("m1", "m2", "f1", "f2", "cell")}
    for field in expected:
        getattr(original, field)[:] = 123
    restored = matching.exact_csl_cell(0)
    for field, array in expected.items():
        np.testing.assert_array_equal(getattr(restored, field), array)
        assert not np.shares_memory(getattr(original, field), getattr(restored, field))
    # Normalize crystal aliases before caching, including iterable axes that
    # the geometry API accepts but an outer lru_cache could not hash.
    equivalent = matching.exact_csl_cell(0, lattice="fcc", axis=[1, 1, 0])
    np.testing.assert_array_equal(equivalent.cell, restored.cell)
    assert matching.exact_csl_cell.cache_info().misses == 1
    assert matching.exact_csl_cell.cache_info().hits == 2


@pytest.mark.parametrize("field", ("m1", "m2", "f1", "f2", "cell"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_search_cache_rejects_nonfinite_matrix_entries(field, value):
    cell = matching.exact_csl_cell(0)
    matrix = getattr(cell, field).astype(float)
    matrix[0, 0] = value
    invalid = replace(cell, **{field: matrix})
    with pytest.raises(ValueError, match="finite"):
        strain.cache_cell_search([invalid], 0, 2, 12)
