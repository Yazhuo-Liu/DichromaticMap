"""Matching uses stored point values without losing radius resolution."""

import numpy as np
import pytest
from dichromatic_map import crystal, matching


def grain(points, dtype=np.float64):
    points = np.asarray(points, dtype=dtype).reshape(-1, 2)
    return crystal.ProjectedGrain(points, np.zeros(len(points), dtype=np.int16),
                                  np.zeros((len(points), 3), dtype=int), 1)


def test_float32_points_keep_normal_span_exact_matches():
    first = grain([[-32, 0], [1.8e-6, 0]], dtype=np.float32)
    second = grain([[2.7e-6, 0]], dtype=np.float32)
    radius = 1e-6
    # The oracle uses the actual stored Float32 values promoted to Float64;
    # it does not replace them with the decimal values used to construct them.
    distance = np.hypot(*(first.positions[1].astype(float) - second.positions[0]))
    assert 0 < distance < radius
    sites = matching.same_layer_coincidence_sites(first, second, radius)[0]
    assert len(sites) == 1
    np.testing.assert_allclose(sites[0],
        (first.positions[1].astype(float) + second.positions[0]) / 2, rtol=0, atol=1e-13)


def test_float64_radius_boundary_survives_relative_bin_rounding():
    first = grain([[-32, 0], [-2e-15, 0]])
    second = grain([[.1 - 2e-15, 0]])
    assert np.hypot(*(first.positions[1] - second.positions[0])) == .1
    sites = matching.same_layer_coincidence_sites(first, second, .1)[0]
    pairs = matching.local_near_pairs(first, second, .1, exact_tolerance=0)
    assert len(sites) == 1
    np.testing.assert_array_equal(pairs.first, first.positions[1:])
    np.testing.assert_array_equal(pairs.second, second.positions)


@pytest.mark.parametrize("local", (False, True))
def test_unresolvable_span_is_an_explicit_precision_error(local):
    if local:
        first, second = grain([[-2**53, 0], [.95, 0]]), grain([[1.04, 0]])
        radius = .1
    else:
        first = grain([[-2**33, 0], [.49 * 2**-20, 0]])
        second = grain([[.49 * 2**-20 + .9e-6, 0]])
        radius = 1e-6
    distance = np.hypot(*(first.positions[1] - second.positions[0]))
    assert 0 < distance < radius
    with pytest.raises(ValueError, match="precision"):
        if local:
            matching.local_near_pairs(first, second, radius)
        else:
            matching.same_layer_coincidence_sites(first, second, radius)


def test_tiny_local_pairs_and_distance_readout_do_not_underflow():
    first, second = grain([[0, 0]]), grain([[5e-201, 0]])
    radius = 1e-200
    oracle = np.hypot(*(first.positions[0] - second.positions[0]))
    assert 0 < oracle < radius
    pairs = matching.local_near_pairs(first, second, radius, exact_tolerance=0)
    np.testing.assert_array_equal(pairs.first, first.positions)
    np.testing.assert_array_equal(pairs.second, second.positions)
    assert pairs.distances[0] == oracle
    assert len(matching.local_near_pairs(first, second, radius,
                                       exact_tolerance=oracle).layers) == 0


def test_tiny_radius_still_rejects_points_outside_the_physical_limit():
    first, second = grain([[0, 0]]), grain([[1.1e-200, 0]])
    radius = 1e-200
    assert np.hypot(*(first.positions[0] - second.positions[0])) > radius
    np.testing.assert_array_equal(matching._nearest_in_radius(
        first.positions, second.positions, radius), [-1])
    assert len(matching.local_near_pairs(first, second, radius,
                                       exact_tolerance=0).layers) == 0


def test_tiny_radius_ties_remain_lexicographic_and_order_independent():
    first = grain([[0, 0]])
    endpoints = np.array([[5e-201, 0], [-5e-201, 0]])
    for positions in (endpoints, endpoints[::-1]):
        pairs = matching.local_near_pairs(first, grain(positions), 1e-200,
                                         exact_tolerance=0)
        np.testing.assert_array_equal(pairs.second, [[-5e-201, 0]])
        assert pairs.distances[0] == 5e-201


def test_tiny_distances_choose_the_nearest_endpoint_even_with_a_normal_radius():
    first = grain([[0, 0]])
    second = grain([[-5e-201, 0], [2e-201, 0], [.2, 0]])
    pairs = matching.local_near_pairs(first, second, .1, exact_tolerance=0)
    np.testing.assert_array_equal(pairs.second, [[2e-201, 0]])
    assert pairs.distances[0] == 2e-201


def test_minimum_subnormal_radius_supports_representable_pair_distances():
    smallest = np.nextafter(0., 1.)
    first, second = grain([[0, 0]]), grain([[smallest, 0]])
    pairs = matching.local_near_pairs(first, second, smallest, exact_tolerance=0)
    np.testing.assert_array_equal(pairs.second, second.positions)
    assert pairs.distances[0] == smallest


def test_bin_rounding_margin_never_expands_the_physical_radius():
    outside = np.nextafter(.1, np.inf)
    first, second = grain([[0, 0]]), grain([[outside, 0]])
    assert np.hypot(*(first.positions[0] - second.positions[0])) > .1
    assert len(matching.same_layer_coincidence_sites(first, second, .1)[0]) == 0
    assert len(matching.local_near_pairs(first, second, .1, exact_tolerance=0).layers) == 0
