"""Same-layer local matching against full distance-matrix oracles."""

import unittest
import numpy as np
from dichromatic_map import crystal, matching

class LocalMatchingTests(unittest.TestCase):
    @staticmethod
    def grain(points, layers=None):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if layers is None:
            layers = np.zeros(len(points), dtype=np.int16)
        return crystal.ProjectedGrain(
            points, np.asarray(layers), np.zeros((len(points), 3), dtype=int)
        )

    def test_spatial_search_matches_brute_force_all_bin_occupants(self):
        rng = np.random.default_rng(20260910)
        for radius in (0.01, 0.05, 0.3, 0.5):
            first = rng.uniform(-1, 1, (180, 2))
            second = rng.uniform(-1, 1, (190, 2))
            forward = matching._nearest_in_radius(first, second, radius)
            d2 = np.sum((first[:, None] - second[None, :]) ** 2, axis=2)
            expected = np.argmin(d2, axis=1)
            expected[d2[np.arange(len(first)), expected] > radius**2] = -1
            np.testing.assert_array_equal(forward, expected)
        # Multiple occupants in one bin: first stored site is not a match.
        first = self.grain([[0, 0]])
        second = self.grain([[0.49, 0.49], [0.01, 0]])
        pairs = matching.local_near_pairs(first, second, 0.5)
        np.testing.assert_array_equal(pairs.second, [[0.01, 0]])

    def test_mutual_assignment_layers_cutoff_and_exact_exclusion(self):
        first = self.grain([[0, 0], [0.02, 0], [1, 0], [2, 0], [3, 0]], [0, 0, 1, 2, 3])
        second = self.grain([[0.005, 0], [1, 0], [2.01, 0], [3, 0]], [0, 0, 2, 3])
        pairs = matching.local_near_pairs(first, second, 0.05)
        np.testing.assert_array_equal(pairs.first, [[0, 0], [2, 0]])
        np.testing.assert_array_equal(pairs.layers, [0, 2])
        self.assertEqual(len(set(map(tuple, pairs.second))), len(pairs.second))
        # The wrong layer at [1,0] never matches; exact [3,0] stays gold only.
        self.assertEqual(len(matching.local_near_pairs(first, second, 0.001).layers), 0)
        self.assertEqual(
            len(matching.local_near_pairs(first, self.grain([]), 0.05).layers), 0
        )
        np.testing.assert_array_equal(first.positions[1], [0.02, 0])
        for invalid in (0, -0.1, 0.51, np.nan, np.inf):
            with self.assertRaises(ValueError):
                matching.local_near_pairs(first, second, invalid)

    def test_ties_and_translation_are_deterministic(self):
        first = self.grain([[0, 0]])
        second = self.grain([[0.125, 0], [-0.125, 0]])
        reference = matching.local_near_pairs(first, second, 0.25)
        np.testing.assert_array_equal(reference.second, [[-0.125, 0]])
        permuted = matching.local_near_pairs(
            first, self.grain(second.positions[::-1]), 0.25
        )
        np.testing.assert_array_equal(reference.second, permuted.second)
        shift = np.array([123456.0, -234567.0])
        translated = matching.local_near_pairs(
            self.grain(first.positions + shift),
            self.grain(second.positions + shift),
            0.25,
        )
        np.testing.assert_array_equal(translated.second - shift, reference.second)

    def test_unstrained_multilayer_crystals_against_brute_force(self):
        for lattice in ("FCC", "BCC"):
            for axis in ("100", "110", "111", "112", "1 -1 3"):
                options = dict(lattice=lattice, axis=axis)
                first = crystal.projected_columns(5, 4, +26.56, **options)
                second = crystal.projected_columns(5, 4, -26.56, **options)
                original = first.positions.copy(), second.positions.copy()
                previous = 0
                for radius in (0.02, 0.05, 0.2):
                    pairs = matching.local_near_pairs(first, second, radius)
                    self.assertGreaterEqual(len(pairs.layers), previous)
                    previous = len(pairs.layers)
                    expected = []
                    for layer in range(first.layer_count):
                        a, b = (
                            first.positions[first.layers == layer],
                            second.positions[second.layers == layer],
                        )
                        if not len(a) or not len(b):
                            continue
                        d = np.linalg.norm(a[:, None] - b[None, :], axis=2)
                        j = d.argmin(axis=1)
                        i = np.arange(len(a))
                        mask = (
                            (d[i, j] <= radius)
                            & (d[i, j] > 1e-6)
                            & (d.argmin(axis=0)[j] == i)
                        )
                        expected.extend(
                            (layer, *x, *y) for x, y in zip(a[mask], b[j[mask]])
                        )
                    actual = [
                        (int(layer), *x, *y)
                        for layer, x, y in zip(pairs.layers, pairs.first, pairs.second)
                    ]
                    self.assertEqual(actual, expected, (lattice, axis, radius))
                np.testing.assert_array_equal(first.positions, original[0])
                np.testing.assert_array_equal(second.positions, original[1])



def test_many_pairs_keep_assignments_across_query_batches_and_translation():
    # More than one query batch; spacing ensures each original endpoint has
    # exactly one counterpart. Reverse enumeration to avoid relying on order.
    first_points = np.column_stack((2 * np.arange(9001), np.zeros(9001)))
    second_points = first_points + [0.0625, 0]
    for shift in (np.zeros(2), np.array([1_000_000.0, -1_000_000.0])):
        first = LocalMatchingTests.grain(first_points + shift)
        second = LocalMatchingTests.grain((second_points + shift)[::-1])
        pairs = matching.local_near_pairs(first, second, distance=0.125)
        np.testing.assert_array_equal(pairs.first, first_points + shift)
        np.testing.assert_array_equal(pairs.second, second_points + shift)
        np.testing.assert_allclose(pairs.distances, 0.0625, rtol=0, atol=1e-12)


def test_exact_coincidence_tolerance_spans_neighboring_bins():
    tolerance = 1e-6
    first = LocalMatchingTests.grain([[0.9e-6, 0], [2, 0], [4, 0]])
    second = LocalMatchingTests.grain([[1.1e-6, 0], [2 + 0.5e-6, 0], [4 + 2e-6, 0]])
    sites = matching.same_layer_coincidence_sites(first, second, tolerance)[0]
    np.testing.assert_allclose(sites, [[1e-6, 0], [2 + 0.25e-6, 0]], atol=1e-12)
