"""Full-region, selected-layer and half-open atom counts."""

import unittest
import numpy as np
from dichromatic_map import cells as cell_ops, crystal, matching, strain as strain_ops
from .helpers import corners, fcc22_diamond_pairs

class CountingTests(unittest.TestCase):
    def test_sc_100_single_layer_square_has_integer_site_counts(self):
        # Direct square-grid count: nine closed sites, one interior site,
        # and four representatives in [0,2) × [0,2).
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        for layer in (-1, 0):
            counts = cell_ops.count_cell_atoms(
                polygon, 0, (np.eye(2), np.eye(2)), "SC", "100", layer=layer
            )
            np.testing.assert_array_equal(counts.interior, [[1], [1]])
            np.testing.assert_array_equal(counts.boundary, [[8], [8]])
            np.testing.assert_array_equal(counts.half_open, [[4], [4]])
            np.testing.assert_array_equal(counts.half_open_edges, [[2], [2]])
            np.testing.assert_array_equal(counts.half_open_corners, [[1], [1]])
            self.assertEqual(counts.area, 4)

        # Translate the two whole grains independently. Counts refer to their
        # own physical polygons and must still contain exactly four sites.
        shifts = np.array([[123.25, -46.5], [-8.125, 7.25]])
        shifted = cell_ops.count_cell_atoms(
            polygon[None] + shifts[:, None], 0, (np.eye(2), np.eye(2)),
            "SC", "100", layer=0, translations=shifts
        )
        np.testing.assert_array_equal(shifted.half_open, [[4], [4]])
        with self.assertRaises(ValueError):
            cell_ops.count_cell_atoms(
                polygon, 0, (np.eye(2), np.eye(2)), "SC", "100", layer=1
            )

    def test_independent_grain_polygons_and_selected_layer(self):
        polygons = np.array(
            [
                [[0, 0], [2, 0], [2, 2], [0, 2]],
                [[0, 0], [3, 0], [3, 2], [0, 2]],
            ],
            dtype=float,
        )
        counts = cell_ops.count_cell_atoms(
            polygons, 0, (np.eye(2), np.eye(2)), "BCC", "100", layer=0
        )
        np.testing.assert_array_equal(counts.interior, [[1, 0], [2, 0]])
        np.testing.assert_array_equal(counts.boundary, [[8, 0], [10, 0]])
        np.testing.assert_array_equal(counts.half_open, [[4, 0], [6, 0]])
        np.testing.assert_array_equal(counts.areas, [4, 6])
        np.testing.assert_array_equal(counts.half_open_available, [True, True])
        with self.assertRaises(ValueError):
            _ = counts.area  # No averaging two different grain areas.
        polygons[1, 2] = [2, 2]
        counts = cell_ops.count_cell_atoms(
            polygons, 0, (np.eye(2), np.eye(2)), "BCC", "100", layer=0
        )
        np.testing.assert_array_equal(counts.half_open_available, [True, False])
        np.testing.assert_array_equal(counts.areas, [4, 5])
        self.assertEqual(counts.half_open[0, 0], 4)
        for layer in (-2, 2, 0.5):
            with self.assertRaises(ValueError):
                cell_ops.count_cell_atoms(
                    polygons, 0, (np.eye(2), np.eye(2)), "BCC", "100", layer=layer
                )

    def test_fcc22_actual_pair_polygons_not_midpoints(self):
        midpoints, polygons = fcc22_diamond_pairs()
        counts = cell_ops.count_cell_atoms(
            polygons, 22, (np.eye(2), np.eye(2)), "FCC", "110", layer=1
        )
        np.testing.assert_array_equal(counts.interior, [[0, 34], [0, 34]])
        np.testing.assert_array_equal(counts.boundary, [[0, 14], [0, 14]])
        np.testing.assert_array_equal(counts.half_open, [[0, 40], [0, 40]])
        np.testing.assert_array_equal(counts.half_open_edges, [[0, 5], [0, 5]])
        np.testing.assert_array_equal(counts.half_open_corners, [[0, 1], [0, 1]])
        averaged = cell_ops.count_cell_atoms(
            midpoints, 22, (np.eye(2), np.eye(2)), "FCC", "110", layer=1
        )
        np.testing.assert_array_equal(averaged.half_open, [[0, 36], [0, 36]])

    def test_closed_vs_half_open_and_visible_subset(self):
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        counts = cell_ops.count_cell_atoms(
            polygon, 0, (np.eye(2), np.eye(2)), "BCC", "100"
        )
        np.testing.assert_array_equal(counts.interior, [[1, 4], [1, 4]])
        np.testing.assert_array_equal(counts.boundary, [[8, 0], [8, 0]])
        np.testing.assert_array_equal(counts.half_open, [[4, 4], [4, 4]])
        filtered = cell_ops.count_cell_atoms(
            polygon,
            0,
            (np.eye(2), np.eye(2)),
            "BCC",
            "100",
            [[0, -1], [0, 3]],
            (True, True, False, False),
            1,
        )
        np.testing.assert_array_equal(filtered.half_open, [[0, 4], [0, 0]])
        reverse = cell_ops.count_cell_atoms(
            polygon[::-1], 0, (np.eye(2), np.eye(2)), "BCC", "100"
        )
        np.testing.assert_array_equal(reverse.half_open, counts.half_open)
        self.assertEqual(counts.area, 4)

    def test_exact_and_strained_counts_equal_lattice_determinants(self):
        for lattice in ("FCC", "BCC", "SC"):
            for axis in ("100", "110", "111", "112", "1 -1 3"):
                preset = min(
                    crystal.csl_presets(axis), key=lambda p: (p.sigma, p.angle_deg)
                )
                cell = matching.exact_csl_cell(
                    preset.angle_deg, lattice=lattice, axis=axis
                )
                result = cell_ops.count_cell_atoms(
                    corners(cell), preset.angle_deg, (cell.f1, cell.f2), lattice, axis
                )
                np.testing.assert_array_equal(result.half_open.sum(axis=1), cell.atoms)
                # Enlarging a cell multiplies its physical contents, not by a
                # number inferred from currently generated / displayed atoms.
                larger = cell_ops.count_cell_atoms(
                    2 * corners(cell),
                    preset.angle_deg,
                    (cell.f1, cell.f2),
                    lattice,
                    axis,
                )
                np.testing.assert_array_equal(
                    larger.half_open.sum(axis=1), 4 * np.array(cell.atoms)
                )
        i, j = strain_ops.candidate_vectors(39.5, 2, 12)
        cell = strain_ops.pareto_cells(
            strain_ops.solve_cells_chunk(39.5, 2, i, j, 0, len(i))[1]
        )[0]
        result = cell_ops.count_cell_atoms(corners(cell), 39.5, (cell.f1, cell.f2))
        np.testing.assert_array_equal(result.half_open.sum(axis=1), cell.atoms)

    def test_polygon_validation_and_no_invented_periodic_count(self):
        for polygon in (
            [[0, 0], [1, 1], [0, 1], [1, 0]],
            [[0, 0], [1, 0], [2, 0], [0, 1]],
            [[0, 0], [1, 0], [1, 0], [0, 1]],
            [[0, 0], [2, 0], [0.3, 0.2], [0, 2]],
        ):
            with self.assertRaises(ValueError):
                cell_ops.validate_cell_vertices(polygon)
        polygon = [[0, 0], [2, 0], [1.8, 1.8], [0, 2]]
        result = cell_ops.count_cell_atoms(
            polygon, 0, (np.eye(2), np.eye(2)), "BCC", "100"
        )
        self.assertIsNone(result.half_open)
        self.assertGreater(result.interior.sum(), 0)
        with self.assertRaises(crystal.GeometryLimitError):
            cell_ops.count_cell_atoms(
                np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]]),
                0,
                (np.eye(2), np.eye(2)),
            )



def test_full_region_counts_follow_independent_grain_translations():
    polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    reference = cell_ops.count_cell_atoms(polygon, 0, (np.eye(2), np.eye(2)),
        "BCC", "100")
    # The selected polygon can lie far outside any separately generated view.
    # One full region is counted for each grain with its own uniform shift.
    translations = np.array([[1234.25, -4321.5], [-932.5, 871.25]])
    translated = cell_ops.count_cell_atoms(polygon[None] + translations[:, None],
        0, (np.eye(2), np.eye(2)), "BCC", "100", translations=translations)
    for name in ("interior", "boundary", "half_open", "half_open_edges",
                 "half_open_corners", "areas"):
        np.testing.assert_array_equal(getattr(translated, name), getattr(reference, name))
