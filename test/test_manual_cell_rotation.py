"""Manual-cell counting and display-only rotation regressions (Qt offscreen)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import unittest
import numpy as np
import tilt_gb_crystallography as crystal
import tilt_gb_near_csl as near
import tilt_gb_dichromatic_pattern_qt as q


def corners(cell):
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ cell.cell.T


class CountingTests(unittest.TestCase):
    def test_closed_vs_half_open_and_visible_subset(self):
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        counts = crystal.count_cell_atoms(
            polygon, 0, (np.eye(2), np.eye(2)), "BCC", "100"
        )
        np.testing.assert_array_equal(counts.interior, [[1, 4], [1, 4]])
        np.testing.assert_array_equal(counts.boundary, [[8, 0], [8, 0]])
        np.testing.assert_array_equal(counts.half_open, [[4, 4], [4, 4]])
        filtered = crystal.count_cell_atoms(
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
        reverse = crystal.count_cell_atoms(
            polygon[::-1], 0, (np.eye(2), np.eye(2)), "BCC", "100"
        )
        np.testing.assert_array_equal(reverse.half_open, counts.half_open)
        self.assertEqual(counts.area, 4)

    def test_exact_and_strained_counts_equal_lattice_determinants(self):
        for lattice in ("FCC", "BCC"):
            for axis in ("100", "110", "111", "112", "1 -1 3"):
                preset = min(
                    crystal.csl_presets(axis), key=lambda p: (p.sigma, p.angle_deg)
                )
                cell = near.exact_csl_cell(preset.angle_deg, lattice=lattice, axis=axis)
                result = crystal.count_cell_atoms(
                    corners(cell), preset.angle_deg, (cell.f1, cell.f2), lattice, axis
                )
                np.testing.assert_array_equal(result.half_open.sum(axis=1), cell.atoms)
                # Enlarging a cell multiplies its physical contents, not by a
                # number inferred from currently generated / displayed atoms.
                larger = crystal.count_cell_atoms(
                    2 * corners(cell),
                    preset.angle_deg,
                    (cell.f1, cell.f2),
                    lattice,
                    axis,
                )
                np.testing.assert_array_equal(
                    larger.half_open.sum(axis=1), 4 * np.array(cell.atoms)
                )
        i, j = near.candidate_vectors(39.5, 2, 12)
        cell = near.pareto_cells(near.solve_cells_chunk(39.5, 2, i, j, 0, len(i))[1])[0]
        result = crystal.count_cell_atoms(corners(cell), 39.5, (cell.f1, cell.f2))
        np.testing.assert_array_equal(result.half_open.sum(axis=1), cell.atoms)

    def test_polygon_validation_and_no_invented_periodic_count(self):
        for polygon in (
            [[0, 0], [1, 1], [0, 1], [1, 0]],
            [[0, 0], [1, 0], [2, 0], [0, 1]],
            [[0, 0], [1, 0], [1, 0], [0, 1]],
            [[0, 0], [2, 0], [0.3, 0.2], [0, 2]],
        ):
            with self.assertRaises(ValueError):
                crystal.validate_cell_vertices(polygon)
        polygon = [[0, 0], [2, 0], [1.8, 1.8], [0, 2]]
        result = crystal.count_cell_atoms(
            polygon, 0, (np.eye(2), np.eye(2)), "BCC", "100"
        )
        self.assertIsNone(result.half_open)
        self.assertGreater(result.interior.sum(), 0)
        with self.assertRaises(crystal.GeometryLimitError):
            crystal.count_cell_atoms(
                np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]]),
                0,
                (np.eye(2), np.eye(2)),
            )


class ManualQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = q.create_application()

    def wait_for(self, w, predicate, timeout=30):
        until = time.monotonic() + timeout
        while not predicate() and time.monotonic() < until:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(
            predicate(), w.manual_info.toPlainText() + "\n" + w.near_info.toPlainText()
        )

    def window(self, **kwargs):
        workers = kwargs.pop("workers", 1)
        w = q.DichromaticPatternWindow(
            q.PatternParameters(**kwargs), worker_count=workers
        )
        w.show()
        self.app.processEvents()
        self.addCleanup(w.close)
        return w

    def pick_exact_cell(self, w):
        polygon = corners(w.common_cell)
        w._fit_model_corners(polygon)
        w._regenerate_buffer(True)
        w._start_manual_cell()
        for point in polygon:
            w._handle_view_click(w._to_view(point))
        self.assertEqual(len(w.manual_vertices), 4, w.status_label.text())
        self.wait_for(w, lambda: w.manual_counts is not None)
        return polygon

    def test_rotation_picking_vectors_boundary_and_physics_invariant(self):
        w = self.window(lattice="BCC", axis="100", workers=2)
        angle = w.angle_deg
        signature = w._geometry_signature()
        original = [grain.positions.copy() for grain in w.grains]
        w._start_vector_measurement()
        point = w.grains[0].positions[
            np.argmin(np.linalg.norm(w.grains[0].positions - [1, 1], axis=1))
        ]
        w._handle_view_click(point)
        picked = w.selected_atoms[0]
        w.rotation_slider.setValue(573)
        self.assertEqual(w.display_rotation_deg, 57.3)
        self.assertEqual(w._geometry_signature(), signature)
        self.assertEqual(w.angle_deg, angle)
        self.assertIs(w.selected_atoms[0], picked)
        for grain, old in zip(w.grains, original):
            np.testing.assert_array_equal(grain.positions, old)
        w._handle_view_click(w._to_view([0, 0]))
        self.assertEqual(len(w.selected_atoms), 2)
        text = w._selected_vector_readout()
        line = np.column_stack(w.vector_item.getData())
        np.testing.assert_allclose(
            line, w._to_view([a.position for a in w.selected_atoms])
        )
        w.rotation_spin.setValue(-38)
        self.assertEqual(w._selected_vector_readout(), text)
        w._start_new_boundary()
        w._handle_view_click(w._to_view([0, 0]))
        w._handle_view_click(w._to_view(point))
        self.assertEqual(len(w.selected_points), 2)
        boundary = np.array(w.selected_points)
        w._set_region_states((True, False, False, True))
        old_masks = [mask.copy() for mask in w.visible_atom_masks]
        w.rotation_spin.setValue(90)
        np.testing.assert_array_equal(w.selected_points, boundary)
        for mask, old in zip(w.visible_atom_masks, old_masks):
            np.testing.assert_array_equal(mask, old)
        self.wait_for(w, lambda: w.parallel_stage is None and w._buffer_contains_view())
        np.testing.assert_allclose(
            np.column_stack(w.near_cell_item.getData()),
            w._to_view(np.vstack((corners(w.common_cell), [0, 0]))),
            atol=1e-10,
        )

    def test_manual_counts_pan_rotation_and_filters(self):
        w = self.window(lattice="BCC", axis="100", workers=2)
        w.rotation_spin.setValue(37)
        polygon = self.pick_exact_cell(w)
        np.testing.assert_array_equal(
            w.manual_counts.half_open.sum(axis=1), w.common_cell.atoms
        )
        counts = w.manual_counts
        key = w.manual_count_key
        w.view_box.translateBy(x=40, y=-35)
        self.wait_for(w, lambda: w.parallel_stage is None and w._buffer_contains_view())
        self.assertIs(w.manual_counts, counts)
        w.rotation_slider.setValue(-920)
        self.assertIs(w.manual_counts, counts)
        self.assertEqual(w.manual_count_key, key)
        np.testing.assert_allclose(
            np.column_stack(w.manual_cell_item.getData()),
            w._to_view(np.vstack((polygon, polygon[0]))),
            atol=1e-10,
        )
        w.manual_visible_check.setChecked(True)
        w.layer_combo.setCurrentIndex(w.layer_combo.findData(1))
        w.selected_points = [np.array([0, -10]), np.array([0, 10])]
        w._set_region_states((True, True, False, False))
        self.wait_for(w, lambda: w.manual_counts is not None)
        np.testing.assert_array_equal(w.manual_counts.half_open, [[0, 5], [0, 0]])
        w.manual_visible_check.setChecked(False)
        self.wait_for(w, lambda: w.manual_counts is not None)
        np.testing.assert_array_equal(w.manual_counts.half_open.sum(axis=1), [10, 10])
        w._undo_manual_vertex()
        self.assertEqual(len(w.manual_vertices), 3)
        self.assertEqual(w.interaction_mode, "cell")
        self.assertIsNone(w.manual_counts)
        w._clear_manual_cell()
        self.assertEqual(len(w.manual_vertex_item.points()), 0)

    def test_local_vertices_hidden_sources_and_no_periodicity_claim(self):
        w = self.window(lattice="BCC", axis="100", angle_deg=53.12, workers=2)
        w.near_button.setChecked(True)
        self.wait_for(w, lambda: not w.local_updating and w.parallel_stage is None)
        w.rotation_spin.setValue(24)
        candidates = w.local_pairs.midpoints[w.local_pairs.layers == 0]
        # Near Sigma5: pick origin and the three nearest approximate corners.
        side = np.sqrt(5)
        polygon = [np.zeros(2)]
        for target in ([side, 0], [side, side], [0, side]):
            polygon.append(
                candidates[np.argmin(np.linalg.norm(candidates - target, axis=1))]
            )
        w._start_manual_cell()
        for point in polygon:
            w._handle_view_click(w._to_view(point))
        self.assertEqual(len(w.manual_vertices), 4, w.status_label.text())
        self.wait_for(w, lambda: w.manual_counts is not None)
        self.assertIsNone(w.common_cell)
        self.assertIn("periodicity not verified", w.manual_info.toPlainText())
        self.assertTrue(any(v.source == "local" for v in w.manual_vertices))
        w.local_distance_spin.setValue(0.04)
        self.assertEqual(
            w.manual_vertices, []
        )  # invalidate previous near-pair source definition
        self.wait_for(w, lambda: not w.local_updating and w.parallel_stage is None)
        w.selected_points = [np.array([0, -10]), np.array([0, 10])]
        w._set_region_states((True, True, False, False))
        self.assertEqual(w._common_site_candidates(), [])
        w._start_manual_cell()
        w._handle_view_click(w._to_view([0, 0]))
        self.assertEqual(w.manual_vertices, [])

    def test_partial_pick_survives_navigation_and_stale_counts_discarded(self):
        w = self.window(lattice="BCC", axis="100", workers=2)
        polygon = corners(w.common_cell) * 4
        w._regenerate_buffer(True)
        w._start_manual_cell()
        w._handle_view_click(w._to_view(polygon[0]))
        first = w.manual_vertices[0]
        for point in polygon[1:]:
            view = w._to_view(point)
            w.view_box.setRange(
                xRange=(view[0] - 2, view[0] + 2),
                yRange=(view[1] - 2, view[1] + 2),
                padding=0,
            )
            self.wait_for(
                w, lambda: w.parallel_stage is None and w._buffer_contains_view()
            )
            w._handle_view_click(view)
        self.assertIs(w.manual_vertices[0], first)
        self.assertEqual(len(w.manual_vertices), 4)
        self.wait_for(w, lambda: w.manual_counts is not None)
        np.testing.assert_array_equal(w.manual_counts.half_open.sum(axis=1), [160, 160])
        # Queue a count and change the physical crystal before its result is polled.
        w.manual_count_key = None
        w._queue_manual_count()
        w.axis_combo.setCurrentIndex(w.axis_combo.findData("111"))
        self.wait_for(
            w, lambda: w.parallel_stage is None and w.manual_count_future is None
        )
        self.assertEqual(w.manual_vertices, [])
        self.assertIsNone(w.manual_counts)


if __name__ == "__main__":
    unittest.main()
