"""Selected local-cell deformation, registration, and Qt lifecycle regressions."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import tilt_gb_crystallography as crystal
import tilt_gb_near_csl as near
import tilt_gb_dichromatic_pattern_qt as q
import test_manual_cell_rotation as manual


class SelectedStrainPhysicsTests(unittest.TestCase):
    def test_fcc22_four_pairs_bulk_strain_and_rotation_are_separate(self):
        _, vertices = manual.fcc22_diamond_pairs()
        before = vertices.copy()
        fit = near.strain_selected_cell(vertices, 22, layer=1)
        np.testing.assert_array_equal(vertices, before)
        np.testing.assert_allclose(fit.vertices[0], fit.vertices[1], atol=1e-12)
        self.assertAlmostEqual(100 * fit.cell.max_strain, 0.444310154, places=8)
        np.testing.assert_allclose(fit.rotations_deg, [0.168543, -0.168543], atol=1e-7)
        for g, f in enumerate((fit.cell.f1, fit.cell.f2)):
            t = np.deg2rad(fit.rotations_deg[g])
            r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            u = r.T @ f
            np.testing.assert_allclose(u, u.T, atol=1e-12)
            np.testing.assert_allclose(
                np.sort(np.linalg.eigvalsh(u)), np.sort(fit.stretches[g]), atol=1e-12
            )
            self.assertGreater(np.linalg.det(f), 0)
        counts = crystal.count_cell_atoms(
            fit.vertices,
            22,
            (fit.cell.f1, fit.cell.f2),
            layer=1,
            translations=fit.translations,
        )
        np.testing.assert_array_equal(counts.half_open, [[0, 40], [0, 40]])
        grains = [
            crystal.projected_columns(
                40, 40, sign * 11, deformation=f, translation=shift
            )
            for sign, f, shift in zip(
                (1, -1), (fit.cell.f1, fit.cell.f2), fit.translations
            )
        ]
        sites = q.same_layer_coincidence_sites(*grains, 1e-6)[1]
        self.assertGreater(len(sites), 4)
        # Not merely four snapped endpoints: translated cells also coincide.
        target = fit.origin + fit.cell.cell @ [2, -1]
        self.assertLess(np.linalg.norm(sites - target, axis=1).min(), 1e-10)

    def test_shifted_non_a_layer_keeps_registration_and_counting(self):
        _, vertices = manual.fcc22_diamond_pairs()
        for g, b in enumerate(near.bases(22)):
            vertices[g] += b @ [3, 1]
        fit = near.strain_selected_cell(vertices, 22, layer=1)
        self.assertGreater(
            np.linalg.norm(fit.translations[0] - fit.translations[1]), 0.01
        )
        np.testing.assert_allclose(fit.vertices[0], fit.vertices[1], atol=1e-12)
        counts = crystal.count_cell_atoms(
            fit.vertices,
            22,
            (fit.cell.f1, fit.cell.f2),
            layer=1,
            translations=fit.translations,
        )
        np.testing.assert_array_equal(counts.half_open, [[0, 40], [0, 40]])
        for g, sign in enumerate((1, -1)):
            points = crystal.projected_columns(
                20,
                20,
                sign * 11,
                center=fit.vertices[g].mean(axis=0),
                deformation=(fit.cell.f1, fit.cell.f2)[g],
                translation=fit.translations[g],
            )
            for vertex in fit.vertices[g]:
                self.assertLess(
                    np.linalg.norm(
                        points.positions[points.layers == 1] - vertex, axis=1
                    ).min(),
                    1e-10,
                )

    def test_limits_and_incompatible_fourth_corner_are_not_silently_applied(self):
        _, vertices = manual.fcc22_diamond_pairs()
        for kwargs in (
            {"percent": 0.1},
            {"max_rotation_deg": 0.01},
            {"max_rotation_deg": 0},
            {"layer": 0},
        ):
            arguments = dict(layer=1)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                near.strain_selected_cell(vertices, 22, **arguments)
        vertices[1, 2] += near.bases(22)[1][:, 0]
        with self.assertRaises(ValueError):
            near.strain_selected_cell(vertices, 22, 10, layer=1, max_rotation_deg=5)

    def test_bcc_near_sigma5_rotation_is_not_reported_as_strain(self):
        exact_angle = crystal.csl_angle_deg(2, 1, "100")
        exact = near.exact_csl_cell(exact_angle, lattice="BCC", axis="100")
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        b1, b2 = near.bases(53.12, "BCC", "100")
        vertices = np.stack((uv @ (b1 @ exact.m1).T, uv @ (b2 @ exact.m2).T))
        fit = near.strain_selected_cell(vertices, 53.12, lattice="BCC", axis="100")
        self.assertLess(fit.cell.max_strain, 1e-7)
        self.assertGreater(np.max(np.abs(fit.rotations_deg)), 0.005)
        self.assertLess(fit.residual, 1e-12)

    def test_exact_identity_all_axes_and_original_pure_strain_method(self):
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        for lattice in ("FCC", "BCC"):
            for axis in ("100", "110", "111", "112", "1 -1 3"):
                angle = min(
                    crystal.csl_presets(axis), key=lambda p: (p.sigma, p.angle_deg)
                ).angle_deg
                cell = near.exact_csl_cell(angle, lattice=lattice, axis=axis)
                bases = near.bases(angle, lattice, axis)
                vertices = np.stack(
                    [uv @ (b @ m).T for b, m in zip(bases, (cell.m1, cell.m2))]
                )
                fit = near.strain_selected_cell(
                    vertices, angle, lattice=lattice, axis=axis, max_rotation_deg=0
                )
                np.testing.assert_allclose(fit.cell.f1, np.eye(2), atol=1e-10)
                np.testing.assert_allclose(fit.cell.f2, np.eye(2), atol=1e-10)
        i, j = near.candidate_vectors(39.5, 2, 12)
        cell = near.solve_cells_chunk(39.5, 2, i, j, 0, len(i))[1][0]
        vertices = np.stack(
            [uv @ (b @ m).T for b, m in zip(near.bases(39.5), (cell.m1, cell.m2))]
        )
        fit = near.strain_selected_cell(vertices, 39.5, max_rotation_deg=0)
        np.testing.assert_allclose(fit.cell.f1, cell.f1, atol=1e-10)
        np.testing.assert_allclose(fit.rotations_deg, 0, atol=1e-10)


class SelectedStrainQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = q.create_application()

    window = manual.ManualQtTests.window
    wait_for = manual.ManualQtTests.wait_for

    def setUp(self):
        # Qt slot exceptions otherwise print a traceback but can leave unittest
        # reporting success. Fail explicitly if any asynchronous slot raises.
        old_hook = sys.excepthook
        errors = []
        sys.excepthook = lambda *error: errors.append(error)
        self.addCleanup(setattr, sys, "excepthook", old_hook)
        self.addCleanup(lambda: self.assertEqual(errors, []))

    def test_exact_only_cell_does_not_enable_local_bulk_action(self):
        w = self.window(lattice="BCC", axis="100", workers=1)
        manual.ManualQtTests.pick_exact_cell(self, w)
        self.assertFalse(w.manual_strain_button.isEnabled())

    def pick_local(self, workers=2):
        w = self.window(lattice="FCC", axis="110", angle_deg=22, workers=workers)
        self.assertFalse(w.manual_strain_button.isEnabled())
        points, _ = manual.fcc22_diamond_pairs()
        w._fit_model_corners(points)
        w.near_button.setChecked(True)
        self.wait_for(w, lambda: not w.local_updating and w.parallel_stage is None)
        w._start_manual_cell()
        for point in points[:3]:
            w._handle_view_click(w._to_view(point))
            self.assertFalse(w.manual_strain_button.isEnabled())
        w._handle_view_click(w._to_view(points[3]))
        self.assertTrue(w.manual_strain_button.isEnabled())
        self.wait_for(w, lambda: w.manual_counts is not None)
        return w

    def test_apply_recompute_pan_workers_and_restore_original_local_cell(self):
        w = self.pick_local()
        original = w.manual_vertices
        signature = w._geometry_signature()
        w.manual_strain_button.click()
        self.assertIsNotNone(w.manual_strain_fit, w.manual_strain_note.text())
        fit = w.manual_strain_fit
        self.assertFalse(w.local_active)
        self.assertFalse(w.manual_pick_button.isEnabled())
        self.assertFalse(w.local_distance_spin.isEnabled())
        self.assertNotEqual(w._geometry_signature(), signature)
        self.wait_for(
            w,
            lambda: w.parallel_stage is None
            and not w.csl_updating
            and w.manual_counts is not None,
        )
        np.testing.assert_array_equal(w.manual_counts.half_open, [[0, 40], [0, 40]])
        for vertex in w.manual_vertices:
            self.assertEqual(vertex.source, "CSL")
            self.assertLess(
                np.linalg.norm(w.coincident_points[1] - vertex.position, axis=1).min(),
                1e-10,
            )
        self.assertIn("rotation", w.near_info.toPlainText())
        self.assertIn("principal strains", w.near_info.toPlainText())
        w.rotation_spin.setValue(41)
        w.view_box.translateBy(x=24, y=-21)
        self.wait_for(w, lambda: w.parallel_stage is None and w._buffer_contains_view())
        w.worker_spin.setValue(1)
        self.wait_for(w, lambda: w.parallel_stage is None and not w.csl_updating)
        self.assertIs(w.manual_strain_fit, fit)
        self.assertIn("SELECTED-CELL", w.plot_item.titleLabel.text)
        corners = (
            np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ fit.cell.cell.T
            + fit.origin
        )
        np.testing.assert_allclose(
            np.column_stack(w.near_cell_item.getData()), w._to_view(corners), atol=1e-10
        )
        w.manual_strain_button.click()
        self.wait_for(
            w,
            lambda: w.parallel_stage is None
            and not w.local_updating
            and w.manual_counts is not None,
        )
        self.assertIsNone(w.manual_strain_fit)
        self.assertEqual(w._geometry_signature(), signature)
        self.assertIs(w.manual_vertices, original)
        self.assertTrue(w.local_active)
        np.testing.assert_array_equal(w.manual_counts.half_open, [[0, 40], [0, 40]])

    def test_rejected_fit_keeps_atoms_and_selection_unchanged(self):
        w = self.pick_local(workers=1)
        original = w.manual_vertices
        signature = w._geometry_signature()
        w.manual_strain_limit.setValue(0.1)
        w.manual_strain_button.click()
        self.assertIsNone(w.manual_strain_fit)
        self.assertIs(w.manual_vertices, original)
        self.assertEqual(w._geometry_signature(), signature)
        self.assertIn("above", w.manual_strain_note.text())
        w.manual_strain_limit.setValue(2)
        w.manual_rotation_limit.setValue(0.01)
        w.manual_strain_button.click()
        self.assertIsNone(w.manual_strain_fit)
        self.assertIn("rotation", w.manual_strain_note.text())
        w._undo_manual_vertex()
        self.assertFalse(w.manual_strain_button.isEnabled())

    def test_disable_or_geometry_change_discards_inflight_strained_results(self):
        w = self.pick_local()
        w.manual_strain_button.click()
        w.near_button.setChecked(False)
        self.wait_for(
            w, lambda: w.parallel_stage is None and w.manual_count_future is None
        )
        self.assertIsNone(w.manual_strain_fit)
        np.testing.assert_array_equal(w.translations, np.zeros((2, 2)))
        self.assertEqual(w.manual_vertices, [])
        w.axis_combo.setCurrentIndex(w.axis_combo.findData("111"))
        self.wait_for(w, lambda: w.parallel_stage is None)
        self.assertEqual(w.geometry.layer_count, 3)
        self.assertEqual(len(w.grain_layer_items[0]), 3)
        self.assertIsNone(w.manual_counts)

    def test_direct_axis_change_during_bulk_regeneration_clears_transform(self):
        w = self.pick_local()
        w.manual_strain_button.click()
        w.axis_combo.setCurrentIndex(w.axis_combo.findData("111"))
        self.wait_for(
            w,
            lambda: w.parallel_stage is None
            and not w.local_updating
            and w.manual_count_future is None,
        )
        self.assertEqual(w.geometry.layer_count, 3)
        self.assertEqual(len(w.grain_layer_items[0]), 3)
        self.assertIsNone(w.manual_strain_fit)
        self.assertEqual(w.manual_vertices, [])
        np.testing.assert_array_equal(w.deformations, [np.eye(2), np.eye(2)])
        np.testing.assert_array_equal(w.translations, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
