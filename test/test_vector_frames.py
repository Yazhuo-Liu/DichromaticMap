"""Current and material direction coordinates in independently deformed grains."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import tilt_gb_crystallography as c
import tilt_gb_dichromatic_pattern_qt as q
import test_manual_cell_rotation as manual
import test_selected_cell_strain as strain_tests


def rotation(angle):
    t = np.deg2rad(angle)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def basis(angle, f, lattice, axis):
    transform = np.eye(3)
    transform[:2, :2] = f @ rotation(angle)
    return transform @ c.get_geometry(lattice, axis).frame.T


class VectorPhysicsTests(unittest.TestCase):
    def test_unstrained_vector_has_two_crystal_representations(self):
        value = np.array([0.5, 0.5, 1.0])
        d = basis(30, np.eye(2), "BCC", "100") @ value
        first = c.crystal_vector_coordinates(d, 30, lattice="BCC", axis="100")
        second = c.crystal_vector_coordinates(d, -30, lattice="BCC", axis="100")
        np.testing.assert_allclose(first.current, value, atol=1e-12)
        self.assertGreater(np.linalg.norm(first.current - second.current), 0.1)
        for v in (first, second):
            np.testing.assert_allclose(v.current_frame @ v.current, d, atol=1e-12)
            np.testing.assert_allclose(v.lattice_basis @ v.lattice, d, atol=1e-12)
            np.testing.assert_allclose(v.current, v.lattice, atol=1e-12)
            self.assertFalse(v.strained)
        self.assertEqual(c.format_direction_components(first.current), "a₀/2[1 1 2]")

    def test_strain_changes_current_components_not_material_indices(self):
        for lattice in ("FCC", "BCC"):
            for axis in ("100", "110", "111", "112", "1 -1 3"):
                u = np.array([[1.04, 0.009], [0.009, 0.98]])
                r = rotation(0.7)
                f = r @ u
                ref = np.array([0.5, 1.0, 1.5])
                d = basis(19, f, lattice, axis) @ ref
                v = c.crystal_vector_coordinates(d, 19, f, lattice, axis)
                np.testing.assert_allclose(v.lattice, ref, atol=1e-12)
                u3 = np.eye(3)
                u3[:2, :2] = u
                ref_frame = basis(19, np.eye(2), lattice, axis)
                expected = ref_frame.T @ u3 @ ref_frame @ ref
                np.testing.assert_allclose(v.current, expected, atol=1e-12)
                self.assertGreater(np.linalg.norm(v.current - ref), 1e-4)
                self.assertTrue(v.strained)
                self.assertAlmostEqual(
                    np.linalg.norm(v.current), np.linalg.norm(d), places=12
                )

    def test_rigid_rotation_alone_is_not_strain(self):
        ref = np.array([0.5, 0.5, 1.0])
        for angle in (0, 19, -25):
            f = rotation(1.1)
            d = basis(angle, f, "FCC", "110") @ ref
            v = c.crystal_vector_coordinates(d, angle, f)
            np.testing.assert_allclose(v.current, ref, atol=1e-12)
            np.testing.assert_allclose(v.lattice, ref, atol=1e-12)
            self.assertFalse(v.strained)

    def test_cross_grain_uses_both_endpoint_deformations_and_translations(self):
        fs = (
            rotation(0.6) @ np.array([[1.03, 0.008], [0.008, 0.98]]),
            rotation(-0.4) @ np.array([[0.99, -0.006], [-0.006, 1.02]]),
        )
        shifts = (np.array([0.3, -0.2, 0]), np.array([-0.1, 0.4, 0]))
        refs = (np.array([0.5, 0.5, 1]), np.array([2, 1, 0.5]))
        bs = [basis(a, f, "FCC", "112") for a, f in zip((17, -17), fs)]
        d = bs[1] @ refs[1] + shifts[1] - (bs[0] @ refs[0] + shifts[0])
        for g, angle in enumerate((17, -17)):
            v = c.crystal_vector_coordinates(d, angle, fs[g], "FCC", "112")
            np.testing.assert_allclose(v.lattice, np.linalg.solve(bs[g], d), atol=1e-12)
            np.testing.assert_allclose(v.current_frame @ v.current, d, atol=1e-12)
            self.assertGreater(np.linalg.norm(v.lattice - (refs[1] - refs[0])), 0.1)
            self.assertGreater(np.linalg.norm(v.current - v.lattice), 0.001)

    def test_format_preserves_length_and_does_not_invent_integer_indices(self):
        for vector, expected in (
            ([0.5, 0.5, 1], "a₀/2[1 1 2]"),
            ([-1, -1, -2], "a₀[-1 -1 -2]"),
            ([1.5, 1.5, 3], "3a₀/2[1 1 2]"),
            ([0, 0, 0], "a₀[0 0 0]"),
            ([1 / 3, 2 / 3, 0], "a₀/3[1 2 0]"),
        ):
            self.assertEqual(c.format_direction_components(vector), expected)
        self.assertEqual(
            c.format_direction_components([0.5, 0.5, 1], unit=""), "1/2[1 1 2]"
        )
        for vector in (
            [0.500001, 0.5, 1],
            [np.sqrt(2), 1, 0],
            [1 / 37, 1 / 41, 0],
            [257, 1, 0],
        ):
            self.assertTrue(c.format_direction_components(vector).startswith("≈ "))
        for deformation in (np.zeros((2, 2)), np.diag([-1, 1]), [[1, 0], [0, np.nan]]):
            with self.assertRaises(ValueError):
                c.crystal_vector_coordinates([1, 2, 3], 0, deformation)


class VectorQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = q.create_application()

    window = manual.ManualQtTests.window
    wait_for = manual.ManualQtTests.wait_for
    setUp = strain_tests.SelectedStrainQtTests.setUp

    def pick_visible(self, w, grain, target):
        points = w.grains[grain].positions
        for index in np.argsort(np.linalg.norm(points - target, axis=1))[:100]:
            point = points[index]
            selected, _ = w._nearest_atom(point)
            if selected is None or selected.grain_index != grain:
                continue
            if (
                w.selected_atoms
                and np.linalg.norm(point - w.selected_atoms[0].position) < 0.1
            ):
                continue
            w._handle_view_click(w._to_view(point))
            return
        self.fail("No visible unique atom found for the intended grain")

    def test_cross_grain_readout_and_arrow_use_two_frames(self):
        w = self.window(lattice="BCC", axis="100", angle_deg=37)
        w._start_vector_measurement()
        self.pick_visible(w, 0, [-1, -1])
        self.pick_visible(w, 1, [1, 1])
        self.assertEqual([a.grain_index for a in w.selected_atoms], [0, 1])
        values = w._selected_crystal_vectors()
        self.assertEqual(len(values), 2)
        text = w.vector_annotation.toPlainText()
        self.assertIn("G1 current", text)
        self.assertIn("G2 current", text)
        self.assertNotIn("no unique", text)
        self.assertGreater(
            w.vector_annotation.zValue(),
            max(item.zValue() for item in w.coincidence_items),
        )
        w.rotation_spin.setValue(47)
        for (_, new), (_, old) in zip(w._selected_crystal_vectors(), values):
            np.testing.assert_allclose(new.current, old.current, atol=1e-12)
        w.selected_atoms.reverse()
        w._draw_vector()
        for (_, new), (_, old) in zip(w._selected_crystal_vectors(), values):
            np.testing.assert_allclose(new.current, -old.current, atol=1e-12)
        np.testing.assert_allclose(
            np.column_stack(w.vector_item.getData()),
            w._to_view([a.position for a in w.selected_atoms]),
            atol=1e-12,
        )

    def test_applied_bulk_strain_shows_current_and_lattice_vectors(self):
        w = strain_tests.SelectedStrainQtTests.pick_local(self, workers=2)
        w.manual_strain_button.click()
        self.wait_for(w, lambda: w.parallel_stage is None and not w.csl_updating)
        w._start_vector_measurement()
        self.pick_visible(w, 0, [-1, -1])
        self.pick_visible(w, 1, [1, 1])
        d = w._selected_vector_displacement()
        values = w._selected_crystal_vectors()
        for g, v in values:
            b = basis(
                (1 if g == 0 else -1) * w.angle_deg / 2, w.deformations[g], "FCC", "110"
            )
            np.testing.assert_allclose(b @ v.lattice, d, atol=1e-12)
            self.assertGreater(np.linalg.norm(v.current - v.lattice), 1e-4)
        text = w.vector_annotation.toPlainText()
        for label in (
            "G1 current",
            "G2 current",
            "G1 lattice [uvw]",
            "G2 lattice [uvw]",
        ):
            self.assertIn(label, text)
        w.rotation_spin.setValue(-41)
        w.view_box.translateBy(x=20, y=-16)
        self.wait_for(w, lambda: w.parallel_stage is None and w._buffer_contains_view())
        for (_, new), (_, old) in zip(w._selected_crystal_vectors(), values):
            np.testing.assert_allclose(new.current, old.current, atol=1e-12)

    def test_phase_heights_and_axial_image_in_both_grain_frames(self):
        for axis in ("110", "111", "112", "1 -1 3"):
            w = self.window(lattice="FCC", axis=axis, angle_deg=22)
            atoms = []
            for g in (0, 1):
                candidates = np.flatnonzero(w.grains[g].layers == g)
                i = candidates[
                    np.argmin(np.linalg.norm(w.grains[g].positions[candidates], axis=1))
                ]
                atoms.append(
                    q.SelectedAtom(
                        w.grains[g].positions[i].copy(),
                        g,
                        g,
                        w.grains[g].half_indices[i].copy(),
                    )
                )
            w.selected_atoms = atoms
            w.axial_spin.setValue(2)
            d = w._selected_vector_displacement()
            axial = w.geometry.frame[:, 2]
            expected = (
                0.5 * (atoms[1].half_indices - atoms[0].half_indices) @ axial
                + 2 * w.geometry.axial_period
            )
            self.assertAlmostEqual(d[2], expected, places=12)
            for g, v in w._selected_crystal_vectors():
                np.testing.assert_allclose(
                    basis((1 if g == 0 else -1) * 11, np.eye(2), "FCC", axis)
                    @ v.current,
                    d,
                    atol=1e-12,
                )

    def test_same_grain_strained_direction_is_not_frozen_reference_index(self):
        w = self.window(lattice="BCC", axis="100", angle_deg=0)
        w.deformations = (np.diag([1.04, 0.98]), np.eye(2))
        w._regenerate_buffer(True)
        w._start_vector_measurement()
        self.pick_visible(w, 0, [1, 1])
        self.pick_visible(w, 0, [3, 2])
        g, v = w._selected_crystal_vectors()[0]
        delta = 0.5 * (
            w.selected_atoms[1].half_indices - w.selected_atoms[0].half_indices
        )
        np.testing.assert_allclose(v.lattice, delta, atol=1e-12)
        self.assertGreater(np.linalg.norm(v.current - delta), 0.01)
        self.assertIn("G1 lattice [uvw]", w.vector_annotation.toPlainText())
        self.assertIn(
            c.format_direction_components(v.current), w.vector_annotation.toPlainText()
        )


if __name__ == "__main__":
    unittest.main()
