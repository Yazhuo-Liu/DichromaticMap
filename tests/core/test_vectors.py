"""Actual crystal-frame vectors and material coordinates under strain and rotation."""

import unittest
import numpy as np
from dichromatic_map import crystal
from .helpers import rotation


def basis(angle, f, lattice, axis):
    transform = np.eye(3)
    transform[:2, :2] = f @ rotation(angle)
    return transform @ crystal.get_geometry(lattice, axis).frame.T


class VectorPhysicsTests(unittest.TestCase):
    def test_sc_110_integer_translation_includes_axial_component(self):
        # SC conventional [2,-1,3] projects onto [-1,1,0]/sqrt(2),
        # [0,0,1], [1,1,0]/sqrt(2). Build this without the geometry helper.
        reference = np.array([2, -1, 3])
        angle = 19.0
        f = rotation(0.7) @ np.array([[1.03, 0.015], [0.015, 0.98]])
        planar = f @ rotation(angle) @ [-3 / np.sqrt(2), 3]
        displacement = np.r_[planar, 1 / np.sqrt(2)]
        value = crystal.crystal_vector_coordinates(
            displacement, angle, f, lattice="SC", axis="110"
        )
        np.testing.assert_allclose(value.lattice, reference, atol=1e-12)
        np.testing.assert_allclose(value.current_frame @ value.current,
                                   displacement, atol=1e-12)
        self.assertEqual(crystal.format_direction_components(value.lattice, unit=""),
                         "[2 -1 3]")
        self.assertTrue(value.strained)

    def test_unstrained_vector_has_two_crystal_representations(self):
        value = np.array([0.5, 0.5, 1.0])
        d = basis(30, np.eye(2), "BCC", "100") @ value
        first = crystal.crystal_vector_coordinates(d, 30, lattice="BCC", axis="100")
        second = crystal.crystal_vector_coordinates(d, -30, lattice="BCC", axis="100")
        np.testing.assert_allclose(first.current, value, atol=1e-12)
        self.assertGreater(np.linalg.norm(first.current - second.current), 0.1)
        for v in (first, second):
            np.testing.assert_allclose(v.current_frame @ v.current, d, atol=1e-12)
            np.testing.assert_allclose(v.lattice_basis @ v.lattice, d, atol=1e-12)
            np.testing.assert_allclose(v.current, v.lattice, atol=1e-12)
            self.assertFalse(v.strained)
        self.assertEqual(
            crystal.format_direction_components(first.current), "a₀/2[1 1 2]"
        )

    def test_strain_changes_current_components_not_material_indices(self):
        for lattice in ("FCC", "BCC", "SC"):
            for axis in ("100", "110", "111", "112", "1 -1 3"):
                u = np.array([[1.04, 0.009], [0.009, 0.98]])
                r = rotation(0.7)
                f = r @ u
                ref = np.array([0.5, 1.0, 1.5])
                d = basis(19, f, lattice, axis) @ ref
                v = crystal.crystal_vector_coordinates(d, 19, f, lattice, axis)
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
            v = crystal.crystal_vector_coordinates(d, angle, f)
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
            v = crystal.crystal_vector_coordinates(d, angle, fs[g], "FCC", "112")
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
            self.assertEqual(crystal.format_direction_components(vector), expected)
        self.assertEqual(
            crystal.format_direction_components([0.5, 0.5, 1], unit=""), "1/2[1 1 2]"
        )
        for vector in (
            [0.500001, 0.5, 1],
            [np.sqrt(2), 1, 0],
            [1 / 37, 1 / 41, 0],
            [257, 1, 0],
        ):
            self.assertTrue(
                crystal.format_direction_components(vector).startswith("≈ ")
            )
        for deformation in (np.zeros((2, 2)), np.diag([-1, 1]), [[1, 0], [0, np.nan]]):
            with self.assertRaises(ValueError):
                crystal.crystal_vector_coordinates([1, 2, 3], 0, deformation)
