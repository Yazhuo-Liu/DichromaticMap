"""Independent cubic enumeration, axial phases and exact common translations."""

import unittest
import numpy as np
from dichromatic_map import cells as cell_ops, crystal, matching, strain as strain_ops
from itertools import product
from .helpers import rotation, solve

MODELS = tuple(product(("FCC", "BCC"), ("110", "100")))
GENERAL_MODELS = tuple(product(("FCC", "BCC"),
    ("100", "110", "111", "112", "1 -1 3", "2 3 5", "0 0 1")))

class CrystalPhysicsTests(unittest.TestCase):
    def test_geometry_against_independent_cubic_lattice_enumeration(self):
        # The oracle enumerates conventional cubic sites, not the planar basis
        # used by the optimized generator. This catches missing/extra columns.
        indices = np.array(list(product(range(-20, 21), repeat=3)), dtype=int)
        width, height = 6.2, 5.4
        center = np.array([0.3, -0.6])
        for lattice, axis in GENERAL_MODELS:
            geometry = crystal.get_geometry(lattice, axis)
            axial_direction = np.array(crystal.parse_axis(axis))
            # A shortest axial half-index vector must itself be a Bravais site.
            is_site = (
                (sum(axial_direction) % 2 == 0)
                if lattice == "FCC"
                else np.all(axial_direction % 2 == axial_direction[0] % 2)
            )
            repeat = axial_direction * (1 if is_site else 2)
            np.testing.assert_array_equal(geometry.axial_repeat_half_indices, repeat)
            np.testing.assert_allclose(
                geometry.frame.T @ geometry.frame, np.eye(3), atol=1e-12
            )
            self.assertGreater(np.linalg.det(geometry.frame), 0)
            parity = indices % 2
            if lattice == "FCC":
                allowed = indices.sum(axis=1) % 2 == 0
            else:
                allowed = np.all(parity == parity[:, :1], axis=1)
            axial_integer = indices @ axial_direction
            period_integer = int(repeat @ axial_direction)
            allowed &= (axial_integer >= 0) & (axial_integer < period_integer)
            possible = indices[allowed]
            for angle, deformation in (
                (0.0, np.eye(2)),
                (21.7, np.array([[1.03, 0.015], [0.015, 0.98]])),
            ):
                with self.subTest(lattice=lattice, axis=axis, angle=angle):
                    transform = deformation @ rotation(angle)
                    projected = possible @ geometry.frame[:, :2] @ transform.T / 2
                    keep = np.all(
                        np.abs(projected - center) <= [width / 2, height / 2], axis=1
                    )
                    expected = set(map(tuple, possible[keep]))
                    grain = crystal.projected_columns(
                        width, height, angle, center, deformation, lattice, axis
                    )
                    actual = set(map(tuple, grain.half_indices))
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(actual), len(grain.positions))
                    self.assertEqual(
                        set(grain.layers), set(range(geometry.layer_count))
                    )
                    self.assertEqual(grain.layer_count, geometry.layer_count)
                    np.testing.assert_allclose(
                        grain.positions,
                        grain.half_indices @ geometry.frame[:, :2] @ transform.T / 2,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        grain.half_indices @ geometry.frame[:, 2] / 2,
                        grain.layers * geometry.layer_spacing,
                        atol=1e-12,
                    )

    def test_axis_parsing_computed_layers_and_resource_limits(self):
        for lattice in ("FCC", "BCC"):
            for axis, count in (
                ("100", 2),
                ("110", 2),
                ("111", 3),
                ("112", 6),
                ("1 -1 3", 11),
                ("2 3 5", 38),
            ):
                geometry = crystal.get_geometry(lattice, axis)
                self.assertEqual(geometry.layer_count, count)
                self.assertAlmostEqual(
                    geometry.layer_spacing * count, geometry.axial_period
                )
                np.testing.assert_array_equal(
                crystal.get_geometry(lattice, "[2, 2, 4]").axis_indices,
                crystal.get_geometry(lattice, "112").axis_indices,
            )
            np.testing.assert_array_equal(crystal.parse_axis("<-2 2 -6>"), [-1, 1, -3])
            for invalid in ("0 0 0", "1 2", "1.5 0 1", "nan 1 1"):
                with self.assertRaises(ValueError):
                    crystal.get_geometry(lattice, invalid)
            with self.assertRaises(crystal.GeometryLimitError):
                crystal.get_geometry(lattice, "15 16 1")
            with self.assertRaises(crystal.GeometryLimitError):
                crystal.projected_columns(1000, 1000, 0, lattice=lattice)

    def test_exact_cells_and_strains_for_multilayer_axes(self):
        for lattice, axis in GENERAL_MODELS:
            if axis in ("100", "110"):
                continue
            geometry = crystal.get_geometry(lattice, axis)
            options = dict(lattice=lattice, axis=axis)
            presets = sorted(
                crystal.csl_presets(axis), key=lambda p: (p.sigma, p.angle_deg)
            )[:3]
            for preset in presets:
                with self.subTest(lattice=lattice, axis=axis, angle=preset.angle_deg):
                    cell = matching.exact_csl_cell(preset.angle_deg, **options)
                    self.assertIsNotNone(cell)
                    b1, b2 = cell_ops.bases(preset.angle_deg, **options)
                    np.testing.assert_allclose(b1 @ cell.m1, b2 @ cell.m2, atol=1e-9)
                    self.assertEqual(
                        cell.atoms,
                        tuple(
                            geometry.layer_count * abs(cell_ops.determinant(m))
                            for m in (cell.m1, cell.m2)
                        ),
                    )
                    # Each common translation is a legal, zero-axial lattice
                    # translation, so it preserves *every* phase offset.
                    for m in (cell.m1, cell.m2):
                        translation = geometry.basis_half_indices @ m
                        np.testing.assert_array_equal(
                            geometry.axis_indices @ translation, [0, 0]
                        )
                        for offset in geometry.layer_offsets_half_indices:
                            sites = offset[:, None] + translation
                            if lattice == "FCC":
                                self.assertTrue(np.all(sites.sum(axis=0) % 2 == 0))
                            else:
                                self.assertTrue(np.all(sites % 2 == sites[:1] % 2))
            if axis not in ("111", "112", "1 -1 3"):
                continue
            angle = 21.4 if axis == "111" else presets[0].angle_deg + 0.4
            cells = solve(lattice, axis, angle)
            self.assertTrue(cells, (lattice, axis))
            b1, b2 = cell_ops.bases(angle, **options)
            for cell in cells:
                np.testing.assert_allclose(
                    cell.f1 @ b1 @ cell.m1, cell.f2 @ b2 @ cell.m2, atol=1e-9
                )
                for f, e in zip(
                    (cell.f1, cell.f2), strain_ops.strain_tensors(cell, angle)
                ):
                    np.testing.assert_allclose(f, f.T, atol=1e-12)
                    self.assertGreater(np.linalg.eigvalsh(f).min(), 0)
                    self.assertLessEqual(
                        np.max(np.abs(np.linalg.eigvalsh(f) - 1)), 0.02 + 1e-10
                    )
                    np.testing.assert_allclose(e @ geometry.frame[:, 2], 0, atol=1e-12)

    def test_coincidences_never_mix_computed_axial_phases(self):
        from dichromatic_map.matching import same_layer_coincidence_sites

        for count in (3, 6, 11, 38):
            positions = np.column_stack((np.arange(count), np.zeros(count)))
            indices = np.zeros((count, 3), dtype=int)
            first = crystal.ProjectedGrain(positions, np.arange(count), indices, count)
            wrong_layers = crystal.ProjectedGrain(
                positions, np.roll(np.arange(count), 1), indices, count
            )
            self.assertEqual(
                sum(map(len, same_layer_coincidence_sites(first, wrong_layers, 1e-6))),
                0,
            )
            correct = same_layer_coincidence_sites(first, first, 1e-6)
            self.assertEqual(list(map(len, correct)), [1] * count)

    def test_exact_cells_presets_preserve_both_phases(self):
        for lattice, axis in MODELS:
            options = dict(lattice=lattice, axis=axis)
            for preset in crystal.csl_presets(axis):
                with self.subTest(lattice=lattice, axis=axis, preset=preset.label):
                    cell = matching.exact_csl_cell(preset.angle_deg, **options)
                    self.assertIsNotNone(cell)
                    self.assertEqual((cell.lattice, cell.axis), (lattice, axis))
                    self.assertEqual(cell.max_strain, 0)
                    b1, b2 = cell_ops.bases(preset.angle_deg, **options)
                    np.testing.assert_allclose(b1 @ cell.m1, cell.cell, atol=1e-9)
                    np.testing.assert_allclose(b2 @ cell.m2, cell.cell, atol=1e-9)
                    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) @ cell.cell.T
                    span = 2 * np.max(np.abs(corners), axis=0) + 6
                    for sign in (1, -1):
                        grain = crystal.projected_columns(
                            *span, sign * preset.angle_deg / 2, **options
                        )
                        a_points = grain.positions[grain.layers == 0]
                        for point in corners:
                            self.assertLess(
                                np.min(np.linalg.norm(a_points - point, axis=1)), 1e-8
                            )
                        for phase in (0, 1):
                            points = grain.positions[grain.layers == phase]
                            anchor = points[np.argmin(np.linalg.norm(points, axis=1))]
                            for translation in cell.cell.T:
                                self.assertLess(
                                    np.min(
                                        np.linalg.norm(
                                            points - anchor - translation, axis=1
                                        )
                                    ),
                                    1e-8,
                                )



def test_projected_crop_keeps_edges_and_uniform_translation():
    for lattice in ("FCC", "BCC"):
        boundary = crystal.projected_columns(2, 2, 0, lattice=lattice, axis="100")
        assert np.any(boundary.positions[:, 0] == -1)
        assert np.any(boundary.positions[:, 0] == 1)
        assert np.any(boundary.positions[:, 1] == -1)
        assert np.any(boundary.positions[:, 1] == 1)
        deformation = np.array([[1.02, 0.07], [-0.03, 0.97]])
        center = np.array([0.3, -0.4])
        shift = np.array([1234.25, -4321.5])
        reference = crystal.projected_columns(5, 4, 19.2, center,
            deformation, lattice, "112")
        translated = crystal.projected_columns(5, 4, 19.2, center + shift,
            deformation, lattice, "112", translation=shift)
        np.testing.assert_array_equal(translated.half_indices, reference.half_indices)
        np.testing.assert_array_equal(translated.layers, reference.layers)
        np.testing.assert_allclose(translated.positions, reference.positions + shift,
            rtol=0, atol=2e-12)
