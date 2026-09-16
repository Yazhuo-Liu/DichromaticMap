"""Bounded homogeneous and selected-cell strains, with physical acceptance checks."""

import unittest
import numpy as np
from dichromatic_map import cells as cell_ops, crystal, matching, strain as strain_ops
import math
from .helpers import rotation, solve, fcc22_diamond_pairs

MODELS = (("FCC", "100"), ("FCC", "110"), ("BCC", "100"), ("BCC", "110"),
          ("SC", "100"), ("SC", "110"))

class CrystalPhysicsTests(unittest.TestCase):
    def test_near_cells_and_cubic_strain_tensors_all_models(self):
        for lattice, axis in MODELS:
            angle = 39.5 if (lattice, axis) == ("FCC", "110") else 37.2
            options = dict(lattice=lattice, axis=axis)
            cells = solve(lattice, axis, angle)
            self.assertTrue(cells, (lattice, axis))
            geometry = crystal.get_geometry(lattice, axis)
            for cell in cells:
                with self.subTest(lattice=lattice, axis=axis, atoms=cell.atoms):
                    self.assertEqual((cell.lattice, cell.axis), (lattice, axis))
                    b1, b2 = cell_ops.bases(angle, **options)
                    np.testing.assert_allclose(
                        cell.f1 @ b1 @ cell.m1, cell.cell, atol=1e-9
                    )
                    np.testing.assert_allclose(
                        cell.f2 @ b2 @ cell.m2, cell.cell, atol=1e-9
                    )
                    self.assertLessEqual(cell.max_strain, 0.02 + 1e-10)
                    for sign, f, e in zip(
                        (1, -1),
                        (cell.f1, cell.f2),
                        strain_ops.strain_tensors(cell, angle),
                    ):
                        np.testing.assert_allclose(f, f.T, atol=1e-12)
                        self.assertGreater(np.min(np.linalg.eigvalsh(f)), 0)
                        np.testing.assert_allclose(e, e.T, atol=1e-12)
                        np.testing.assert_allclose(
                            e @ geometry.frame[:, 2], 0, atol=1e-12
                        )
                        for vector in np.array([[1, 2, 3], [-2, 1, 0], [1, 0, 0]]):
                            coordinates = vector @ geometry.frame
                            deformed = f @ rotation(sign * angle / 2) @ coordinates[:2]
                            length_squared = deformed @ deformed + coordinates[2] ** 2
                            self.assertAlmostEqual(
                                length_squared,
                                vector @ (np.eye(3) + 2 * e) @ vector,
                                places=10,
                            )

class PhysicsTests(unittest.TestCase):
    def test_exact_cells_general_integer_rotations(self):
        # Include many nonpreset CSLs; the algorithm is not Sigma-specific.
        for m in range(1, 25):
            for k in range(math.floor(m / np.sqrt(2)) + 1):
                if math.gcd(m, k) != 1:
                    continue
                angle = float(np.degrees(2 * np.arctan2(np.sqrt(2) * k, m)))
                cell = matching.exact_csl_cell(angle)
                self.assertIsNotNone(cell, (m, k))
                sigma = m * m + 2 * k * k
                while sigma % 2 == 0:
                    sigma //= 2
                self.assertEqual(abs(cell_ops.determinant(cell.m1)), sigma)
                b1, b2 = cell_ops.bases(angle)
                np.testing.assert_allclose(b1 @ cell.m1, b2 @ cell.m2, atol=1e-8)
        for angle in (13.25, 39.5, 58.5, 90.0):
            self.assertIsNone(matching.exact_csl_cell(angle))

    def test_exact_csl_needs_no_strain(self):
        angle = np.degrees(2 * np.arctan2(np.sqrt(2), 4))
        best = solve("FCC", "110", angle)[0]
        self.assertLess(best.max_strain, 1e-10)
        self.assertEqual(best.atoms, (18, 18))

    def test_low_limit_can_find_no_cell(self):
        self.assertEqual(solve("FCC", "110", 13.25, 0.001, 3), [])

class SelectedStrainPhysicsTests(unittest.TestCase):
    def test_sc_100_shifted_near_sigma5_four_pairs_fit_and_count(self):
        # Independent square-lattice construction: at cos(theta)=3/5 and
        # sin(theta)=4/5 these integer edge pairs become a common square.
        # A slight angle change needs small polar rotations, not large strain.
        angle = 53.12
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        integer_cells = (np.array([[2, 1], [-1, 2]]),
                         np.array([[2, -1], [1, 2]]))
        reference_origins = (np.array([3, -2]), np.array([-1, 2]))
        vertices = np.stack([
            (uv @ matrix.T + origin) @ rotation(sign * angle / 2).T
            for matrix, origin, sign in zip(integer_cells, reference_origins, (1, -1))
        ])
        fit = strain_ops.strain_selected_cell(
            vertices, angle, lattice="SC", axis="100", layer=0
        )
        self.assertLess(fit.cell.max_strain, 1e-7)
        self.assertGreater(np.max(np.abs(fit.rotations_deg)), 0.005)
        self.assertGreater(np.linalg.norm(fit.translations[0] - fit.translations[1]), 1)
        np.testing.assert_allclose(fit.vertices[0], fit.vertices[1], atol=1e-12)
        self.assertEqual(fit.cell.atoms, (5, 5))
        counts = cell_ops.count_cell_atoms(
            fit.vertices, angle, (fit.cell.f1, fit.cell.f2), "SC", "100",
            layer=0, translations=fit.translations
        )
        np.testing.assert_array_equal(counts.half_open, [[5], [5]])

    def test_fcc22_four_pairs_bulk_strain_and_rotation_are_separate(self):
        _, vertices = fcc22_diamond_pairs()
        before = vertices.copy()
        fit = strain_ops.strain_selected_cell(vertices, 22, layer=1)
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
        counts = cell_ops.count_cell_atoms(
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
        sites = matching.same_layer_coincidence_sites(*grains, 1e-6)[1]
        self.assertGreater(len(sites), 4)
        # Not merely four snapped endpoints: translated cells also coincide.
        target = fit.origin + fit.cell.cell @ [2, -1]
        self.assertLess(np.linalg.norm(sites - target, axis=1).min(), 1e-10)

    def test_shifted_non_a_layer_keeps_registration_and_counting(self):
        _, vertices = fcc22_diamond_pairs()
        for g, b in enumerate(cell_ops.bases(22)):
            vertices[g] += b @ [3, 1]
        fit = strain_ops.strain_selected_cell(vertices, 22, layer=1)
        self.assertGreater(
            np.linalg.norm(fit.translations[0] - fit.translations[1]), 0.01
        )
        np.testing.assert_allclose(fit.vertices[0], fit.vertices[1], atol=1e-12)
        counts = cell_ops.count_cell_atoms(
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
        _, vertices = fcc22_diamond_pairs()
        for kwargs in (
            {"percent": 0.1},
            {"max_rotation_deg": 0.01},
            {"max_rotation_deg": 0},
            {"layer": 0},
        ):
            arguments = dict(layer=1)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                strain_ops.strain_selected_cell(vertices, 22, **arguments)
        vertices[1, 2] += cell_ops.bases(22)[1][:, 0]
        with self.assertRaises(ValueError):
            strain_ops.strain_selected_cell(
                vertices, 22, 10, layer=1, max_rotation_deg=5
            )

    def test_bcc_near_sigma5_rotation_is_not_reported_as_strain(self):
        exact_angle = crystal.csl_angle_deg(2, 1, "100")
        exact = matching.exact_csl_cell(exact_angle, lattice="BCC", axis="100")
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        b1, b2 = cell_ops.bases(53.12, "BCC", "100")
        vertices = np.stack((uv @ (b1 @ exact.m1).T, uv @ (b2 @ exact.m2).T))
        fit = strain_ops.strain_selected_cell(
            vertices, 53.12, lattice="BCC", axis="100"
        )
        self.assertLess(fit.cell.max_strain, 1e-7)
        self.assertGreater(np.max(np.abs(fit.rotations_deg)), 0.005)
        self.assertLess(fit.residual, 1e-12)

    def test_exact_identity_all_axes_and_original_pure_strain_method(self):
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        for lattice in ("FCC", "BCC", "SC"):
            for axis in ("100", "110", "111", "112", "1 -1 3"):
                angle = min(
                    crystal.csl_presets(axis), key=lambda p: (p.sigma, p.angle_deg)
                ).angle_deg
                cell = matching.exact_csl_cell(angle, lattice=lattice, axis=axis)
                bases = cell_ops.bases(angle, lattice, axis)
                vertices = np.stack(
                    [uv @ (b @ m).T for b, m in zip(bases, (cell.m1, cell.m2))]
                )
                fit = strain_ops.strain_selected_cell(
                    vertices, angle, lattice=lattice, axis=axis, max_rotation_deg=0
                )
                np.testing.assert_allclose(fit.cell.f1, np.eye(2), atol=1e-10)
                np.testing.assert_allclose(fit.cell.f2, np.eye(2), atol=1e-10)
        i, j = strain_ops.candidate_vectors(39.5, 2, 12)
        cell = strain_ops.solve_cells_chunk(39.5, 2, i, j, 0, len(i))[1][0]
        vertices = np.stack(
            [uv @ (b @ m).T for b, m in zip(cell_ops.bases(39.5), (cell.m1, cell.m2))]
        )
        fit = strain_ops.strain_selected_cell(vertices, 39.5, max_rotation_deg=0)
        np.testing.assert_allclose(fit.cell.f1, cell.f1, atol=1e-10)
        np.testing.assert_allclose(fit.rotations_deg, 0, atol=1e-10)



def test_candidate_prefilters_retain_all_small_range_compatible_vectors():
    # Exhaustive Cartesian-product oracle in physical space, small enough that
    # the production 320-vector candidate limit does not discard any result.
    from itertools import product

    extent, percent = 5, 3.0
    first_integers = np.array([(x, y) for x, y in product(range(extent + 1),
        range(-extent, extent + 1)) if x > 0 or y > 0])
    second_integers = np.array([(x, y) for x, y in product(
        range(-extent, extent + 1), repeat=2) if x or y])
    for lattice, axis, angle in (("FCC", "110", 22), ("BCC", "100", 37.2),
                                 ("FCC", "112", 0), ("BCC", "1 -1 3", 21.4),
                                 ("SC", "100", 37.2), ("SC", "112", 0),
                                 ("SC", "1 -1 3", 21.4)):
        first_basis, second_basis = cell_ops.bases(angle, lattice, axis)
        v, w = first_integers @ first_basis.T, second_integers @ second_basis.T
        allowed = np.linalg.norm(v[:, None] - w[None], axis=2) <= (
            percent / 100 + 1e-12) * (np.linalg.norm(v, axis=1)[:, None]
                                    + np.linalg.norm(w, axis=1)[None])
        rows, columns = np.nonzero(allowed)
        expected = {tuple(np.r_[first_integers[i], second_integers[j]])
                    for i, j in zip(rows, columns)}
        assert len(expected) <= strain_ops.MAX_VECTORS
        first, second = strain_ops.candidate_vectors(angle, percent, extent, lattice, axis)
        actual = {tuple(np.r_[i, j]) for i, j in zip(first, second)}
        assert actual == expected, (lattice, axis, angle)
