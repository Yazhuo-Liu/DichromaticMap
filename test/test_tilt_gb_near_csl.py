"""Physics and GUI regressions: run with the base environment, offscreen."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tilt-test-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tilt-test-mpl")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time
import unittest
import math
import numpy as np
import tilt_gb_near_csl as n
import tilt_gb_crystallography as crystal


def solve(angle, percent=2.0, extent=12):
    i, j = n.candidate_vectors(angle, percent, extent)
    return n.pareto_cells(n.solve_cells_chunk(angle, percent, i, j, 0, len(i))[1])


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
            forward = n._nearest_in_radius(first, second, radius)
            d2 = np.sum((first[:, None] - second[None, :]) ** 2, axis=2)
            expected = np.argmin(d2, axis=1)
            expected[d2[np.arange(len(first)), expected] > radius**2] = -1
            np.testing.assert_array_equal(forward, expected)
        # Multiple occupants in one bin: first stored site is not a match.
        first = self.grain([[0, 0]])
        second = self.grain([[0.49, 0.49], [0.01, 0]])
        pairs = n.local_near_pairs(first, second, 0.5)
        np.testing.assert_array_equal(pairs.second, [[0.01, 0]])

    def test_mutual_assignment_layers_cutoff_and_exact_exclusion(self):
        first = self.grain([[0, 0], [0.02, 0], [1, 0], [2, 0], [3, 0]], [0, 0, 1, 2, 3])
        second = self.grain([[0.005, 0], [1, 0], [2.01, 0], [3, 0]], [0, 0, 2, 3])
        pairs = n.local_near_pairs(first, second, 0.05)
        np.testing.assert_array_equal(pairs.first, [[0, 0], [2, 0]])
        np.testing.assert_array_equal(pairs.layers, [0, 2])
        self.assertEqual(len(set(map(tuple, pairs.second))), len(pairs.second))
        # The wrong layer at [1,0] never matches; exact [3,0] stays gold only.
        self.assertEqual(len(n.local_near_pairs(first, second, 0.001).layers), 0)
        self.assertEqual(len(n.local_near_pairs(first, self.grain([]), 0.05).layers), 0)
        np.testing.assert_array_equal(first.positions[1], [0.02, 0])
        for invalid in (0, -0.1, 0.51, np.nan, np.inf):
            with self.assertRaises(ValueError):
                n.local_near_pairs(first, second, invalid)

    def test_ties_and_translation_are_deterministic(self):
        first = self.grain([[0, 0]])
        second = self.grain([[0.125, 0], [-0.125, 0]])
        reference = n.local_near_pairs(first, second, 0.25)
        np.testing.assert_array_equal(reference.second, [[-0.125, 0]])
        permuted = n.local_near_pairs(first, self.grain(second.positions[::-1]), 0.25)
        np.testing.assert_array_equal(reference.second, permuted.second)
        shift = np.array([123456.0, -234567.0])
        translated = n.local_near_pairs(
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
                    pairs = n.local_near_pairs(first, second, radius)
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


class PhysicsTests(unittest.TestCase):
    def test_exact_cells_presets_and_layer_corners(self):
        import tilt_gb_dichromatic_pattern_qt as q

        for preset in q.CSL_PRESETS:
            with self.subTest(preset=preset.label):
                cell = n.exact_csl_cell(preset.angle_deg)
                self.assertIsNotNone(cell)
                self.assertEqual(abs(n.determinant(cell.m1)), preset.sigma)
                self.assertEqual(abs(n.determinant(cell.m2)), preset.sigma)
                self.assertEqual(cell.max_strain, 0)
                for sign in (1, -1):
                    grain = q.fcc_110_projected_columns(
                        3.52, 140, 140, sign * preset.angle_deg / 2
                    )
                    for coeff in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        corner = cell.cell @ coeff * 3.52
                        distance = np.linalg.norm(grain.positions - corner, axis=1)
                        index = np.argmin(distance)
                        self.assertLess(distance[index], 1e-7)
                        self.assertEqual(grain.layers[index], 0)

    def test_exact_cells_general_integer_rotations(self):
        # Include many nonpreset CSLs; the algorithm is not Sigma-specific.
        for m in range(1, 65):
            for k in range(math.floor(m / np.sqrt(2)) + 1):
                if math.gcd(m, k) != 1:
                    continue
                angle = float(np.degrees(2 * np.arctan2(np.sqrt(2) * k, m)))
                cell = n.exact_csl_cell(angle)
                self.assertIsNotNone(cell, (m, k))
                sigma = m * m + 2 * k * k
                while sigma % 2 == 0:
                    sigma //= 2
                self.assertEqual(abs(n.determinant(cell.m1)), sigma)
                b1, b2 = n.bases(angle)
                np.testing.assert_allclose(b1 @ cell.m1, b2 @ cell.m2, atol=1e-8)
        for angle in (13.25, 39.5, 58.5, 90.0):
            self.assertIsNone(n.exact_csl_cell(angle))

    def test_common_cell_strain_and_phase(self):
        import tilt_gb_dichromatic_pattern_qt as q

        a, angle = 3.52, 39.5
        cells = solve(angle)
        self.assertTrue(cells)
        for cell in cells:
            b1, b2 = n.bases(angle)
            np.testing.assert_allclose(cell.f1 @ b1 @ cell.m1, cell.cell, atol=1e-9)
            np.testing.assert_allclose(cell.f2 @ b2 @ cell.m2, cell.cell, atol=1e-9)
            self.assertLessEqual(cell.max_strain, 0.02 + 1e-12)
            for f in (cell.f1, cell.f2):
                np.testing.assert_allclose(f, f.T, atol=1e-12)
                self.assertGreater(np.min(np.linalg.eigvalsh(f)), 0)
            # Reported cubic-frame tensors reproduce physical deformed lengths.
            for g, e in enumerate(n.strain_tensors(cell, angle)):
                np.testing.assert_allclose(e, e.T, atol=1e-12)
                np.testing.assert_allclose(e @ [1, 1, 0], 0, atol=1e-12)
                f = (cell.f1, cell.f2)[g]
                rot = q.rotation_matrix_2d((1 if g == 0 else -1) * angle / 2)
                for vector in np.array([[1, 2, 3], [-2, 1, 0], [1, 1, 0]], float):
                    x, y, z = vector
                    projected = f @ rot @ [(y - x) / np.sqrt(2), z]
                    actual_squared = projected @ projected + (x + y) ** 2 / 2
                    self.assertAlmostEqual(
                        actual_squared,
                        vector @ (np.eye(3) + 2 * e) @ vector,
                        places=10,
                    )
            # Actual atoms at all four common-cell corners in both crystals.
            for g, (f, sign) in enumerate(((cell.f1, 1), (cell.f2, -1))):
                grain = q.fcc_110_projected_columns(
                    a, 140, 140, sign * angle / 2, deformation=f
                )
                for vector in (
                    np.zeros(2),
                    cell.cell[:, 0],
                    cell.cell[:, 1],
                    cell.cell.sum(axis=1),
                ):
                    distance = np.linalg.norm(grain.positions - vector * a, axis=1)
                    k = np.argmin(distance)
                    self.assertLess(distance[k], 1e-7)
                    self.assertEqual(grain.layers[k], 0)
                # In-plane translations map A->A and B->B, axial phase unchanged.
                for layer in (0, 1):
                    points = grain.positions[grain.layers == layer]
                    p = points[np.argmin(np.linalg.norm(points, axis=1))]
                    for vector in cell.cell.T:
                        self.assertLess(
                            np.min(np.linalg.norm(points - p - vector * a, axis=1)),
                            1e-7,
                        )

    def test_exact_csl_needs_no_strain(self):
        angle = np.degrees(2 * np.arctan2(np.sqrt(2), 4))
        best = solve(angle)[0]
        self.assertLess(best.max_strain, 1e-10)
        self.assertEqual(best.atoms, (18, 18))

    def test_cross_layer_overlap_is_not_a_common_site(self):
        import tilt_gb_dichromatic_pattern_qt as q
        import tilt_gb_dichromatic_pattern as m

        first = q.ProjectedGrain(
            np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]), np.zeros((2, 3), int)
        )
        wrong_phase = q.ProjectedGrain(
            first.positions, 1 - first.layers, first.half_indices
        )
        for viewer in (m, q):
            self.assertEqual(
                [
                    len(p)
                    for p in viewer.same_layer_coincidence_sites(
                        first, wrong_phase, 1e-6
                    )
                ],
                [0, 0],
            )
            self.assertEqual(
                [
                    len(p)
                    for p in viewer.same_layer_coincidence_sites(first, first, 1e-6)
                ],
                [1, 1],
            )

    def test_low_limit_can_find_no_cell(self):
        self.assertEqual(solve(13.25, 0.001, 3), [])

    def test_parallel_latest_request_and_off(self):
        search = n.NearSearch(workers=3)
        try:
            search.request(39.5, 2, 12)
            search.poll()
            search.request(58.5, 2, 12)
            start = time.monotonic()
            result = None
            while result is None and time.monotonic() - start < 30:
                result = search.poll()
                time.sleep(0.01)
                self.assertLessEqual(len(search.running), 3)
            self.assertIsNotNone(result, search.error)
            expected = solve(58.5)
            self.assertEqual([c.atoms for c in result], [c.atoms for c in expected])
            for c, e in zip(result, expected):
                np.testing.assert_allclose(c.f1, e.f1, atol=1e-9)
            self.assertGreaterEqual(len(search.process_ids), 2)
            search.request(20.0, 2, 12)
            search.poll()
            search.cancel()
            for _ in range(10):
                self.assertIsNone(search.poll())
                time.sleep(0.01)
            self.assertFalse(search.busy)
        finally:
            search.close()


class ViewerTests(unittest.TestCase):
    def test_qt_exact_cell_controls(self):
        import tilt_gb_dichromatic_pattern_qt as q

        app = q.create_application()
        w = q.DichromaticPatternWindow(q.PatternParameters(), worker_count=1)
        w.show()
        app.processEvents()
        try:
            self.assertIsNone(w.near_search)
            self.assertIsNone(w.near_cell)
            self.assertTrue(w.near_cell_item.isVisible())
            np.testing.assert_array_equal(w.deformations[0], np.eye(2))
            x, y = w.near_cell_item.getData()
            self.assertEqual(len(x), 5)
            self.assertEqual((x[0], y[0]), (x[-1], y[-1]))
            w._start_vector_measurement()
            w._handle_view_click(np.zeros(2))
            picked = w.selected_atoms[0]
            w.cell_check.setChecked(False)
            self.assertFalse(w.near_cell_item.isVisible())
            w.cell_fit_button.click()
            w.cell_check.setChecked(True)
            self.assertIs(w.selected_atoms[0], picked)
            self.assertIsNone(w.near_search)
            for preset in (q.CSL_PRESETS[0], q.CSL_PRESETS[5]):
                w._queue_angle_update(preset.angle_deg)
                w._finish_angle_update()
                self.assertEqual(w.common_cell.atoms, (66, 66))
                self.assertTrue(w.near_cell_item.isVisible())
            w._queue_angle_update(39.5)
            w._finish_angle_update()
            self.assertIsNone(w.common_cell)
            self.assertFalse(w.cell_fit_button.isEnabled())
            self.assertFalse(w.near_cell_item.isVisible())
            # Disabling an active Near-CSL returns the exact outline at a preset.
            w._queue_angle_update(q.DEFAULT_ANGLE_DEG)
            w._finish_angle_update()
            w.near_button.setChecked(True)
            w.near_button.setChecked(False)
            self.assertTrue(w.near_cell_item.isVisible())
            self.assertEqual(w.common_cell.max_strain, 0)
        finally:
            w.close()

    def test_matplotlib_exact_cell_controls(self):
        import tilt_gb_dichromatic_pattern as m

        w = m.DichromaticPatternApp(m.PatternParameters())
        try:
            self.assertIsNone(w.near_search)
            self.assertIsNone(w.near_cell)
            self.assertTrue(w.near_cell_artist.get_visible())
            self.assertEqual(len(w.near_cell_artist.get_xdata()), 5)
            w._start_vector_measurement()
            atom, _ = w._nearest_atom(*w.axes.transData.transform([0.0, 0.0]))
            w.selected_atoms.append(atom)
            w.cell_checks.set_active(0)
            self.assertFalse(w.near_cell_artist.get_visible())
            w._fit_near_cell()
            w.cell_checks.set_active(0)
            self.assertIs(w.selected_atoms[0], atom)
            self.assertIsNone(w.near_search)
            for preset in (m.CSL_PRESETS[0], m.CSL_PRESETS[5]):
                w.angle_slider.set_val(preset.angle_deg)
                self.assertEqual(w.common_cell.atoms, (66, 66))
                self.assertTrue(w.near_cell_artist.get_visible())
            w.angle_slider.set_val(39.5)
            self.assertIsNone(w.common_cell)
            self.assertFalse(w.cell_fit_button.get_active())
            self.assertFalse(w.near_cell_artist.get_visible())
            w.angle_slider.set_val(m.DEFAULT_ANGLE_DEG)
            w._toggle_near()
            w._toggle_near()
            self.assertTrue(w.near_cell_artist.get_visible())
            self.assertEqual(w.common_cell.max_strain, 0)
        finally:
            w._close_near()
            m.plt.close(w.figure)

    def test_qt_apply_pan_switch_and_disable(self):
        import tilt_gb_dichromatic_pattern_qt as q

        app = q.create_application()
        w = q.DichromaticPatternWindow(
            q.PatternParameters(angle_deg=39.5), worker_count=2
        )
        w.show()

        def wait_for(predicate):
            start = time.monotonic()
            while not predicate() and time.monotonic() - start < 25:
                app.processEvents()
                time.sleep(0.01)
            self.assertTrue(predicate(), w.near_info.toPlainText())

        try:
            self.assertFalse(w.near_enabled)
            self.assertIsNone(w.near_search)
            w.near_method_combo.setCurrentIndex(w.near_method_combo.findData("strain"))
            w.near_button.setChecked(True)
            wait_for(lambda: w.near_cell is not None and w.parallel_stage is None)
            self.assertFalse(np.allclose(w.deformations[0], np.eye(2)))
            self.assertGreater(sum(map(len, w.coincident_points)), 1)
            current = w.near_cell
            w._start_vector_measurement()
            w._handle_view_click(np.zeros(2))
            self.assertEqual(len(w.selected_atoms), 1)
            first_atom = w.selected_atoms[0]
            w.view_box.translateBy(x=100, y=-80)
            wait_for(lambda: w._buffer_contains_view() and w.parallel_stage is None)
            self.assertIs(w.selected_atoms[0], first_atom)
            self.assertIs(w.near_cell, current)  # cell search independent of panning
            w._queue_angle_update(58.5)
            wait_for(
                lambda: w.angle_deg == 58.5
                and w.near_cell is not None
                and w.parallel_stage is None
            )
            b1, b2 = n.bases(58.5)
            np.testing.assert_allclose(
                w.near_cell.f1 @ b1 @ w.near_cell.m1,
                w.near_cell.f2 @ b2 @ w.near_cell.m2,
                atol=1e-9,
            )
            # Reconfigure the shared pool while a Near-CSL job is in flight.
            w._queue_near_search()
            w.near_debounce_timer.stop()
            w._start_near_search()
            w._poll_near_search()
            w.worker_spin.setValue(1)
            w._poll_near_search()  # safe even before the replacement search exists
            wait_for(lambda: w.near_cell is not None and w.parallel_stage is None)
            w.near_button.setChecked(False)
            wait_for(lambda: w.parallel_stage is None)
            self.assertIsNone(w.near_cell)
            np.testing.assert_array_equal(w.deformations[0], np.eye(2))
            # Turn off while active; no stale solution may reapply strain.
            w.near_button.setChecked(True)
            w._start_near_search()
            w._poll_near_search()
            w.near_button.setChecked(False)
            for _ in range(30):
                app.processEvents()
                time.sleep(0.01)
            self.assertIsNone(w.near_cell)
        finally:
            w.close()

    def test_matplotlib_same_solution(self):
        import tilt_gb_dichromatic_pattern as m

        w = m.DichromaticPatternApp(m.PatternParameters(angle_deg=39.5), worker_count=1)
        try:
            self.assertIsNone(w.near_search)
            w._toggle_near()
            start = time.monotonic()
            while w.near_cell is None and time.monotonic() - start < 15:
                w._poll_near_search()
                time.sleep(0.01)
            self.assertIsNotNone(w.near_cell)
            np.testing.assert_allclose(w.near_cell.f1, solve(39.5)[0].f1)
            current = w.near_cell
            w._fit_near_cell()
            self.assertIs(w.near_cell, current)
            self.assertGreater(sum(map(len, w.coincident_points)), 1)
            w._toggle_near()
            self.assertIsNone(w.near_cell)
            np.testing.assert_array_equal(w.deformations[1], np.eye(2))
        finally:
            w._close_near()
            m.plt.close(w.figure)


if __name__ == "__main__":
    unittest.main()
