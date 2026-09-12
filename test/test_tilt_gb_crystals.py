"""Regressions for normalized FCC/BCC integer-axis geometry and the Qt viewer.

Run with the conda base Python; Qt runs offscreen and no GBClaw imports are used.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tilt-test-cache")

from functools import lru_cache
from itertools import product
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import dichromatic_map.cells as cell_ops
import dichromatic_map.compute as compute
import dichromatic_map.crystal as crystal
import dichromatic_map.matching as matching
import dichromatic_map.state as state
import dichromatic_map.strain as strain_ops
import dichromatic_map.ui.controls as view_controls
import dichromatic_map.ui.window as view_window


MODELS = tuple(product(("FCC", "BCC"), ("110", "100")))
GENERAL_MODELS = tuple(
    product(("FCC", "BCC"), ("100", "110", "111", "112", "1 -1 3", "2 3 5", "0 0 1"))
)


def rotation(angle):
    t = np.deg2rad(angle)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


@lru_cache(maxsize=8)
def solve(lattice, axis, angle):
    options = dict(lattice=lattice, axis=axis)
    i, j = strain_ops.candidate_vectors(angle, 2.0, 12, **options)
    return strain_ops.pareto_cells(
        strain_ops.solve_cells_chunk(angle, 2.0, i, j, 0, len(i), **options)[1]
    )


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
                self.assertEqual((geometry.x_label, geometry.y_label), ("x", "y"))
            self.assertIs(
                crystal.get_geometry(lattice, "[2, 2, 4]"),
                crystal.get_geometry(lattice, "112"),
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
        from dichromatic_map.compute import coincidence_layers_worker

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
            batch = coincidence_layers_worker(first, first, 1e-6, range(count))[1]
            for phase, points in batch:
                np.testing.assert_array_equal(points, correct[phase])

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

    def test_parallel_latest_crystal_request_wins(self):
        search = compute.NearSearch(workers=2)
        try:
            search.request(39.5, 2.0, 12, lattice="FCC", axis="110")
            search.poll()
            search.request(37.2, 2.0, 12, lattice="BCC", axis="100")
            deadline = time.monotonic() + 30
            result = None
            while result is None and time.monotonic() < deadline:
                result = search.poll()
                self.assertLessEqual(len(search.running), 2)
                time.sleep(0.01)
            self.assertIsNotNone(result, search.error)
            expected = solve("BCC", "100", 37.2)
            self.assertEqual(
                [cell.atoms for cell in result], [cell.atoms for cell in expected]
            )
            for cell, reference in zip(result, expected):
                self.assertEqual((cell.lattice, cell.axis), ("BCC", "100"))
                np.testing.assert_allclose(cell.f1, reference.f1, atol=1e-9)
            search.cancel()
            self.assertFalse(search.busy)
            self.assertIsNone(search.poll())
        finally:
            search.close()


class QtCrystalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.app = view_controls.create_application()

    def wait_for(self, window, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate(), window.near_info.toPlainText())

    def test_control_sections_are_ordered_and_contextual(self):
        w = view_window.DichromaticPatternWindow(
            state.PatternParameters(), worker_count=1
        )
        w.show()
        self.app.processEvents()
        try:
            self.assertEqual(w.orientation_layers_tabs.currentIndex(), 0)
            self.assertTrue(w.interaction_section.isExpanded())
            self.assertFalse(w.view_performance_section.isExpanded())
            self.assertFalse(w.near_section.isExpanded())
            self.assertFalse(w.manual_section.isExpanded())
            self.assertTrue(all(check.isChecked() for check in w.layer_checks))
            self.assertFalse(w.cell_check.isChecked())
            self.assertLess(w.orientation_layers_tabs.y(), w.interaction_section.y())

            hidden_layer = 0
            w.layer_checks[hidden_layer].setChecked(False)
            for grain, mask in zip(w.grains, w.visible_atom_masks, strict=True):
                self.assertFalse(np.any(mask & (grain.layers == hidden_layer)))
            self.assertEqual(len(w.coincidence_items[hidden_layer].points()), 0)
            target = w.grains[0].positions[w.grains[0].layers == hidden_layer][0]
            atom, _distance = w._nearest_atom(target)
            self.assertTrue(atom is None or atom.layer != hidden_layer)
            w.layer_checks[hidden_layer].setChecked(True)
            self.assertTrue(w.gb_region_widget.isHidden())
            self.assertTrue(w.vector_options_widget.isHidden())

            w._start_new_boundary()
            w._handle_view_click(np.zeros(2))
            self.assertTrue(w.gb_region_widget.isHidden())
            first = w.selected_points[0]
            second = w.grains[0].positions[
                np.argmax(np.linalg.norm(w.grains[0].positions - first, axis=1))
            ]
            w._handle_view_click(w._to_view(second))
            self.assertEqual(len(w.selected_points), 2)
            self.assertFalse(w.gb_region_widget.isHidden())

            w._clear_lattice_selections()
            w._start_vector_measurement()
            self.assertFalse(w.vector_options_widget.isHidden())
            self.assertEqual(w.axial_spin.value(), 0)
            self.assertIn("does not add thickness", w.axial_spin.toolTip())
        finally:
            w.close()

    def test_local_matching_no_strain_parallel_threshold_pan_and_visibility(self):
        w = view_window.DichromaticPatternWindow(
            state.PatternParameters(lattice="BCC", axis="100", angle_deg=53.12),
            worker_count=2,
        )
        w.show()
        self.app.processEvents()
        try:
            w._start_parallel_regeneration(True)
            self.wait_for(w, lambda: w.parallel_stage is None)
            original = [grain.positions.copy() for grain in w.grains]
            self.assertEqual(w.near_method, "local")
            self.assertFalse(w.near_enabled)
            self.assertIsNone(w.local_thread_executor)
            w._start_vector_measurement()
            w._handle_view_click(np.zeros(2))
            picked = w.selected_atoms[0]
            w.near_button.setChecked(True)
            self.wait_for(w, lambda: not w.local_updating and w.parallel_stage is None)
            self.assertIsNone(w.near_search)
            self.assertIsNone(w.near_cell)
            self.assertIsNone(w.common_cell)  # approximate pairs do not create a cell
            self.assertGreater(len(w.local_pairs.layers), 20)
            for grain, positions, f in zip(w.grains, original, w.deformations):
                np.testing.assert_array_equal(grain.positions, positions)
                np.testing.assert_array_equal(f, np.eye(2))
            self.assertIs(w.selected_atoms[0], picked)
            reference = matching.local_near_pairs(
                *w.grains, w.local_distance_spin.value()
            )
            np.testing.assert_array_equal(w.local_pairs.first, reference.first)
            np.testing.assert_array_equal(w.local_pairs.second, reference.second)
            hidden_layer = int(w.local_pairs.layers[0])
            w.layer_checks[hidden_layer].setChecked(False)
            self.assertFalse(
                np.any(w._local_pair_mask() & (w.local_pairs.layers == hidden_layer))
            )
            self.assertEqual(
                len(w.local_match_item.points()), np.count_nonzero(w._local_pair_mask())
            )
            w.layer_checks[hidden_layer].setChecked(True)
            w.local_distance_spin.setValue(0.0005)
            self.assertEqual(
                len(w.local_pairs.layers), 0
            )  # old cutoff overlay clears immediately
            self.wait_for(w, lambda: not w.local_updating and w.parallel_stage is None)
            self.assertLess(len(w.local_pairs.layers), len(reference.layers))
            self.assertTrue(np.all(w.local_pairs.distances <= 0.0005))
            self.assertIs(w.selected_atoms[0], picked)
            w.local_distance_spin.setValue(0.05)
            w._start_near_search()
            self.assertEqual(w.parallel_stage, "local")
            w.view_box.translateBy(x=8, y=-7)
            self.wait_for(
                w,
                lambda: w._buffer_contains_view()
                and not w.local_updating
                and w.parallel_stage is None,
            )
            self.assertIs(w.selected_atoms[0], picked)
            w.worker_spin.setValue(1)
            self.wait_for(w, lambda: not w.local_updating and w.parallel_stage is None)
            self.assertIsNotNone(w.local_thread_executor)
            reference = matching.local_near_pairs(*w.grains, 0.05)
            np.testing.assert_array_equal(w.local_pairs.first, reference.first)
            # Both original endpoints must be visible; testing just their
            # midpoint on the boundary would incorrectly keep this pair.
            center = w._view_geometry()[0]
            w.local_pairs = matching.LocalPairs(
                np.array([center + [-0.02, 0]]),
                np.array([center + [0.02, 0]]),
                np.array([0]),
            )
            w.selected_points = [center + [0, -1], center + [0, 1]]
            w._set_region_states((True, True, True, True))
            self.assertEqual(len(w.local_match_item.points()), 1)
            w._set_region_states((False, True, True, True))
            self.assertEqual(len(w.local_match_item.points()), 0)
            w._set_region_states((True, True, True, True))
            w.layer_combo.setCurrentIndex(w.layer_combo.findData(1))
            self.assertEqual(len(w.local_match_item.points()), 0)
            # A completed but unpolled old job must not reappear after Off.
            w._start_local_matching()
            w.near_button.setChecked(False)
            self.wait_for(w, lambda: w.parallel_stage is None)
            for _ in range(10):
                self.app.processEvents()
            self.assertEqual(len(w.local_pairs.layers), 0)
            self.assertEqual(len(w.local_match_item.points()), 0)
            self.assertIsNone(w.near_cell)
        finally:
            w.close()

    def test_local_strain_method_switch_stale_jobs_and_custom_axis(self):
        w = view_window.DichromaticPatternWindow(
            state.PatternParameters(angle_deg=39.5), worker_count=2
        )
        w.show()
        self.app.processEvents()
        try:
            w.near_method_combo.setCurrentIndex(w.near_method_combo.findData("strain"))
            w.near_button.setChecked(True)
            self.wait_for(
                w, lambda: w.near_cell is not None and w.parallel_stage is None
            )
            self.assertFalse(np.allclose(w.deformations[0], np.eye(2)))
            w._start_vector_measurement()
            w._handle_view_click(np.zeros(2))
            w.near_method_combo.setCurrentIndex(w.near_method_combo.findData("local"))
            self.assertEqual(
                w.selected_atoms, []
            )  # switching strain really moves atoms
            self.wait_for(w, lambda: not w.local_updating and w.parallel_stage is None)
            for f in w.deformations:
                np.testing.assert_array_equal(f, np.eye(2))
            self.assertIsNone(w.near_cell)
            w._start_local_matching()
            w.near_method_combo.setCurrentIndex(w.near_method_combo.findData("strain"))
            w.near_debounce_timer.stop()
            w._start_near_search()
            w._poll_near_search()
            # Return to local while homogeneous strain work is still active.
            w.near_method_combo.setCurrentIndex(w.near_method_combo.findData("local"))
            w.axis_combo.setCurrentIndex(w.axis_combo.findData(None))
            w.custom_axis_edit.setText("1 -1 3")
            w.custom_axis_apply.click()
            self.wait_for(w, lambda: not w.local_updating and w.parallel_stage is None)
            self.assertEqual(w.geometry.layer_count, 11)
            self.assertIsNone(w.near_cell)
            reference = matching.local_near_pairs(
                *w.grains, w.local_distance_spin.value()
            )
            np.testing.assert_array_equal(w.local_pairs.layers, reference.layers)
            np.testing.assert_array_equal(w.local_pairs.first, reference.first)
            # At the narrowest view and largest cutoff, keep a 2*d query halo.
            w.local_distance_spin.setValue(0.5)
            w.view_slider.setValue(10)
            self.wait_for(
                w,
                lambda: w._buffer_contains_view()
                and not w.local_updating
                and w.parallel_stage is None,
            )
            bx0, bx1, by0, by1 = w.buffer_bounds
            x0, x1, y0, y1 = w._view_range()
            self.assertGreaterEqual(
                min(x0 - bx0, bx1 - x1, y0 - by0, by1 - y1), 1.0 - 1e-10
            )
        finally:
            w.close()

    def test_lattice_constant_does_not_rescale_coordinates_or_cells(self):
        for lattice, axis in MODELS:
            windows = []
            try:
                for a0 in (2.8, 4.0):
                    w = view_window.DichromaticPatternWindow(
                        state.PatternParameters(
                            lattice_constant=a0, lattice=lattice, axis=axis
                        ),
                        worker_count=1,
                    )
                    windows.append(w)
                first, second = windows
                self.assertEqual(first.parameters.width, 12.0)
                self.assertEqual(first.parameters.height, 9.0)
                self.assertEqual(first._view_range(), second._view_range())
                for g1, g2 in zip(first.grains, second.grains):
                    np.testing.assert_array_equal(g1.half_indices, g2.half_indices)
                    np.testing.assert_allclose(g1.positions, g2.positions, atol=1e-12)
                for points1, points2 in zip(
                    first.coincident_points, second.coincident_points
                ):
                    np.testing.assert_allclose(points1, points2, atol=1e-12)
                if first.common_cell is not None:
                    for data1, data2 in zip(
                        first.near_cell_item.getData(), second.near_cell_item.getData()
                    ):
                        np.testing.assert_allclose(data1, data2, atol=1e-12)
            finally:
                for w in windows:
                    w.close()

    def test_custom_axis_multilayer_parallel_filter_and_invalid_input(self):
        w = view_window.DichromaticPatternWindow(
            state.PatternParameters(), worker_count=2
        )
        w.show()
        self.app.processEvents()
        try:
            self.assertEqual(w.plot_item.getAxis("bottom").labelText, "x / a₀")
            self.assertEqual(w.plot_item.getAxis("left").labelText, "y / a₀")
            for axis, count in (("111", 3), ("112", 6)):
                w.axis_combo.setCurrentIndex(w.axis_combo.findData(axis))
                self.wait_for(w, lambda: w.parallel_stage is None)
                self.assertEqual(w.geometry.layer_count, count)
                self.assertEqual(len(w.layer_checks), count)
                self.assertTrue(all(check.isChecked() for check in w.layer_checks))
                self.assertEqual(len(w.coincident_points), count)
                self.assertEqual(len(w.grain_layer_items[0]), count)
            w.axis_combo.setCurrentIndex(w.axis_combo.findData(None))
            w.custom_axis_edit.setText("2 -2 6")
            w.custom_axis_apply.click()
            self.wait_for(w, lambda: w.parallel_stage is None)
            self.assertEqual(w.geometry.axis, "1 -1 3")
            self.assertEqual(w.geometry.layer_count, 11)
            expected = matching.same_layer_coincidence_sites(
                *w.grains, matching.COINCIDENCE_TOLERANCE_FACTOR
            )
            for actual, reference in zip(w.coincident_points, expected, strict=True):
                np.testing.assert_allclose(actual, reference, atol=1e-12)
            w.layer_checks[0].setChecked(False)
            w.layer_checks[5].setChecked(False)
            self.assertEqual(w.visible_layers, set(range(11)) - {0, 5})
            for grain, mask in zip(w.grains, w.visible_atom_masks):
                self.assertTrue(np.any(mask))
                self.assertFalse(np.any(np.isin(grain.layers[mask], (0, 5))))
            for layer, item in enumerate(w.grain_layer_items[0]):
                self.assertEqual(len(item.points()) == 0, layer in (0, 5))
            # Even clicking a hidden phase can only pick a visible phase.
            hidden_position = w.grains[0].positions[w.grains[0].layers == 0][0]
            atom, _ = w._nearest_atom(hidden_position)
            self.assertIsNotNone(atom)
            self.assertNotIn(atom.layer, (0, 5))
            w._start_vector_measurement()
            positions = w.grains[0].positions[w.grains[0].layers == 10]
            selected_position = positions[np.argmin(np.linalg.norm(positions, axis=1))]
            w._handle_view_click(selected_position)
            self.assertEqual(len(w.selected_atoms), 1)
            picked = w.selected_atoms[0]
            w.view_box.translateBy(x=7, y=-6)
            self.wait_for(
                w, lambda: w._buffer_contains_view() and w.parallel_stage is None
            )
            self.assertIs(w.selected_atoms[0], picked)
            for invalid in ("0 0 0", "15 16 1", "1.5 1 1"):
                w.custom_axis_edit.setText(invalid)
                w.custom_axis_apply.click()
                self.assertEqual(w.geometry.axis, "1 -1 3")
                self.assertTrue(w.axis_error.text())
            w.custom_axis_edit.setText("1 -1 3")
            w.structure_combo.setCurrentIndex(w.structure_combo.findData("BCC"))
            self.wait_for(w, lambda: w.parallel_stage is None)
            self.assertEqual((w.geometry.lattice, w.geometry.axis), ("BCC", "1 -1 3"))
            self.assertEqual(w.selected_atoms, [])
            self.assertEqual(w.selected_layer, -1)
            w.near_method_combo.setCurrentIndex(w.near_method_combo.findData("strain"))
            w.near_button.setChecked(True)
            # Switch while the previous-axis Near-CSL work is in flight.
            w.near_debounce_timer.stop()
            w._start_near_search()
            w._poll_near_search()
            w.axis_combo.setCurrentIndex(w.axis_combo.findData("112"))
            self.wait_for(
                w, lambda: w.near_cell is not None and w.parallel_stage is None
            )
            self.assertEqual(w.near_cell.axis, "112")
            self.assertEqual(len(w.coincident_points), 6)
            for f in w.deformations:
                np.testing.assert_allclose(f, f.T, atol=1e-12)
        finally:
            w.close()

    def test_axial_vector_readout_uses_the_selected_crystal_repeat(self):
        expected = {
            ("FCC", "110"): ("[1 1 2]", np.sqrt(1.5)),
            ("BCC", "110"): ("[1 1 1]", np.sqrt(3)),
            ("FCC", "100"): ("[1 0 1]", np.sqrt(2)),
            ("BCC", "100"): ("[1 0 1]", np.sqrt(2)),
        }
        for lattice, axis in MODELS:
            w = view_window.DichromaticPatternWindow(
                state.PatternParameters(angle_deg=0, lattice=lattice, axis=axis),
                worker_count=1,
            )
            try:
                w.selected_atoms = [
                    state.SelectedAtom(np.array([0.0, 0.0]), 0, 0, np.array([0, 0, 0])),
                    state.SelectedAtom(np.array([0.0, 1.0]), 0, 0, np.array([0, 0, 2])),
                ]
                w.axial_repeat = 1
                readout = w._selected_vector_readout()
                indices, magnitude = expected[lattice, axis]
                self.assertIn(indices, readout)
                self.assertIn(f"{magnitude:.4f}", readout)
                w.selected_atoms[1] = state.SelectedAtom(
                    np.array([1.0, 0.0]), 1, 0, np.array([0, 0, 0])
                )
                self.assertIn("(1.0000, 0.0000)", w._selected_vector_readout())
            finally:
                w.close()

    def test_live_model_switch_stale_jobs_hidden_atoms_and_pan(self):
        w = view_window.DichromaticPatternWindow(
            state.PatternParameters(angle_deg=39.5), worker_count=2
        )
        w.show()
        self.app.processEvents()
        try:
            w.near_method_combo.setCurrentIndex(w.near_method_combo.findData("strain"))
            w.near_button.setChecked(True)
            w.near_debounce_timer.stop()
            w._start_near_search()
            w._poll_near_search()
            w.structure_combo.setCurrentIndex(w.structure_combo.findData("BCC"))
            w.axis_combo.setCurrentIndex(w.axis_combo.findData("100"))
            self.wait_for(
                w, lambda: w.near_cell is not None and w.parallel_stage is None
            )
            self.assertEqual((w.geometry.lattice, w.geometry.axis), ("BCC", "100"))
            self.assertEqual((w.near_cell.lattice, w.near_cell.axis), ("BCC", "100"))
            self.assertAlmostEqual(w.angle_deg, crystal.csl_angle_deg(3, 1, "100"))
            for grain in w.grains:
                parity = grain.half_indices % 2
                self.assertTrue(np.all(parity == parity[:, :1]))
            w._start_vector_measurement()
            w._handle_view_click(np.zeros(2))
            self.assertEqual(len(w.selected_atoms), 1)
            picked = w.selected_atoms[0]
            w.view_box.translateBy(x=10, y=-8)
            self.wait_for(
                w, lambda: w._buffer_contains_view() and w.parallel_stage is None
            )
            self.assertIs(w.selected_atoms[0], picked)
            w.selected_points = [np.array([-1.0, 0]), np.array([1.0, 0])]
            w._set_region_states((True, True, False, False))
            self.assertFalse(np.any(w.visible_atom_masks[1]))
            atom, _distance = w._nearest_atom(w.grains[1].positions[0])
            self.assertIsNotNone(atom)
            self.assertEqual(atom.grain_index, 0)
            # Changing the model invalidates reference Miller indices and jobs.
            w.axis_combo.setCurrentIndex(w.axis_combo.findData("110"))
            self.assertEqual(w.selected_atoms, [])
            w.near_button.setChecked(False)
            self.wait_for(w, lambda: w.parallel_stage is None)
            for _ in range(30):
                self.app.processEvents()
                time.sleep(0.01)
            self.assertIsNone(w.near_cell)
            self.assertEqual(
                (w.common_cell.lattice, w.common_cell.axis), ("BCC", "110")
            )
            np.testing.assert_array_equal(w.deformations[0], np.eye(2))
        finally:
            w.close()

    def test_strain_change_rejects_old_positions_but_pan_remains_pickable(self):
        w = view_window.DichromaticPatternWindow(
            state.PatternParameters(angle_deg=39.5), worker_count=2
        )
        w.show()
        self.app.processEvents()
        try:
            w.near_method_combo.setCurrentIndex(w.near_method_combo.findData("strain"))
            w.near_button.setChecked(True)
            self.wait_for(
                w, lambda: w.near_cell is not None and w.parallel_stage is None
            )
            self.assertGreaterEqual(len(w.near_solutions), 2)
            old_positions = w.grains[0].positions
            old_point = old_positions[
                np.argmin(np.linalg.norm(old_positions - [1.0, 1.0], axis=1))
            ].copy()
            old_deformation = w.deformations[0].copy()
            w._select_near_cell(1)
            self.assertFalse(np.allclose(old_deformation, w.deformations[0]))
            # Do not dispatch events: the old atoms are still on the canvas,
            # while the desired strain and asynchronously requested atoms differ.
            self.assertEqual(w.parallel_stage, "grains")
            w._start_vector_measurement()
            w._handle_view_click(old_point)
            self.assertEqual(w.selected_atoms, [])
            self.assertIsNone(w._nearest_atom(old_point)[0])

            self.wait_for(w, lambda: w.parallel_stage is None)
            positions = w.grains[0].positions
            current_point = positions[
                np.argmin(np.linalg.norm(positions - [1.0, 1.0], axis=1))
            ].copy()
            w._handle_view_click(current_point)
            self.assertEqual(len(w.selected_atoms), 1)

            # A pan requests a new crop, not a new crystal configuration.
            # Picking the first endpoint must remain possible while it computes.
            w._start_vector_measurement()
            w.view_box.translateBy(x=0.25, y=-0.35)
            w.view_refresh_timer.stop()
            w._start_parallel_regeneration(compute_coincidences=True)
            self.assertEqual(w.parallel_stage, "grains")
            w._handle_view_click(current_point)
            self.assertEqual(len(w.selected_atoms), 1)
            picked = w.selected_atoms[0]
            self.wait_for(w, lambda: w.parallel_stage is None)
            self.assertIs(w.selected_atoms[0], picked)
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
