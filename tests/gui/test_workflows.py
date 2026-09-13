"""Regression tests for the current viewer's public controls and plot picking.

Historical scientific cases come from the FCC [110] 22-degree local diamond
and BCC [100] exact CSL workflows. Actions use the visible grain/layer controls;
the hidden compatibility widgets are deliberately excluded.
"""

import os
from pathlib import Path
from threading import Event

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from dichromatic_map import crystal, matching
from dichromatic_map.ui import controls as controls_module
from dichromatic_map.ui import window as window_module

pytestmark = pytest.mark.gui


def cell_corners(cell):
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ cell.cell.T


def plotted_points(item):
    return np.column_stack(item.getData())


@pytest.fixture
def local_diamond():
    """Four B-layer local pairs with a known 40-atom half-open count per grain."""
    grains = [
        crystal.projected_columns(25, 20, angle, lattice="FCC", axis="110")
        for angle in (11, -11)
    ]
    pairs = matching.local_near_pairs(*grains, distance=0.05)
    layer = np.flatnonzero(pairs.layers == 1)
    selected = [
        layer[np.argmin(np.linalg.norm(pairs.midpoints[layer] - target, axis=1))]
        for target in ([0, 5.6], [-2.5, 0], [0, -5.6], [2.5, 0])
    ]
    return pairs.midpoints[selected], np.stack(
        (pairs.first[selected], pairs.second[selected])
    )


def pick_exact_cell(gui, window):
    gui.select_layer(window, 0)
    window.controls.cell_check.click()
    window.controls.cell_fit_button.click()
    gui.settle(window)
    corners = cell_corners(window.common_cell)
    window.controls.manual_section.toggle.setChecked(True)
    window.controls.manual_pick_button.click()
    for point in corners:
        gui.click_plot(window, point)
    assert len(window.state.manual_vertices) == 4
    gui.settle(window)
    assert window.state.manual_counts is not None
    return corners


def pick_local_cell(gui, local_diamond, workers=1):
    window = gui.window(lattice="FCC", axis="110", angle_deg=22, workers=workers)
    controls = window.controls
    gui.select_layer(window, 1)
    controls.view_performance_section.toggle.setChecked(True)
    controls.scale_stop_buttons[3].click()  # 2x field includes all four vertices.
    controls.near_section.toggle.setChecked(True)
    controls.near_button.click()
    gui.settle(window)
    controls.manual_section.toggle.setChecked(True)
    controls.manual_pick_button.click()
    for index, point in enumerate(local_diamond[0]):
        gui.click_plot(window, point)
        assert len(window.state.manual_vertices) == index + 1
        assert controls.manual_strain_button.isEnabled() == (index == 3)
    gui.settle(window)
    return window


def test_orientation_custom_axis_and_independent_layer_controls(gui):
    window = gui.window(lattice="BCC", axis="100", angle_deg=0)
    controls = window.controls
    gui.select_layer(window, 0)
    assert window.state.visible_grain_layers == [{0}, {0}]
    assert len(window.plot.coincidence_items[0].points()) > 0
    controls.grain_layer_checks[0][0].click()
    assert window.state.visible_grain_layers == [set(), {0}]
    assert len(window.plot.coincidence_items[0].points()) == 0
    assert len(window.plot.grain_layer_items[1][0].points()) > 0
    controls.all_layers_button.click()
    assert window.state.visible_grain_layers == [{0, 1}, {0, 1}]

    controls.orientation_layers_tabs.setCurrentIndex(0)
    controls.structure_combo.setCurrentIndex(controls.structure_combo.findData("FCC"))
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData(None))
    controls.custom_axis_edit.setText("0 0 0")
    controls.custom_axis_apply.click()
    assert not controls.axis_error.isHidden()
    assert window.state.geometry.axis == "100"
    controls.custom_axis_edit.setText("1 -1 3")
    QtTest.QTest.keyClick(controls.custom_axis_edit, QtCore.Qt.Key_Return)
    gui.settle(window)
    assert controls.axis_error.isHidden()
    np.testing.assert_array_equal(window.state.geometry.axis_indices, [1, -1, 3])
    assert window.state.geometry.lattice == "FCC"
    assert len(controls.grain_layer_checks[0]) == window.state.geometry.layer_count
    controls.angle_spin.setValue(22)
    gui.settle(window)
    assert window.state.angle_deg == 22
    assert window.state.grain_signature == window._geometry_signature()
    assert "22.00000000" in controls.angle_exact_label.text()


def test_vector_and_boundary_picking_respect_display_rotation(gui):
    window = gui.window(lattice="BCC", axis="100", angle_deg=37)
    controls, state, plot = window.controls, window.state, window.plot
    gui.select_layer(window, 0, grains=(0,))
    controls.vector_button.click()
    first = state.grains[0].positions[state.grains[0].layers == 0]
    point = first[np.argmin(np.linalg.norm(first - [-1, -1], axis=1))]
    gui.click_plot(window, point)
    gui.select_layer(window, 0, grains=(1,))
    second = state.grains[1].positions[state.grains[1].layers == 0]
    point = second[np.argmin(np.linalg.norm(second - [1, 1], axis=1))]
    gui.click_plot(window, point)
    assert [atom.grain_index for atom in state.selected_atoms] == [0, 1]
    assert "G1 current" in plot.vector_annotation.toPlainText()
    assert "G2 current" in plot.vector_annotation.toPlainText()
    assert plot.vector_arrow.isVisible()
    frames = [value.current.copy() for _, value in window._selected_crystal_vectors()]
    displacement = window._selected_vector_displacement().copy()
    controls.axial_spin.setValue(2)
    np.testing.assert_allclose(
        window._selected_vector_displacement() - displacement,
        [0, 0, 2 * state.geometry.axial_period], atol=1e-12,
    )
    controls.axial_spin.setValue(0)
    controls.rotation_spin.setValue(47)
    for (_, actual), expected in zip(window._selected_crystal_vectors(), frames):
        np.testing.assert_allclose(actual.current, expected, atol=1e-12)
    np.testing.assert_allclose(
        plotted_points(plot.vector_item),
        plot._to_view([atom.position for atom in state.selected_atoms]), atol=1e-12,
    )
    assert state.angle_deg == 37

    gui.select_layer(window, 0, grains=(0,))
    controls.pick_gb_button.click()
    for target in ([0, 0], [1, 1]):
        grain = state.grains[0]
        candidates = grain.positions[grain.layers == 0]
        point = candidates[np.argmin(np.linalg.norm(candidates - target, axis=1))]
        gui.click_plot(window, point)
    assert len(state.selected_points) == 2
    assert not controls.gb_region_widget.isHidden()
    controls.all_layers_button.click()
    controls.first_bicrystal_button.click()
    assert window._region_states() == (True, False, False, True)
    boundary = np.array(state.selected_points)
    masks = [mask.copy() for mask in state.visible_atom_masks]
    controls.rotation_spin.setValue(-29)
    np.testing.assert_array_equal(state.selected_points, boundary)
    for actual, expected in zip(state.visible_atom_masks, masks):
        np.testing.assert_array_equal(actual, expected)
    controls.full_button.click()
    assert window._region_states() == (True, True, True, True)
    gui.key(window, QtCore.Qt.Key_R)
    gui.key(window, QtCore.Qt.Key_Escape)
    assert state.selected_points == []
    assert state.interaction_mode == "idle"


def test_exact_manual_count_navigation_filters_and_editing(gui):
    window = gui.window(lattice="BCC", axis="100")
    corners = pick_exact_cell(gui, window)
    controls, state, plot = window.controls, window.state, window.plot
    np.testing.assert_array_equal(state.manual_counts.half_open, [[5, 0], [5, 0]])
    assert not controls.manual_strain_button.isEnabled()
    original_counts = state.manual_counts
    controls.rotation_spin.setValue(37)
    plot.view_box.translateBy(x=30, y=-20)
    gui.settle(window)
    assert state.manual_counts is original_counts
    gui.select_layer(window, 1)
    assert state.manual_counts is original_counts
    np.testing.assert_allclose(
        plotted_points(plot.manual_cell_item),
        plot._to_view(np.vstack((corners, corners[0]))), atol=1e-12,
    )
    controls.manual_fit_button.click()
    gui.settle(window)
    gui.select_layer(window, 0)
    controls.pick_gb_button.click()
    gui.click_plot(window, corners[0])
    gui.click_plot(window, corners[1])
    controls.region_checks[2].click()
    controls.region_checks[3].click()
    controls.manual_visible_check.click()
    gui.settle(window)
    np.testing.assert_array_equal(state.manual_counts.half_open, [[5, 0], [0, 0]])
    controls.manual_visible_check.click()
    gui.settle(window)
    np.testing.assert_array_equal(state.manual_counts.half_open, [[5, 0], [5, 0]])
    controls.manual_undo_button.click()
    assert len(state.manual_vertices) == 3
    assert state.manual_counts is None
    assert state.interaction_mode == "cell"
    gui.key(window, QtCore.Qt.Key_Escape)
    assert len(state.manual_vertices) == 3
    gui.key(window, QtCore.Qt.Key_M)
    assert state.interaction_mode == "cell"
    controls.manual_clear_button.click()
    assert state.manual_vertices == []
    assert not plot.manual_annotation.isVisible()


def test_wrong_layer_manual_vertex_is_rejected_without_losing_first(gui):
    window = gui.window(lattice="BCC", axis="100", angle_deg=0)
    window.controls.manual_section.toggle.setChecked(True)
    window.controls.manual_pick_button.click()
    gui.click_plot(window, [0, 0])
    gui.click_plot(window, [0.5, 0.5])
    assert len(window.state.manual_vertices) == 1
    assert "Wrong layer/symbol" in window.controls.manual_info.toPlainText()
    gui.click_plot(window, [2, 0])
    assert len(window.state.manual_vertices) == 2
    assert all(vertex.layer == 0 for vertex in window.state.manual_vertices)


def test_local_cell_uses_actual_pair_polygons_and_restores_fit(gui, local_diamond):
    window = pick_local_cell(gui, local_diamond)
    controls, state, plot = window.controls, window.state, window.plot
    np.testing.assert_allclose(window._manual_grain_polygons(), local_diamond[1], atol=1e-12)
    np.testing.assert_array_equal(state.manual_counts.interior, [[0, 34], [0, 34]])
    np.testing.assert_array_equal(state.manual_counts.boundary, [[0, 14], [0, 14]])
    np.testing.assert_array_equal(state.manual_counts.half_open, [[0, 40], [0, 40]])
    assert "40 atoms · 34 inside + 5 edge + 1 corner" in plot.manual_annotation.toPlainText()
    original_vertices = state.manual_vertices
    original_polygons = window._manual_grain_polygons().copy()
    controls.manual_strain_button.click()
    assert state.manual_strain_fit is not None, controls.manual_strain_note.text()
    fit = state.manual_strain_fit
    assert controls.manual_strain_button.text() == "Restore original local structure"
    assert not controls.manual_pick_button.isEnabled()
    assert not controls.local_distance_spin.isEnabled()
    gui.settle(window)
    np.testing.assert_array_equal(state.manual_counts.half_open, [[0, 40], [0, 40]])
    for vertex in state.manual_vertices:
        assert vertex.source == "CSL"
        assert np.linalg.norm(state.coincident_points[1] - vertex.position, axis=1).min() < 1e-10
    details = controls.manual_strain_details.toPlainText()
    assert "G1 (blue)" in details and "G2 (red)" in details
    assert "Four-pair alignment" in details
    controls.rotation_spin.setValue(41)
    plot.view_box.translateBy(x=20, y=-16)
    gui.settle(window)
    assert state.manual_strain_fit is fit
    assert controls.manual_strain_details.toPlainText() == details
    controls.manual_strain_button.click()
    gui.settle(window)
    assert state.manual_strain_fit is None
    assert state.manual_vertices is original_vertices
    assert window.local_active
    assert controls.manual_pick_button.isEnabled()
    assert controls.manual_strain_details.isHidden()
    assert controls.manual_strain_details.toPlainText() == ""
    np.testing.assert_allclose(window._manual_grain_polygons(), original_polygons, atol=1e-12)
    np.testing.assert_array_equal(state.manual_counts.half_open, [[0, 40], [0, 40]])


def test_rejected_strain_and_threshold_changes_keep_consistent_state(gui, local_diamond):
    window = pick_local_cell(gui, local_diamond)
    controls, state = window.controls, window.state
    original_vertices = state.manual_vertices
    original_atoms = [grain.positions.copy() for grain in state.grains]
    controls.manual_strain_limit.setValue(0.1)
    controls.manual_strain_button.click()
    assert state.manual_strain_fit is None
    assert state.manual_vertices is original_vertices
    assert "above" in controls.manual_strain_note.text()
    for grain, expected in zip(state.grains, original_atoms):
        np.testing.assert_array_equal(grain.positions, expected)
    controls.local_distance_spin.setValue(0.06)
    gui.settle(window)
    assert state.manual_vertices == []
    assert state.manual_counts is None
    assert not controls.manual_strain_button.isEnabled()


def test_late_count_result_cannot_restore_a_cleared_cell(gui, monkeypatch):
    window = gui.window(lattice="BCC", axis="100")
    gui.select_layer(window, 0)
    window.controls.cell_check.click()
    window.controls.cell_fit_button.click()
    gui.settle(window)
    started, release = Event(), Event()
    actual_count = window_module.cell_count_worker

    def delayed_count(*args):
        started.set()
        assert release.wait(10), "Test did not release the count worker"
        return actual_count(*args)

    monkeypatch.setattr(window_module, "cell_count_worker", delayed_count)
    window.controls.manual_section.toggle.setChecked(True)
    window.controls.manual_pick_button.click()
    try:
        for point in cell_corners(window.common_cell):
            gui.click_plot(window, point)
        gui.wait_for(window, started.is_set)
        assert window.compute.manual_count_future is not None
        window.controls.manual_clear_button.click()
        release.set()
        gui.settle(window)
        assert window.state.manual_vertices == []
        assert window.state.manual_counts is None
        assert "No manual cell" in window.controls.manual_info.toPlainText()
        assert not window.plot.manual_annotation.isVisible()
    finally:
        release.set()


def test_spawned_updates_publish_only_the_latest_geometry(gui):
    window = gui.window(lattice="FCC", axis="110", workers=2)
    assert window.compute.worker_count == 2
    controls = window.controls
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData("111"))
    assert window.compute.parallel_stage == "grains"
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData("100"))
    controls.angle_spin.setValue(17)
    gui.settle(window)
    assert window.state.geometry.axis == "100"
    assert window.state.angle_deg == 17
    assert window.state.grain_signature == window._geometry_signature()
    assert len(window.plot.grain_layer_items[0]) == window.state.geometry.layer_count
    assert window.compute.worker_process_ids - {os.getpid()}
    assert window.compute.worker_count == 2  # A silent serial fallback is a failure.
    for grain, sign in zip(window.state.grains, (1, -1)):
        reference = crystal.get_geometry("FCC", "100")
        planar = (0.5 * grain.half_indices) @ reference.frame[:, :2]
        expected = planar @ crystal.rotation_matrix_2d(sign * 8.5).T
        np.testing.assert_allclose(grain.positions, expected, atol=1e-12)


def test_gui_export_dialog_writes_plot_png_and_theme_icons(gui, monkeypatch, tmp_path):
    window = gui.window(lattice="BCC", axis="100")
    pick_exact_cell(gui, window)
    window.controls.rotation_spin.setValue(32)
    gui.settle(window)
    output = tmp_path / "plots" / "pattern.png"
    calls = []

    def choose_filename(*args):
        calls.append(args)
        return str(output), "PNG image (*.png)"

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", choose_filename)
    export = next(
        button for button in window.findChildren(QtWidgets.QPushButton)
        if button.text() == "Export plot as PNG…"
    )
    export.click()
    assert calls[0][1:] == (
        "Export dichromatic pattern", "dichromatic_pattern.png", "PNG image (*.png)"
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    rendered = QtGui.QImage(str(output))
    assert not rendered.isNull()
    assert rendered.width() == 1800 and rendered.height() > 500
    assert output.stat().st_size > 10000
    resources = Path(controls_module.__file__).parent / "resources" / "icons"
    for name in ("chevron-down.svg", "chevron-up.svg"):
        path = resources / name
        assert path.as_posix() in gui.app.styleSheet()
        assert not QtGui.QIcon(str(path)).pixmap(12, 12).isNull()


def test_automatic_strain_candidate_applies_and_disable_restores_grains(gui):
    window = gui.window(lattice="FCC", axis="110", angle_deg=39.5)
    controls, state = window.controls, window.state
    controls.near_section.toggle.setChecked(True)
    controls.near_method_combo.setCurrentIndex(
        controls.near_method_combo.findData("strain")
    )
    assert not controls.strain_spin.isEnabled()
    controls.near_button.click()
    gui.settle(window)
    assert state.near_solutions, controls.near_info.toPlainText()
    assert controls.near_combo.currentIndex() == 0
    assert state.near_cell is state.near_solutions[0]
    np.testing.assert_allclose(state.deformations, [state.near_cell.f1, state.near_cell.f2])
    assert not np.allclose(state.deformations, [np.eye(2), np.eye(2)])
    assert "Green-Lagrange" in controls.near_info.toPlainText()
    assert len(state.near_solutions) > 1
    controls.near_combo.setCurrentIndex(1)
    gui.settle(window)
    assert state.near_cell is state.near_solutions[1]
    np.testing.assert_allclose(state.deformations, [state.near_cell.f1, state.near_cell.f2])
    controls.orientation_layers_tabs.setCurrentIndex(1)
    controls.cell_check.click()
    controls.cell_fit_button.click()
    gui.settle(window)
    assert len(window.plot.near_cell_item.getData()[0]) == 5
    controls.near_button.click()
    gui.settle(window)
    assert not state.near_enabled
    assert state.near_cell is None
    np.testing.assert_array_equal(state.deformations, [np.eye(2), np.eye(2)])
    np.testing.assert_array_equal(state.translations, np.zeros((2, 2)))
    assert state.angle_deg == 39.5


def test_axis_change_discards_pending_selected_strain_results(gui, local_diamond):
    window = pick_local_cell(gui, local_diamond, workers=2)
    controls, state = window.controls, window.state
    controls.manual_strain_button.click()
    assert state.manual_strain_fit is not None
    assert window.compute.parallel_stage == "grains"
    controls.orientation_layers_tabs.setCurrentIndex(0)
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData("111"))
    gui.settle(window)
    assert state.geometry.axis == "111"
    assert state.geometry.layer_count == 3
    assert state.grain_signature == window._geometry_signature()
    assert state.manual_strain_fit is None
    assert state.manual_vertices == []
    assert state.manual_counts is None
    assert controls.manual_strain_details.isHidden()
    assert controls.manual_strain_details.toPlainText() == ""
    np.testing.assert_array_equal(state.deformations, [np.eye(2), np.eye(2)])
    np.testing.assert_array_equal(state.translations, np.zeros((2, 2)))
    assert window.compute.worker_count == 2
    assert window.compute.worker_process_ids - {os.getpid()}
