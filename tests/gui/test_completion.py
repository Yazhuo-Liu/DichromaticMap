"""Cell completion through real controls, previews, fitting and sessions."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from dichromatic_map import crystal
from dichromatic_map.ui.completion import CellCompletionDialog
from test_workflows import cell_corners, local_diamond, pick_exact_cell, pick_local_cell

pytestmark = pytest.mark.gui


def pick_fcc21_edge(gui):
    window = gui.window(lattice="FCC", axis="110", angle_deg=21, width=54, height=54)
    controls = window.controls
    gui.select_layer(window, 0)
    controls.near_section.toggle.setChecked(True)
    controls.near_button.click()
    gui.settle(window)
    geometry = window.state.geometry
    ends = np.array([
        crystal.rotation_matrix_2d(sign * 10.5) @ (np.array(vector) @ geometry.frame[:, :2] / 2)
        for sign, vector in zip((1, -1), ([21, -21, -42], [9, -9, -50]))
    ])
    controls.manual_section.toggle.setChecked(True)
    controls.manual_pick_button.click()
    gui.click_plot(window, [0, 0])
    assert not controls.manual_complete_button.isEnabled()
    gui.click_plot(window, ends.mean(axis=0))
    assert len(window.state.manual_vertices) == 2
    assert controls.manual_complete_button.isEnabled()
    assert controls.manual_strain_limit.isEnabled()
    assert controls.manual_rotation_limit.isEnabled()
    return window


def open_completion(gui, window):
    window.controls.manual_complete_button.click()
    gui.app.processEvents()
    dialog = window.findChild(CellCompletionDialog)
    assert dialog is not None and dialog.isVisible()
    return dialog


def test_symmetry_preview_accept_fit_session_and_restore(gui, tmp_path):
    window = pick_fcc21_edge(gui)
    controls, state = window.controls, window.state
    first_vertices = tuple(state.manual_vertices)
    original_atoms = [grain.positions.copy() for grain in state.grains]
    dialog = open_completion(gui, window)
    assert dialog.candidate_combo.count() == 2
    for index in (1, 0):
        dialog.candidate_combo.setCurrentIndex(index)
        assert "714 / 714" in dialog.details.toPlainText()
        assert "0.198524%" in dialog.details.toPlainText()
        assert dialog.use_button.isEnabled()
    expected = dialog.selected_candidate.vertices.copy()
    assert len(state.manual_vertices) == 2
    assert state.manual_strain_fit is None
    for grain, points in zip(state.grains, original_atoms):
        np.testing.assert_array_equal(grain.positions, points)
    dialog.use_button.click()
    gui.settle(window)
    assert state.interaction_mode == "idle"
    assert all(a is b for a, b in zip(state.manual_vertices[:2], first_vertices))
    assert [v.source for v in state.manual_vertices] == ["CSL", "local", "symmetry", "symmetry"]
    np.testing.assert_allclose(window._manual_grain_polygons(), expected, atol=1e-12)
    assert np.linalg.norm(expected[0, 2] - expected[1, 2]) > controls.local_distance_spin.value()
    np.testing.assert_array_equal(state.manual_counts.half_open, [[714, 0], [714, 0]])
    assert controls.manual_strain_button.isEnabled()
    assert not controls.manual_complete_button.isEnabled()

    controls.manual_strain_button.click()
    gui.settle(window)
    assert state.manual_strain_fit is not None
    assert state.manual_strain_fit.cell.max_strain == pytest.approx(0.00198524093639)
    output = tmp_path / "symmetry.dmap"
    window.save_session(output)
    restored = gui.window()
    restored.load_session(output)
    gui.settle(restored)
    restored.controls.manual_strain_button.click()
    gui.settle(restored)
    assert restored.state.manual_strain_fit is None
    np.testing.assert_allclose(restored._manual_grain_polygons(), expected, atol=1e-12)
    assert restored.state.manual_vertices[2].source == "symmetry"
    restored.controls.near_button.click()
    gui.settle(restored)
    assert restored.state.manual_vertices == []


def test_rejected_completion_and_cancel_preserve_partial_selection(gui):
    window = pick_fcc21_edge(gui)
    controls, state = window.controls, window.state
    picked = tuple(state.manual_vertices)
    controls.manual_strain_limit.setValue(0.1)
    dialog = open_completion(gui, window)
    assert not dialog.use_button.isEnabled()
    assert "above" in dialog.details.toPlainText()
    dialog.reject()
    gui.app.processEvents()
    assert len(state.manual_vertices) == 2
    assert all(a is b for a, b in zip(state.manual_vertices, picked))
    assert state.interaction_mode == "cell"
    assert controls.manual_pick_button.isChecked()

    controls.manual_strain_limit.setValue(2)
    dialog = open_completion(gui, window)
    assert dialog.use_button.isEnabled()
    # Programmatic changes while a modal preview is open cannot commit stale vertices.
    window._clear_manual_cell()
    dialog.use_button.click()
    gui.app.processEvents()
    assert state.manual_vertices == []
    assert "Selection changed" in controls.manual_info.toPlainText()


def test_three_vertices_complete_parallelogram_and_support_undo(gui, local_diamond):
    window = pick_local_cell(gui, local_diamond)
    expected = window._manual_grain_polygons().copy()
    controls = window.controls
    controls.manual_undo_button.click()
    assert len(window.state.manual_vertices) == 3
    assert controls.manual_complete_button.text() == "Complete parallelogram…"
    dialog = open_completion(gui, window)
    assert dialog.candidate_combo.count() == 1
    assert dialog.selected_candidate.source == "closure"
    assert dialog.use_button.isEnabled()
    dialog.use_button.click()
    gui.settle(window)
    np.testing.assert_allclose(window._manual_grain_polygons(), expected, atol=1e-12)
    assert window.state.manual_vertices[-1].source == "closure"
    np.testing.assert_array_equal(window.state.manual_counts.half_open, [[0, 40], [0, 40]])
    controls.manual_undo_button.click()
    assert len(window.state.manual_vertices) == 3
    assert window.state.interaction_mode == "cell"
    assert controls.manual_complete_button.isEnabled()


@pytest.mark.parametrize("lattice,axis", [("FCC", "110"), ("BCC", "100")])
@pytest.mark.parametrize("picked_count", [2, 3])
def test_exact_csl_completion_without_near_csl(gui, lattice, axis, picked_count):
    window = gui.window(lattice=lattice, axis=axis)
    controls, state = window.controls, window.state
    pick_exact_cell(gui, window)
    corners = cell_corners(window.common_cell)
    controls.manual_clear_button.click()
    # Neither enabling Near-CSL nor selecting its local method is required.
    controls.near_method_combo.setCurrentIndex(controls.near_method_combo.findData("strain"))
    assert not state.near_enabled
    assert not window.local_active
    controls.manual_pick_button.click()
    # A rectangular [110] cell edge parallel to a mirror has no independent
    # mirror image. Use its diagonal as the first edge for symmetry completion.
    points = corners[[0, 2]] if picked_count == 2 else corners[:3]
    for point in points:
        gui.click_plot(window, point)
    assert len(state.manual_vertices) == picked_count
    assert all(vertex.source == "CSL" for vertex in state.manual_vertices)
    assert controls.manual_complete_button.isEnabled()
    picked = tuple(state.manual_vertices)
    dialog = open_completion(gui, window)
    assert dialog.use_button.isEnabled()
    candidate = dialog.selected_candidate
    expected = candidate.vertices.copy()
    np.testing.assert_allclose(expected[0], expected[1], atol=1e-12)
    assert candidate.fit.cell.max_strain < 1e-12
    np.testing.assert_allclose(candidate.fit.rotations_deg, 0, atol=1e-12)
    dialog.use_button.click()
    gui.settle(window)
    assert len(state.manual_vertices) == 4
    assert all(a is b for a, b in zip(state.manual_vertices, picked))
    assert all(vertex.source == "CSL" for vertex in state.manual_vertices)
    np.testing.assert_allclose(window._manual_grain_polygons(), expected, atol=1e-12)
    np.testing.assert_array_equal(state.manual_counts.half_open[:, 0], candidate.atoms)
    assert not state.near_enabled
    assert state.manual_strain_fit is None
    assert not controls.manual_strain_button.isEnabled()
    assert "No strain is needed" in controls.manual_strain_note.text()
    controls.manual_undo_button.click()
    assert len(state.manual_vertices) == 3
    assert state.interaction_mode == "cell"
    assert controls.manual_complete_button.isEnabled()
