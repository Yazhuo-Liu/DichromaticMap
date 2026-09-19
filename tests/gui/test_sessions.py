"""Session round trips through the viewer, including active analysis state."""

from threading import Event
from zipfile import ZipFile

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets

from dichromatic_map import crystal
from dichromatic_map.ui import window as window_module
from test_workflows import cell_corners, local_diamond, pick_exact_cell, pick_local_cell

pytestmark = pytest.mark.gui


def test_session_restores_cell_vector_boundary_and_display_filters(gui, tmp_path):
    window = gui.window(lattice="BCC", axis="100", lattice_constant=2.87)
    controls, state = window.controls, window.state
    corners = pick_exact_cell(gui, window)
    controls.pick_gb_button.click()
    gui.click_plot(window, corners[0])
    gui.click_plot(window, corners[1])
    controls.vector_button.click()
    for grain, point in enumerate(corners[:2]):
        gui.select_layer(window, 0, grains=(grain,))
        gui.click_plot(window, point)
    assert [atom.grain_index for atom in state.selected_atoms] == [0, 1]
    controls.grain_layer_checks[0][0].click()
    controls.grain_layer_checks[1][1].click()
    controls.region_checks[2].click()
    controls.region_checks[3].click()
    controls.manual_visible_check.click()
    controls.axial_spin.setValue(2)
    controls.rotation_spin.setValue(31)
    controls.reference_axes_check.setChecked(False)
    window.plot.view_box.translateBy(x=1.25, y=-0.75)
    gui.settle(window)

    expected_angle = state.angle_deg
    expected_range = np.array(window.plot.view_box.viewRange())
    expected_points = np.array(state.selected_points)
    expected_atoms = list(state.selected_atoms)
    expected_polygons = window._manual_grain_polygons().copy()
    expected_count = state.manual_counts.half_open.copy()
    expected_vector = window._selected_vector_displacement().copy()
    expected_readout = window._selected_vector_readout()
    expected_regions = window._region_states()
    expected_layers = [set(layers) for layers in state.visible_grain_layers]
    assert np.any(expected_count)
    assert expected_layers == [{0}, {0, 1}]
    output = tmp_path / "boundary.dmap"
    window.save_session(output)
    with ZipFile(output) as archive:
        assert {"session.json", "counts.csv", "vectors.csv", "strain.csv", "README.txt"} <= set(archive.namelist())

    # Import into this same window after changing the crystal and its layer count.
    controls.structure_combo.setCurrentIndex(controls.structure_combo.findData("SC"))
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData("111"))
    gui.settle(window)
    assert window.state.manual_vertices == []
    window.load_session(output)
    gui.settle(window)
    state = window.state
    assert state.geometry.lattice == "BCC"
    assert state.geometry.axis == "100"
    assert state.angle_deg == expected_angle
    assert state.parameters.lattice_constant == 2.87
    assert controls.structure_combo.currentData() == "BCC"
    assert controls.axis_combo.currentData() == "100"
    assert state.visible_grain_layers == expected_layers
    assert window._region_states() == expected_regions
    assert controls.manual_visible_check.isChecked()
    np.testing.assert_allclose(state.selected_points, expected_points, atol=1e-12)
    np.testing.assert_allclose(window._manual_grain_polygons(), expected_polygons, atol=1e-12)
    np.testing.assert_array_equal(state.manual_counts.half_open, expected_count)
    assert len(state.selected_atoms) == 2
    for actual, expected in zip(state.selected_atoms, expected_atoms):
        assert (actual.grain_index, actual.layer) == (expected.grain_index, expected.layer)
        np.testing.assert_array_equal(actual.half_indices, expected.half_indices)
        np.testing.assert_allclose(actual.position, expected.position, atol=1e-12)
    assert state.axial_repeat == controls.axial_spin.value() == 2
    assert state.display_rotation_deg == controls.rotation_spin.value() == 31
    assert not state.show_reference_axes
    assert not controls.reference_axes_check.isChecked()
    assert not window.plot.reference_axes_item.isVisible()
    np.testing.assert_allclose(window._selected_vector_displacement(), expected_vector, atol=1e-12)
    assert window._selected_vector_readout() == expected_readout
    assert window.plot.vector_arrow.isVisible()
    assert window.plot.manual_annotation.isVisible()
    assert not controls.gb_region_widget.isHidden()
    np.testing.assert_allclose(window.plot.view_box.viewRange(), expected_range, atol=1e-10)
    assert state.grain_signature == window._geometry_signature()


def test_session_preserves_custom_axis_exact_angle_and_search_settings(gui, tmp_path):
    source = gui.window(lattice="SC", axis="1 -1 3", angle_deg=137.12345678,
                        lattice_constant=4.21, width=10, height=8, marker_size=25,
                        view_scale=1.5)
    controls = source.controls
    controls.local_distance_spin.setValue(0.075)
    controls.strain_spin.setValue(0.8)
    controls.search_index_spin.setValue(7)
    controls.manual_strain_limit.setValue(1.25)
    controls.manual_rotation_limit.setValue(1.5)
    controls.near_method_combo.setCurrentIndex(controls.near_method_combo.findData("strain"))
    gui.settle(source)
    output = tmp_path / "custom.dmap"
    source.save_session(output)
    restored = gui.window(lattice="FCC", axis="110")
    restored.load_session(output)
    gui.settle(restored)
    assert restored.state.parameters == source.state.parameters
    np.testing.assert_array_equal(restored.state.geometry.axis_indices, [1, -1, 3])
    assert restored.state.angle_deg == source.state.angle_deg
    assert restored.controls.axis_combo.currentData() is None
    assert restored.state.near_method == "strain"
    assert not restored.state.near_enabled
    assert restored.controls.near_method_combo.currentData() == "strain"
    for name in ("local_distance_spin", "strain_spin", "search_index_spin",
                 "manual_strain_limit", "manual_rotation_limit"):
        assert getattr(restored.controls, name).value() == getattr(controls, name).value()
    # Different dock widths may expand one axis to retain equal aspect ratio.
    original_range = np.array(source.plot.view_box.viewRange())
    actual_range = np.array(restored.plot.view_box.viewRange())
    np.testing.assert_allclose(actual_range.mean(axis=1), original_range.mean(axis=1), atol=1e-12)
    assert np.all(actual_range[:, 0] <= original_range[:, 0] + 1e-12)
    assert np.all(actual_range[:, 1] >= original_range[:, 1] - 1e-12)


def test_session_restores_selected_cell_strain_and_original_local_structure(gui, local_diamond, tmp_path):
    source = pick_local_cell(gui, local_diamond)
    original_polygons = source._manual_grain_polygons().copy()
    original_cutoff = source.state.manual_local_cutoff
    source.controls.manual_strain_button.click()
    gui.settle(source)
    assert source.state.manual_strain_fit is not None
    expected_polygons = source._manual_grain_polygons().copy()
    expected_count = source.state.manual_counts.half_open.copy()
    expected_deformations = np.array(source.state.deformations)
    expected_translations = source.state.translations.copy()
    expected_details = source.controls.manual_strain_details.toPlainText()
    assert not np.allclose(expected_deformations, [np.eye(2), np.eye(2)])
    output = tmp_path / "strained.dmap"
    source.save_session(output)

    restored = gui.window(lattice="BCC", axis="111")
    restored.load_session(output)
    gui.settle(restored)
    state, controls = restored.state, restored.controls
    assert state.manual_strain_fit is not None
    assert state.near_enabled
    assert state.near_method == "local"
    assert controls.near_button.isChecked()
    assert controls.manual_strain_button.text() == "Restore original local structure"
    assert not controls.manual_pick_button.isEnabled()
    assert not controls.local_distance_spin.isEnabled()
    assert controls.manual_strain_details.toPlainText() == expected_details
    np.testing.assert_allclose(state.deformations, expected_deformations, atol=1e-12)
    np.testing.assert_allclose(state.translations, expected_translations, atol=1e-12)
    np.testing.assert_allclose(restored._manual_grain_polygons(), expected_polygons, atol=1e-12)
    np.testing.assert_array_equal(state.manual_counts.half_open, expected_count)

    controls.manual_strain_button.click()
    gui.settle(restored)
    assert restored.state.manual_strain_fit is None
    assert restored.local_active
    assert controls.manual_pick_button.isEnabled()
    assert controls.local_distance_spin.isEnabled()
    assert controls.manual_strain_details.isHidden()
    np.testing.assert_allclose(restored._manual_grain_polygons(), original_polygons, atol=1e-12)
    np.testing.assert_array_equal(restored.state.deformations, [np.eye(2), np.eye(2)])
    np.testing.assert_array_equal(restored.state.translations, np.zeros((2, 2)))
    np.testing.assert_array_equal(restored.state.manual_counts.half_open, expected_count)
    assert restored.state.manual_local_cutoff == original_cutoff


@pytest.mark.parametrize("malformed_zip", [False, True])
def test_invalid_session_does_not_change_existing_analysis(gui, tmp_path, malformed_zip):
    window = gui.window(lattice="BCC", axis="100")
    pick_exact_cell(gui, window)
    state, counts = window.state, window.state.manual_counts
    polygons = window._manual_grain_polygons().copy()
    view_range = np.array(window.plot.view_box.viewRange())
    output = tmp_path / "invalid.dmap"
    if malformed_zip:
        with ZipFile(output, "w") as archive:
            archive.writestr("session.json", "{}")
    else:
        output.write_text("This is not a DichromaticMap session.", encoding="utf-8")
    with pytest.raises(ValueError):
        window.load_session(output)
    gui.settle(window)
    assert window.state is state
    assert state.manual_counts is counts
    assert state.geometry.lattice == "BCC"
    assert state.geometry.axis == "100"
    np.testing.assert_array_equal(window._manual_grain_polygons(), polygons)
    np.testing.assert_allclose(window.plot.view_box.viewRange(), view_range, atol=1e-12)


def test_session_buttons_use_file_dialogs_and_handle_cancel_and_invalid_file(gui, monkeypatch, tmp_path):
    window = gui.window(lattice="SC", axis="100", angle_deg=12)
    calls, warnings = [], []
    output = tmp_path / "dialog.dmap"

    def save_dialog(*args, **kwargs):
        calls.append(("save", args, kwargs))
        return str(output), "DichromaticMap session (*.dmap)"

    def open_dialog(*args, **kwargs):
        calls.append(("open", args, kwargs))
        return str(output), "DichromaticMap session (*.dmap)"

    def message(*args, **kwargs):
        warnings.append((args, kwargs))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", save_dialog)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", open_dialog)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", message)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", message)
    window.controls.save_session_button.click()
    assert output.exists()
    assert calls[-1][0] == "save"
    assert "*.dmap" in repr(calls[-1])
    window.controls.angle_spin.setValue(23)
    gui.settle(window)
    window.controls.import_session_button.click()
    gui.settle(window)
    assert calls[-1][0] == "open"
    assert window.state.angle_deg == 12
    assert not warnings

    before = output.read_bytes()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    window.controls.save_session_button.click()
    window.controls.import_session_button.click()
    assert output.read_bytes() == before
    assert window.state.angle_deg == 12
    assert not warnings

    output.write_text("invalid session", encoding="utf-8")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", open_dialog)
    window.controls.import_session_button.click()
    assert len(warnings) == 1
    assert window.state.angle_deg == 12
    assert not gui.errors


def test_import_discards_an_old_pending_cell_count(gui, monkeypatch, tmp_path):
    source = gui.window(lattice="SC", axis="111", angle_deg=13)
    output = tmp_path / "clean.dmap"
    source.save_session(output)
    window = gui.window(lattice="BCC", axis="100")
    gui.select_layer(window, 0)
    window.controls.cell_check.click()
    window.controls.cell_fit_button.click()
    gui.settle(window)
    started, release = Event(), Event()
    actual_count = window_module.cell_count_worker

    def delayed_count(*args):
        started.set()
        assert release.wait(10), "Session import did not release the old count worker"
        return actual_count(*args)

    monkeypatch.setattr(window_module, "cell_count_worker", delayed_count)
    window.controls.manual_section.toggle.setChecked(True)
    window.controls.manual_pick_button.click()
    try:
        for point in cell_corners(window.common_cell):
            gui.click_plot(window, point)
        gui.wait_for(window, started.is_set)
        assert window.compute.manual_count_future is not None
        window.load_session(output)
        release.set()
        gui.settle(window)
        assert window.state.geometry.lattice == "SC"
        assert window.state.geometry.axis == "111"
        assert window.state.angle_deg == 13
        assert window.state.manual_vertices == []
        assert window.state.manual_counts is None
        assert not window.plot.manual_annotation.isVisible()
    finally:
        release.set()


def test_import_discards_pending_orientation_work(gui, tmp_path):
    source = gui.window(lattice="SC", axis="100", angle_deg=13)
    output = tmp_path / "orientation.dmap"
    source.save_session(output)
    window = gui.window(lattice="FCC", axis="110", workers=2)
    window.controls.axis_combo.setCurrentIndex(window.controls.axis_combo.findData("111"))
    assert window.compute.parallel_stage == "grains"
    window.controls.angle_spin.setValue(42)
    window.load_session(output)
    gui.settle(window)
    state = window.state
    assert state.geometry.lattice == "SC"
    assert state.geometry.axis == "100"
    assert state.angle_deg == 13
    assert state.pending_angle == 13
    assert state.grain_signature == window._geometry_signature()
    for grain, sign in zip(state.grains, (1, -1)):
        planar = (0.5 * grain.half_indices) @ state.geometry.frame[:, :2]
        expected = planar @ crystal.rotation_matrix_2d(sign * 6.5).T
        np.testing.assert_allclose(grain.positions, expected, atol=1e-12)


def test_partial_manual_cell_can_be_completed_after_import(gui, tmp_path):
    source = gui.window(lattice="BCC", axis="100")
    gui.select_layer(source, 0)
    source.controls.cell_check.click()
    source.controls.cell_fit_button.click()
    gui.settle(source)
    corners = cell_corners(source.common_cell)
    source.controls.manual_section.toggle.setChecked(True)
    source.controls.manual_pick_button.click()
    for point in corners[:2]:
        gui.click_plot(source, point)
    assert source.state.interaction_mode == "cell"
    assert len(source.state.manual_vertices) == 2
    output = tmp_path / "partial-cell.dmap"
    source.save_session(output)
    restored = gui.window(lattice="SC", axis="111")
    restored.load_session(output)
    gui.settle(restored)
    assert restored.state.interaction_mode == "cell"
    assert restored.controls.manual_pick_button.isChecked()
    assert len(restored.state.manual_vertices) == 2
    assert restored.state.manual_counts is None
    for point in corners[2:]:
        gui.click_plot(restored, point)
    gui.settle(restored)
    assert len(restored.state.manual_vertices) == 4
    np.testing.assert_array_equal(restored.state.manual_counts.half_open, [[5, 0], [5, 0]])


def test_session_restores_applied_automatic_strain_candidate(gui, tmp_path):
    source = gui.window(lattice="FCC", axis="110", angle_deg=39.5)
    source.controls.near_section.toggle.setChecked(True)
    source.controls.near_method_combo.setCurrentIndex(
        source.controls.near_method_combo.findData("strain")
    )
    source.controls.near_button.click()
    gui.settle(source)
    assert len(source.state.near_solutions) > 1
    source.controls.near_combo.setCurrentIndex(1)
    gui.settle(source)
    expected_cell = source.state.near_cell
    assert expected_cell is not None
    output = tmp_path / "automatic-strain.dmap"
    source.save_session(output)
    restored = gui.window(lattice="SC", axis="100")
    restored.load_session(output)
    gui.settle(restored)
    assert restored.state.near_enabled
    assert restored.state.near_method == "strain"
    assert restored.state.manual_strain_fit is None
    assert restored.state.near_cell is restored.common_cell
    np.testing.assert_allclose(restored.state.deformations, [expected_cell.f1, expected_cell.f2], atol=1e-12)
    np.testing.assert_allclose(restored.common_cell.cell, expected_cell.cell, atol=1e-12)
    assert restored.common_cell.atoms == expected_cell.atoms
    assert restored.controls.near_combo.currentIndex() == 0
    assert restored.controls.near_combo.count() == 1
    assert "Green-Lagrange" in restored.controls.near_info.toPlainText()
    restored.controls.near_button.click()
    gui.settle(restored)
    assert not restored.state.near_enabled
    assert restored.state.near_cell is None
    np.testing.assert_array_equal(restored.state.deformations, [np.eye(2), np.eye(2)])
    np.testing.assert_array_equal(restored.state.translations, np.zeros((2, 2)))


def test_import_updates_zoom_limits_for_wider_and_narrower_sessions(gui, tmp_path):
    source = gui.window(lattice="SC", axis="100", angle_deg=13,
                        width=100, height=75, lattice_constant=4.21, marker_size=25)
    source.plot.view_box.translateBy(x=7.5, y=-4.25)
    gui.settle(source)
    wide_path = tmp_path / "wide-field.dmap"
    source.save_session(wide_path)
    wide_range = np.array(source.plot.view_box.viewRange())
    restored = gui.window(lattice="SC", axis="100", angle_deg=13)
    narrow_path = tmp_path / "narrow-field.dmap"
    restored.save_session(narrow_path)
    narrow_parameters = restored.state.parameters
    narrow_range = np.array(restored.plot.view_box.viewRange())

    # The saved 100-a0 field exceeds this recipient's original 60-a0 maximum.
    for path, parameters, expected_range in (
        (wide_path, source.state.parameters, wide_range),
        (narrow_path, narrow_parameters, narrow_range),
    ):
        restored.load_session(path)
        gui.settle(restored)
        assert restored.state.parameters == parameters
        actual_range = np.array(restored.plot.view_box.viewRange())
        np.testing.assert_allclose(
            actual_range.mean(axis=1), expected_range.mean(axis=1), atol=1e-10,
        )
        assert np.all(actual_range[:, 0] <= expected_range[:, 0] + 1e-10)
        assert np.all(actual_range[:, 1] >= expected_range[:, 1] - 1e-10)
        np.testing.assert_allclose(
            restored.plot.view_box.state["limits"]["xRange"],
            [0.1 * parameters.width, 5 * parameters.width], atol=1e-12,
        )
        assert restored.state.parameters.marker_size == parameters.marker_size
        assert restored.state.parameters.lattice_constant == parameters.lattice_constant
