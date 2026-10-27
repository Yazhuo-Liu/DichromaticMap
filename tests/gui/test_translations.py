"""Physical grain moves persist across rendering, filtering and sessions."""

from threading import Event

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtCore

from dichromatic_map.state import CellVertex
from dichromatic_map.ui import window as window_module
from test_appearance import image_array

pytestmark = pytest.mark.gui


def open_translation(window, grain=0, delta=(0.25, -0.125)):
    controls = window.controls
    controls.translation_section.toggle.setChecked(True)
    controls.translation_grain_combo.setCurrentIndex(grain)
    controls.translation_dx_spin.setValue(delta[0])
    controls.translation_dy_spin.setValue(delta[1])
    return controls


def atom_map(grain):
    return {tuple(indices): (point.copy(), int(layer))
            for indices, point, layer in zip(grain.half_indices, grain.positions, grain.layers)}


def assert_moved(before, after, delta, bounds=None):
    after = atom_map(after)
    moved_layers = set()
    moved = unchanged = 0
    for identity in before.keys() & after.keys():
        point, layer = before[identity]
        selected = bounds is None or (
            bounds[0] <= point[0] <= bounds[1] and bounds[2] <= point[1] <= bounds[3]
        )
        expected = point + delta if selected else point
        np.testing.assert_allclose(after[identity][0], expected, atol=1e-11)
        assert after[identity][1] == layer
        if selected and np.any(delta):
            moved += 1
            moved_layers.add(layer)
        else:
            unchanged += 1
    return moved, unchanged, moved_layers


def exported_atom_coverage(window, path, points):
    """Sample isolated atoms using this export's actual viewport transform."""
    window.plot.save(path, clean=True, transparent=True)
    pixels = image_array(path)
    box = window.plot.view_box
    bounds = box.rect()
    # PNG preserves the current viewport aspect ratio, rounding its height.
    # A title wrapping after a move can legitimately change that viewport.
    scale = min(pixels.shape[1] / bounds.width(), pixels.shape[0] / bounds.height())
    coverage = []
    for x, y in points:
        point = box.mapFromView(QtCore.QPointF(x, y))
        column = round((point.x() - bounds.left()) * scale)
        row = round((point.y() - bounds.top()) * scale)
        # At this fixture's 1800-pixel export scale, a 25-pixel patch covers
        # the marker while staying clear of neighboring atoms and the GB.
        radius = 12
        assert radius <= row < pixels.shape[0] - radius
        assert radius <= column < pixels.shape[1] - radius
        patch = pixels[row - radius:row + radius + 1, column - radius:column + radius + 1, 3]
        coverage.append(np.count_nonzero(patch))
    return coverage


def test_whole_grain_move_preserves_gb_filters_and_other_grain(gui, tmp_path):
    window = gui.window(lattice="BCC", axis="100", angle_deg=0)
    window.controls.pick_gb_button.click()
    for point in ((-2, 0), (2, 0)):
        gui.click_plot(window, point)
    window.controls.first_bicrystal_button.click()
    gui.settle(window)
    boundary = np.array(window.state.selected_points)
    regions = window._region_states()
    before = [atom_map(grain) for grain in window.state.grains]
    signature = window._geometry_signature()
    probes = ((1, 1), (1.25, 0.875), (1, -1))  # G1 old/new positions and stationary G2
    old, new, stationary = exported_atom_coverage(window, tmp_path / "before.png", probes)
    assert old > 25 and new == 0 and stationary > 25
    controls = open_translation(window)
    controls.translation_whole_button.click()
    gui.settle(window)
    assert window.state.grain_displacements == [[{"delta": [0.25, -0.125], "bounds": None}], []]
    assert window._geometry_signature() != signature
    assert window.state.grain_signature == window._geometry_signature()
    moved, _, layers = assert_moved(before[0], window.state.grains[0], np.array([0.25, -0.125]))
    assert moved > 100 and layers == {0, 1}
    assert_moved(before[1], window.state.grains[1], np.zeros(2))
    np.testing.assert_array_equal(window.state.selected_points, boundary)
    assert window._region_states() == regions
    assert window.plot.boundary_item.getData()[0].size == 2
    assert all(count > 0 for count in window.state.visible_atom_counts)
    assert window.common_cell is None
    old, new, stationary = exported_atom_coverage(window, tmp_path / "after.png", probes)
    assert old == 0 and new > 25 and stationary > 25

    controls.translation_undo_button.click()
    gui.settle(window)
    assert window.state.grain_displacements == [[], []]
    assert_moved(before[0], window.state.grains[0], np.zeros(2))
    np.testing.assert_array_equal(window.state.selected_points, boundary)
    assert window._region_states() == regions


def test_rectangle_move_uses_model_axes_and_hidden_axial_layers(gui):
    window = gui.window(lattice="BCC", axis="100", angle_deg=0)
    gui.select_layer(window, 0, grains=(1,))
    window.controls.rotation_spin.setValue(31)
    gui.settle(window)
    before = [atom_map(grain) for grain in window.state.grains]
    controls = open_translation(window, grain=1, delta=(0.375, 0.25))
    controls.translation_select_button.click()
    for point in ((-1.7, -1.3), (1.4, 1.8)):
        gui.click_plot(window, point)
    assert window.state.interaction_mode == "idle"
    assert controls.translation_region_button.isEnabled()
    assert window.plot.translation_region_item.getData()[0].size == 5
    controls.translation_region_button.click()
    gui.settle(window)
    operation = window.state.grain_displacements[1][0]
    np.testing.assert_allclose(operation["bounds"], [-1.7, 1.4, -1.3, 1.8], atol=0.04)
    moved, unchanged, layers = assert_moved(
        before[1], window.state.grains[1], np.array([0.375, 0.25]), operation["bounds"],
    )
    assert moved > 5 and unchanged > 100 and layers == {0, 1}
    assert_moved(before[0], window.state.grains[0], np.zeros(2))
    assert window.state.visible_grain_layers == [set(), {0}]
    assert window._translation_corners == []
    assert not controls.translation_region_button.isEnabled()
    assert not controls.manual_strain_button.isEnabled()
    assert not controls.manual_complete_button.isEnabled()
    assert not controls.near_method_combo.model().item(controls.near_method_combo.findData("strain")).isEnabled()


def test_moves_survive_pan_and_session_then_reset_with_angle_or_lattice(gui, tmp_path):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    controls = open_translation(window, delta=(0.375, -0.25))
    controls.translation_whole_button.click()
    gui.settle(window)
    window.plot.view_box.translateBy(x=25, y=0)
    gui.settle(window)
    grain = window.state.grains[0]
    original = (0.5 * grain.half_indices) @ window.state.geometry.frame[:, :2]
    np.testing.assert_allclose(grain.positions, original + [0.375, -0.25], atol=1e-11)
    assert window.state.grain_signature == window._geometry_signature()
    expected = atom_map(grain)
    path = tmp_path / "moved.dmap"
    window.save_session(path)
    controls.translation_reset_button.click()
    gui.settle(window)
    window.load_session(path)
    gui.settle(window)
    assert window.state.grain_displacements[0] == [{"delta": [0.375, -0.25], "bounds": None}]
    assert_moved(expected, window.state.grains[0], np.zeros(2))
    assert controls.translation_undo_button.isEnabled()
    assert "G1: 1 moves" in controls.translation_status.text()
    window.controls.angle_spin.setValue(10)
    gui.settle(window)
    assert window.state.grain_displacements == [[], []]
    assert "angle change" in controls.translation_status.text()
    controls.translation_whole_button.click()
    gui.settle(window)
    controls.structure_combo.setCurrentIndex(controls.structure_combo.findData("BCC"))
    gui.settle(window)
    assert window.state.grain_displacements == [[], []]


def test_retained_small_bulk_deformation_survives_moves_and_local_toggles(gui):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    deformation = np.diag([1.000001, 1.0])
    window.state.deformations = (deformation.copy(), np.eye(2))
    window._start_parallel_regeneration(True)
    gui.settle(window)
    controls = open_translation(window)
    controls.translation_whole_button.click()
    gui.settle(window)
    controls.near_button.click()
    gui.settle(window)
    controls.near_button.click()
    gui.settle(window)
    np.testing.assert_array_equal(window.state.deformations[0], deformation)
    controls.translation_reset_button.click()
    gui.settle(window)
    assert window.state.grain_displacements == [[], []]
    np.testing.assert_array_equal(window.state.deformations[0], deformation)
    assert window.common_cell is None
    assert not controls.near_method_combo.model().item(controls.near_method_combo.findData("strain")).isEnabled()
    controls.angle_spin.setValue(10)
    gui.settle(window)
    np.testing.assert_array_equal(window.state.deformations[0], np.eye(2))


def test_manual_cell_count_uses_displaced_atoms(gui):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    controls = open_translation(window, delta=(0.5, 0))
    controls.translation_select_button.click()
    for point in ((-0.1, -0.1), (0.1, 0.1)):
        gui.click_plot(window, point)
    controls.translation_region_button.click()
    gui.settle(window)
    # A small region around the moved atom's new position contains G1 only.
    corners = np.array([[0.35, -0.15], [0.65, -0.15], [0.65, 0.15], [0.35, 0.15]])
    window.state.manual_vertices = [
        CellVertex(point, 0, "local", np.stack([point, point])) for point in corners
    ]
    window._queue_manual_count()
    gui.settle(window)
    counts = window.state.manual_counts
    assert counts is not None
    assert counts.interior.sum(axis=1).tolist() == [1, 0]
    assert "Half-open geometric count" in controls.manual_info.toPlainText()
    assert "duplicates" not in controls.manual_info.toPlainText()


def test_translation_cancel_zero_delta_and_late_worker_cannot_restore_undone_move(gui, monkeypatch):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    controls = open_translation(window, delta=(0, 0))
    controls.translation_whole_button.click()
    assert window.state.grain_displacements == [[], []]
    assert "nonzero" in controls.translation_status.text()
    controls.translation_select_button.click()
    gui.click_plot(window, (-1, -1))
    gui.key(window, gui.QtCore.Qt.Key_Escape)
    assert window._translation_corners == []
    assert window.state.interaction_mode == "idle"
    controls.translation_dx_spin.setValue(0.25)
    before = atom_map(window.state.grains[0])
    release, started = Event(), Event()
    original_worker = window_module.generate_grain_worker

    def worker(*args):
        if args[-1]:
            started.set()
            assert release.wait(10)
        return original_worker(*args)

    monkeypatch.setattr(window_module, "generate_grain_worker", worker)
    try:
        controls.translation_whole_button.click()
        gui.wait_for(window, started.is_set)
        controls.translation_undo_button.click()
        release.set()
        gui.settle(window)
    finally:
        release.set()
    assert window.state.grain_displacements == [[], []]
    assert window.state.grain_signature == window._geometry_signature()
    assert_moved(before, window.state.grains[0], np.zeros(2))
