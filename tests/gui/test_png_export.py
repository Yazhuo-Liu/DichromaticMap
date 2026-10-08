"""Atom-only PNGs preserve displayed styles without changing the live viewer."""

import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtGui, QtWidgets
import pyqtgraph.exporters

from test_appearance import choose_color, image_array, open_appearance, select_symbol
from test_workflows import pick_exact_cell
from dichromatic_map import crystal
from dichromatic_map.ui import window as window_module

pytestmark = pytest.mark.gui


def test_immediate_export_commits_latest_pending_display_rotation(gui, monkeypatch, tmp_path):
    window = gui.window()
    original = pyqtgraph.exporters.ImageExporter.export

    def export(exporter, *args, **kwargs):
        assert window.state.display_rotation_deg == 37
        assert not window.display_rotation_timer.isActive()
        for grain, items in zip(window.state.grains, window.plot.grain_layer_items):
            for layer, item in enumerate(items):
                np.testing.assert_allclose(
                    np.column_stack(item.getData()),
                    window.plot._to_view(grain.positions[grain.layers == layer]), atol=1e-12,
                )
        return original(exporter, *args, **kwargs)

    monkeypatch.setattr(pyqtgraph.exporters.ImageExporter, "export", export)
    for angle in (10, 20, 37):
        window.controls.rotation_spin.setValue(angle)
    output = tmp_path / "rotation.png"
    window.save(output)
    assert output.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("clean", [False, True])
def test_immediate_export_waits_for_pending_angle_scene(gui, monkeypatch, tmp_path, clean):
    window = gui.window(workers=1)
    output = tmp_path / "pending.png"
    release = threading.Event()
    started = threading.Event()
    original_worker = window_module.generate_grain_worker
    original_export = pyqtgraph.exporters.ImageExporter.export
    observations = []
    exported = []

    def blocked_worker(*args):
        started.set()
        assert release.wait(10)
        return original_worker(*args)

    def export(exporter, *args, **kwargs):
        assert window.state.angle_deg == 22
        assert window.state.grain_signature == window._geometry_signature()
        assert not window.state.csl_updating
        assert window.compute.parallel_stage is None
        assert not window.view_counts_timer.isActive()
        for index, (grain, sign) in enumerate(zip(window.state.grains, (1, -1))):
            planar = (0.5 * grain.half_indices) @ window.state.geometry.frame[:, :2]
            expected = planar @ crystal.rotation_matrix_2d(sign * 11).T
            np.testing.assert_allclose(grain.positions, expected, atol=1e-12)
            for layer, item in enumerate(window.plot.grain_layer_items[index]):
                np.testing.assert_allclose(
                    np.column_stack(item.getData()),
                    window.plot._to_view(expected[grain.layers == layer]), atol=1e-12,
                )
        exported.append(window.state.angle_deg)
        return original_export(exporter, *args, **kwargs)

    monkeypatch.setattr(window_module, "generate_grain_worker", blocked_worker)
    monkeypatch.setattr(pyqtgraph.exporters.ImageExporter, "export", export)
    timer = gui.QtCore.QTimer()

    def release_after_save_has_started():
        if not started.is_set():
            return
        observations.append((
            output.exists(), exported[:],
            window.state.grain_signature == window._geometry_signature(),
        ))
        timer.stop()
        release.set()

    timer.timeout.connect(release_after_save_has_started)
    timer.start(10)
    try:
        window.controls.angle_spin.setValue(22)
        # No settle call: this goes through the same alias as the export button.
        window.save(output, clean=clean)
        assert observations == [(False, [], False)]
        assert exported == [22]
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        timer.stop()
        release.set()


def test_immediate_export_flushes_pending_zoom_marker_sizes(gui, monkeypatch, tmp_path):
    window = gui.window(workers=1)
    item = window.plot.grain_layer_items[0][0]
    old_size = item.opts["size"]
    original_export = pyqtgraph.exporters.ImageExporter.export
    exported = []
    window.controls.view_slider.setValue(90)
    assert window.view_counts_timer.isActive()
    assert not window.view_refresh_timer.isActive()
    expected = window.plot._view_marker_diameter()
    assert expected != pytest.approx(old_size)

    def export(exporter, *args, **kwargs):
        assert item.opts["size"] == pytest.approx(expected)
        assert not window.view_counts_timer.isActive()
        exported.append(expected)
        return original_export(exporter, *args, **kwargs)

    monkeypatch.setattr(pyqtgraph.exporters.ImageExporter, "export", export)
    window.save(tmp_path / "zoom.png", clean=True)
    assert exported == [expected]


def plot_snapshot(window):
    plot = window.plot
    return (
        [(item, item.isVisible()) for item in plot.plot_widget.scene().items()],
        np.array(plot.view_box.viewRange()),
        plot.view_box.sceneBoundingRect(),
        window.state.manual_counts,
        tuple(window.state.grains),
    )


def assert_plot_unchanged(window, snapshot):
    visibility, view_range, bounds, counts, grains = snapshot
    assert all(item.isVisible() == visible for item, visible in visibility)
    np.testing.assert_array_equal(window.plot.view_box.viewRange(), view_range)
    assert window.plot.view_box.sceneBoundingRect() == bounds
    assert window.state.manual_counts is counts
    assert all(a is b for a, b in zip(window.state.grains, grains))


def test_clean_png_omits_overlays_axes_grid_and_preserves_live_plot(gui, tmp_path):
    window = gui.window(lattice="BCC", axis="100")
    corners = pick_exact_cell(gui, window)
    window.controls.vector_button.click()
    for point in corners[:2]:
        gui.click_plot(window, point)
    window.controls.pick_gb_button.click()
    for point in corners[1:3]:
        gui.click_plot(window, point)
    gui.settle(window)
    plot = window.plot
    snapshot = plot_snapshot(window)
    before = tmp_path / "before.png"
    plot.save(before)
    output = tmp_path / "atoms.png"
    plot.save(output, clean=True)
    pixels = image_array(output)
    assert pixels.shape[1] == 1800
    bounds = snapshot[2]
    assert abs(pixels.shape[0] - 1800 * bounds.height() / bounds.width()) <= 2
    assert np.count_nonzero(np.any(pixels != pixels[0, 0], axis=2)) > 100
    assert_plot_unchanged(window, snapshot)
    after = tmp_path / "after.png"
    plot.save(after)
    np.testing.assert_array_equal(image_array(before), image_array(after))

    # With grain symbols hidden, even a plot with visible reference axes,
    # legend, title, grid, selected cell, vector and GB must export as blank.
    for items in plot.grain_layer_items:
        for item in items:
            item.hide()
    hidden_snapshot = plot_snapshot(window)
    blank = tmp_path / "no-atoms.png"
    plot.save(blank, clean=True)
    blank_pixels = image_array(blank)
    background = np.array(plot.plot_widget.backgroundBrush().color().getRgb())
    assert np.all(blank_pixels == background)
    assert_plot_unchanged(window, hidden_snapshot)


def test_clean_png_uses_current_colors_sizes_symbols_and_filters(gui, monkeypatch, tmp_path):
    window = gui.window(lattice="BCC", axis="111", angle_deg=20)
    open_appearance(gui, window)
    choose_color(monkeypatch, window, 0, "#7024bd")
    choose_color(monkeypatch, window, 1, "#269244")
    select_symbol(window, 0, "number:11")
    window.controls.rotation_spin.setValue(27)
    # Isolate one hollow-grain layer so the exported pixels identify its style.
    gui.select_layer(window, 0, grains=(1,))
    gui.settle(window)
    small = tmp_path / "small.png"
    window.plot.save(small, clean=True)
    small_pixels = image_array(small)[:, :, :3].astype(int)
    green = np.array(QtGui.QColor("#269244").getRgb()[:3])
    purple = np.array(QtGui.QColor("#7024bd").getRgb()[:3])
    green_count = lambda pixels: np.count_nonzero(np.max(np.abs(pixels - green), axis=2) <= 3)
    assert green_count(small_pixels) > 100
    assert np.count_nonzero(np.max(np.abs(small_pixels - purple), axis=2) <= 3) == 0

    window.controls.layer_size_spins[0].setValue(200)
    gui.settle(window)
    large = tmp_path / "large.png"
    window.plot.save(large, clean=True)
    large_pixels = image_array(large)[:, :, :3].astype(int)
    assert green_count(large_pixels) > green_count(small_pixels) * 1.5
    select_symbol(window, 0, "star")
    gui.settle(window)
    star = tmp_path / "star.png"
    window.plot.save(star, clean=True)
    star_pixels = image_array(star)[:, :, :3].astype(int)
    assert np.count_nonzero(np.any(star_pixels != large_pixels, axis=2)) > 500


def test_clean_png_keeps_atom_positions_after_rotation_and_pan(gui, tmp_path):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    gui.select_layer(window, 0, grains=(0,))
    window.controls.rotation_spin.setValue(31)
    window.plot.view_box.translateBy(x=1.5, y=-0.75)
    gui.settle(window)
    plot = window.plot
    x, y = plot.grain_layer_items[0][0].getData()
    (xmin, xmax), (ymin, ymax) = plot.view_box.viewRange()
    mask = (x > xmin + 0.5) & (x < xmax - 0.5) & (y > ymin + 0.5) & (y < ymax - 0.5)
    points = np.column_stack((x[mask], y[mask]))
    assert len(points) > 20
    output = tmp_path / "positions.png"
    plot.save(output, clean=True)
    pixels = image_array(output)
    background = np.array(plot.plot_widget.backgroundBrush().color().getRgb()[:3])
    # A hidden export ViewBox must update its coordinate transform before
    # painting; otherwise every symbol collapses at the top-left corner.
    for x, y in points:
        column = round((x - xmin) / (xmax - xmin) * pixels.shape[1])
        row = round((ymax - y) / (ymax - ymin) * pixels.shape[0])
        patch = pixels[row - 2:row + 3, column - 2:column + 3, :3].astype(int)
        assert np.any(np.max(np.abs(patch - background), axis=2) > 20), (x, y, column, row)


def test_clean_png_dialog_and_cancel(gui, monkeypatch, tmp_path):
    window = gui.window()
    calls = []
    output = tmp_path / "atoms.png"

    def choose_filename(*args):
        calls.append(args)
        return str(output), "PNG image (*.png)"

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", choose_filename)
    window.controls.clean_export_check.setChecked(True)
    window.controls.export_button.click()
    assert calls[0][1:] == ("Export atoms only", "dichromatic_atoms.png", "PNG image (*.png)")
    assert image_array(output).shape[1] == 1800
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    monkeypatch.setattr(window.plot, "save", lambda *args, **kwargs: pytest.fail("Cancelled export must not save"))
    window.controls.export_button.click()


@pytest.mark.parametrize("failure", ["exception", "false"])
def test_clean_png_failure_preserves_plot_and_reports_error(gui, monkeypatch, tmp_path, failure):
    window = gui.window(lattice="BCC", axis="100")
    pick_exact_cell(gui, window)
    snapshot = plot_snapshot(window)
    output = tmp_path / "failed.png"
    errors = []

    def fail_export(*args, **kwargs):
        if failure == "exception":
            raise OSError("Export write failed")
        return False

    monkeypatch.setattr(pyqtgraph.exporters.ImageExporter, "export", fail_export)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(output), "PNG image (*.png)"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: errors.append(args))
    window.controls.clean_export_check.setChecked(True)
    window.controls.export_button.click()
    assert len(errors) == 1 and errors[0][1] == "PNG export failed"
    assert not output.exists()
    assert_plot_unchanged(window, snapshot)
