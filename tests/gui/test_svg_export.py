"""SVG exports contain editable atoms and preserve the displayed crop and style."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets
import pyqtgraph as pg

from dichromatic_map.ui.plot import PatternPlot

from test_appearance import choose_color, image_array, open_appearance, select_symbol
from test_png_export import assert_plot_unchanged, plot_snapshot

pytestmark = pytest.mark.gui


def render_svg(path, output):
    renderer = QtSvg.QSvgRenderer(str(path))
    assert renderer.isValid()
    image = QtGui.QImage(renderer.defaultSize(), QtGui.QImage.Format_RGBA8888)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    renderer.render(painter)
    painter.end()
    assert image.save(str(output))
    return image_array(output)


@pytest.mark.parametrize("transparent", [False, True])
@pytest.mark.parametrize("symbol", ["star", "number:11"])
def test_svg_is_vector_and_matches_clean_png_styles_crop_and_background(
    gui, monkeypatch, tmp_path, transparent, symbol,
):
    window = gui.window(lattice="BCC", axis="111", angle_deg=20)
    open_appearance(gui, window)
    choose_color(monkeypatch, window, 0, "#7024bd")
    choose_color(monkeypatch, window, 1, "#269244")
    select_symbol(window, 0, symbol)
    window.controls.layer_size_spins[0].setValue(175)
    window.controls.rotation_spin.setValue(31)
    window.plot.view_box.translateBy(x=1.5, y=-0.75)
    gui.select_layer(window, 0)
    gui.settle(window)
    snapshot = plot_snapshot(window)
    output = tmp_path / "atoms.svg"
    window.plot.save_svg(output, transparent=transparent)
    document = ET.parse(output)
    tags = [element.tag.rsplit("}", 1)[-1] for element in document.iter()]
    assert tags.count("path") > 20
    assert "image" not in tags and "text" not in tags
    assert "clipPath" in tags
    svg_pixels = render_svg(output, tmp_path / "svg.png")
    window.plot.save(tmp_path / "reference.png", clean=True, transparent=transparent)
    png_pixels = image_array(tmp_path / "reference.png")
    assert svg_pixels.shape == png_pixels.shape
    if transparent:
        svg_atoms = svg_pixels[:, :, 3] > 20
        png_atoms = png_pixels[:, :, 3] > 20
        assert np.count_nonzero(svg_pixels[:, :, 3] == 0) > 100
        assert np.count_nonzero((svg_pixels[:, :, 3] > 0) & (svg_pixels[:, :, 3] < 255)) > 100
    else:
        assert np.all(svg_pixels[:, :, 3] == 255)
        assert np.count_nonzero(np.all(svg_pixels == [255, 255, 255, 255], axis=2)) > 100
        svg_atoms = np.min(svg_pixels[:, :, :3], axis=2) < 240
        png_atoms = np.min(png_pixels[:, :, :3], axis=2) < 240
    # Comparison covers the complete scene, including the hollow grain,
    # numbered marker holes, pan/rotation, layer filtering and boundary crop.
    overlap = np.count_nonzero(svg_atoms & png_atoms) / np.count_nonzero(svg_atoms | png_atoms)
    assert overlap > 0.9
    green = np.array(QtGui.QColor("#269244").getRgb()[:3])
    assert np.count_nonzero(np.max(np.abs(svg_pixels[:, :, :3].astype(int) - green), axis=2) < 4) > 100
    assert_plot_unchanged(window, snapshot)


def test_svg_matches_png_when_fractional_viewport_height_rounds_down(qt_app, monkeypatch, tmp_path):
    # A detached scene makes the fractional geometry independent of platform
    # fonts and dock layout. Its exact output height is 1493.95 pixels, so PNG
    # must round it down and fit the scene into the resulting 1800 x 1493 image.
    plot = PatternPlot(SimpleNamespace())
    monkeypatch.setattr(plot, "_await_current_scene", lambda: None)
    width = 812.25
    bounds = QtCore.QRectF(0, 0, width, 1493.95 * width / 1800)
    view = pg.GraphicsView(useOpenGL=False)
    try:
        root = pg.GraphicsWidget()
        view.setCentralItem(root)
        view.setRange(bounds, padding=0, disableAutoPixel=False)
        plot.view_box = pg.ViewBox(parent=root, enableMouse=False, enableMenu=False)
        plot.view_box.setGeometry(bounds)
        plot.view_box.setRange(QtCore.QRectF(-1, -1, 2, 2), padding=0)
        x, y = np.meshgrid(np.linspace(-0.9, 0.9, 7), np.linspace(-0.9, 0.9, 7))
        atoms = pg.ScatterPlotItem(
            x=x.ravel(), y=y.ravel(), size=9, symbol="star", pxMode=True,
            pen=pg.mkPen("#269244", width=1.5), brush=pg.mkBrush(0, 0, 0, 0),
        )
        plot.view_box.addItem(atoms)
        plot.view_box.updateMatrix()
        plot.grain_layer_items = [[atoms], []]
        plot.save(tmp_path / "reference.png", clean=True, transparent=True)
        plot.save_svg(tmp_path / "atoms.svg", transparent=True)
        png_pixels = image_array(tmp_path / "reference.png")
        svg_pixels = render_svg(tmp_path / "atoms.svg", tmp_path / "svg.png")
        assert png_pixels.shape == svg_pixels.shape == (1493, 1800, 4)
        png_atoms = png_pixels[:, :, 3] > 20
        svg_atoms = svg_pixels[:, :, 3] > 20
        overlap = np.count_nonzero(svg_atoms & png_atoms) / np.count_nonzero(svg_atoms | png_atoms)
        assert overlap > 0.9
    finally:
        view.close()
        view.deleteLater()


def test_svg_flushes_pending_rotation_and_omits_all_overlays(gui, tmp_path):
    window = gui.window()
    window.controls.rotation_spin.setValue(37)
    for grain in window.plot.grain_layer_items:
        for item in grain:
            item.hide()
    output = tmp_path / "empty.svg"
    window.plot.save_svg(output, transparent=True)
    assert window.state.display_rotation_deg == 37
    assert not window.display_rotation_timer.isActive()
    pixels = render_svg(output, tmp_path / "empty.png")
    assert np.all(pixels[:, :, 3] == 0)


def test_transparent_exports_keep_hollow_atom_centers_clear(gui, tmp_path):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    gui.select_layer(window, 0, grains=(1,))
    gui.settle(window)
    item = window.plot.grain_layer_items[1][0]
    x, y = item.getData()
    index = np.argmin(x * x + y * y)
    position = window.plot.view_box.mapFromItem(item, QtCore.QPointF(x[index], y[index]))
    scale = 1800 / window.plot.view_box.rect().width()
    column, row = round(position.x() * scale), round(position.y() * scale)
    radius = int(item.opts["size"] * scale / 2) + 3
    window.plot.save_svg(tmp_path / "hollow.svg", transparent=True)
    svg_pixels = render_svg(tmp_path / "hollow.svg", tmp_path / "hollow-svg.png")
    window.plot.save(tmp_path / "hollow.png", clean=True, transparent=True)
    for pixels in (svg_pixels, image_array(tmp_path / "hollow.png")):
        assert np.all(pixels[row - 1:row + 2, column - 1:column + 2, 3] == 0)
        assert np.count_nonzero(pixels[row - radius:row + radius, column - radius:column + radius, 3]) > 10


def test_svg_dialog_background_cancel_and_error(gui, monkeypatch, tmp_path):
    window = gui.window()
    output = tmp_path / "atoms.svg"
    calls = []

    def choose_filename(*args):
        calls.append(args)
        return str(output), "SVG image (*.svg)"

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", choose_filename)
    assert not window.controls.clean_export_check.isChecked()
    window.controls.export_svg_button.click()
    assert calls[0][1:] == ("Export atoms as vectors", "dichromatic_atoms.svg", "SVG image (*.svg)")
    assert np.all(render_svg(output, tmp_path / "white.png")[:, :, 3] == 255)
    window.controls.export_background_check.setChecked(False)
    window.controls.export_svg_button.click()
    assert np.count_nonzero(render_svg(output, tmp_path / "transparent.png")[:, :, 3] == 0) > 100

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    original_save = window.plot.save_svg
    monkeypatch.setattr(window.plot, "save_svg", lambda *args, **kwargs: pytest.fail("Cancelled export must not save"))
    window.controls.export_svg_button.click()
    monkeypatch.setattr(window.plot, "save_svg", original_save)

    snapshot = plot_snapshot(window)
    errors = []
    # A directory cannot be overwritten with an SVG file.
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path), ""))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: errors.append(args))
    window.controls.export_svg_button.click()
    assert len(errors) == 1 and errors[0][1] == "SVG export failed"
    assert_plot_unchanged(window, snapshot)
