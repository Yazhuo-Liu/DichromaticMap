"""Custom grain colors and distinct layer markers are display-only preferences."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtCore, QtGui, QtWidgets

from dichromatic_map.ui import GRAIN_1_COLOR, GRAIN_1_EDGE, GRAIN_2_COLOR, LAYER_SYMBOLS
from test_workflows import pick_exact_cell

pytestmark = pytest.mark.gui


def open_appearance(gui, window):
    controls = window.controls
    controls.view_performance_section.toggle.setChecked(True)
    tabs = controls.view_performance_tabs
    index = next(i for i in range(tabs.count()) if tabs.tabText(i) == "APPEARANCE")
    tabs.setCurrentIndex(index)
    gui.settle(window)
    return tabs.widget(index)


def choose_color(monkeypatch, window, grain, color):
    monkeypatch.setattr(
        QtWidgets.QColorDialog, "getColor", lambda *args, **kwargs: QtGui.QColor(color),
    )
    window.controls.grain_color_buttons[grain].click()


def select_symbol(window, layer, symbol):
    combo = window.controls.layer_symbol_combos[layer]
    index = combo.findData(symbol)
    assert index >= 0, f"Layer marker {symbol!r} is unavailable"
    combo.setCurrentIndex(index)


def reference_arrow_colors(item):
    colors = []

    class RecordingPainter(QtGui.QPainter):
        def drawLine(self, start, end):
            colors.append(self.pen().color().name())
            super().drawLine(start, end)

    canvas = QtGui.QImage(item.boundingRect().size().toSize(), QtGui.QImage.Format_RGBA8888)
    canvas.fill(QtCore.Qt.GlobalColor.transparent)
    painter = RecordingPainter(canvas)
    try:
        item.paint(painter, None)
    finally:
        painter.end()
    return colors


def image_array(path):
    image = QtGui.QImage(str(path)).convertToFormat(QtGui.QImage.Format_RGBA8888)
    assert not image.isNull()
    return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine(),
    )[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


def test_appearance_color_dialog_cancel_and_reset_keep_existing_defaults(gui, monkeypatch):
    window = gui.window(lattice="BCC", axis="100", angle_deg=20)
    controls, state, plot = window.controls, window.state, window.plot
    open_appearance(gui, window)
    assert controls.view_performance_tabs.tabText(1) == "PERFORMANCE"
    assert tuple(state.grain_colors) == (GRAIN_1_COLOR, GRAIN_2_COLOR)
    assert tuple(state.layer_symbols) == LAYER_SYMBOLS[:2]
    assert plot.grain_layer_items[0][0].opts["pen"].color().name() == GRAIN_1_EDGE
    assert all(button.isVisible() for button in controls.grain_color_buttons)
    original_icon = controls.grain_layer_checks[0][0].icon().pixmap(18, 18).toImage()

    choose_color(monkeypatch, window, 0, "#7024bd")
    choose_color(monkeypatch, window, 1, "#269244")
    gui.settle(window)
    assert tuple(state.grain_colors) == ("#7024bd", "#269244")
    assert controls.grain_layer_checks[0][0].icon().pixmap(18, 18).toImage() != original_icon
    for layer in range(state.geometry.layer_count):
        assert plot.grain_layer_items[0][layer].opts["brush"].color().name() == "#7024bd"
        assert plot.grain_layer_items[1][layer].opts["pen"].color().name() == "#269244"
    assert reference_arrow_colors(plot.reference_axes_item) == ["#7024bd"] * 2 + ["#269244"] * 2
    assert [item.opts["pen"].color().name() for item in plot.manual_grain_cell_items] == list(state.grain_colors)

    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", lambda *args, **kwargs: QtGui.QColor())
    controls.grain_color_buttons[0].click()
    assert tuple(state.grain_colors) == ("#7024bd", "#269244")
    select_symbol(window, 0, "star")
    controls.appearance_reset_button.click()
    gui.settle(window)
    assert tuple(state.grain_colors) == (GRAIN_1_COLOR, GRAIN_2_COLOR)
    assert tuple(state.layer_symbols) == LAYER_SYMBOLS[:2]
    assert [combo.currentData() for combo in controls.layer_symbol_combos] == list(LAYER_SYMBOLS[:2])
    assert plot.grain_layer_items[0][0].opts["pen"].color().name() == GRAIN_1_EDGE
    assert reference_arrow_colors(plot.reference_axes_item) == [GRAIN_1_COLOR] * 2 + [GRAIN_2_COLOR] * 2


def test_layer_symbols_disable_duplicates_and_reject_programmatic_duplicate(gui):
    window = gui.window(lattice="BCC", axis="111", angle_deg=20)
    controls, state, plot = window.controls, window.state, window.plot
    open_appearance(gui, window)
    assert len(controls.layer_symbol_combos) == 3
    first, second = controls.layer_symbol_combos[:2]
    assert not second.model().item(second.findData("o")).isEnabled()
    assert first.model().item(first.findData("o")).isEnabled()
    select_symbol(window, 0, "star")
    assert state.layer_symbols[0] == "star"
    assert second.model().item(second.findData("o")).isEnabled()
    assert not second.model().item(second.findData("star")).isEnabled()
    select_symbol(window, 1, "star")  # setCurrentIndex can bypass disabled popup rows.
    assert state.layer_symbols[1] == "d"
    assert second.currentData() == "d"
    assert len(set(state.layer_symbols)) == state.geometry.layer_count
    select_symbol(window, 1, "o")
    gui.settle(window)
    assert list(state.layer_symbols) == ["star", "o", "t"]
    for layer, symbol in enumerate(state.layer_symbols):
        for grain in range(2):
            assert plot.grain_layer_items[grain][layer].opts["symbol"] == symbol
        assert plot.coincidence_items[layer].opts["symbol"] == symbol
    # Legend samples must continue to use the current layer graphics.
    samples = [sample.item for sample, _label in plot.legend.items]
    assert plot.grain_layer_items[0][0] in samples
    assert plot.grain_layer_items[1][1] in samples


def test_appearance_changes_preserve_counted_cell_vector_and_generated_grains(gui, monkeypatch):
    window = gui.window(lattice="BCC", axis="100")
    corners = pick_exact_cell(gui, window)
    controls, state, plot = window.controls, window.state, window.plot
    controls.vector_button.click()
    for grain, point in enumerate(corners[:2]):
        gui.select_layer(window, 0, grains=(grain,))
        gui.click_plot(window, point)
    open_appearance(gui, window)
    grains = tuple(state.grains)
    counts = state.manual_counts
    atoms = tuple(state.selected_atoms)
    polygons = window._manual_grain_polygons().copy()
    vector = window._selected_vector_displacement().copy()
    view_range = np.array(plot.view_box.viewRange())

    def unexpected_regeneration(*args, **kwargs):
        pytest.fail("Appearance changes must not regenerate crystal geometry")

    monkeypatch.setattr(window, "_start_parallel_regeneration", unexpected_regeneration)
    choose_color(monkeypatch, window, 0, "#7024bd")
    select_symbol(window, 0, "star")
    gui.settle(window)
    assert all(actual is expected for actual, expected in zip(state.grains, grains))
    assert state.manual_counts is counts
    assert all(actual is expected for actual, expected in zip(state.selected_atoms, atoms))
    np.testing.assert_array_equal(window._manual_grain_polygons(), polygons)
    np.testing.assert_array_equal(window._selected_vector_displacement(), vector)
    np.testing.assert_allclose(plot.view_box.viewRange(), view_range, atol=1e-12)
    assert plot.manual_vertex_item.opts["symbol"] == "star"
    assert plot.vector_arrow.isVisible()


def test_layer_symbol_updates_existing_local_pairs_without_recomputing(gui):
    window = gui.window(lattice="FCC", axis="110", angle_deg=22)
    gui.select_layer(window, 1)
    window.controls.near_section.toggle.setChecked(True)
    window.controls.near_button.click()
    gui.settle(window)
    open_appearance(gui, window)
    pairs = window.state.local_pairs
    item = window.plot.local_match_item
    positions = np.column_stack(item.getData())
    assert len(positions) > 0
    select_symbol(window, 1, "star")
    gui.settle(window)
    assert window.state.local_pairs is pairs
    np.testing.assert_array_equal(np.column_stack(item.getData()), positions)
    assert all(symbol == "star" for symbol in item.data["symbol"])


def test_custom_axis_markers_do_not_repeat_and_appearance_fits_small_window(gui):
    window = gui.window(lattice="SC", axis="2 3 5", angle_deg=15, width=6, height=5)
    page = open_appearance(gui, window)
    controls, state, plot = window.controls, window.state, window.plot
    assert state.geometry.layer_count == 38
    assert len(controls.layer_symbol_combos) == 38
    assert len(set(state.layer_symbols)) == 38
    assert tuple(state.layer_symbols[:12]) == LAYER_SYMBOLS
    assert all(symbol.startswith("number:") for symbol in state.layer_symbols[12:])
    assert [combo.currentData() for combo in controls.layer_symbol_combos] == list(state.layer_symbols)
    rendered = []
    for item in plot.grain_layer_items[0]:
        symbol = item.opts["symbol"]
        if isinstance(symbol, QtGui.QPainterPath):
            rendered.append(tuple((point.x(), point.y()) for point in symbol.toFillPolygon()))
        else:
            rendered.append(symbol)
    assert len(set(rendered)) == 38
    window.resize(1040, 680)
    gui.settle(window)
    assert window.width() <= 1040
    assert page.width() < window.width() / 2
    scrolls = page.findChildren(QtWidgets.QScrollArea)
    assert scrolls and any(scroll.verticalScrollBar().maximum() > 0 for scroll in scrolls)
    assert all(scroll.horizontalScrollBar().maximum() == 0 for scroll in scrolls)
    # Reassign marker 38 to the first layer, then retain it when only one layer
    # remains. The compact palette must still offer this existing assignment.
    select_symbol(window, 0, "number:1")
    select_symbol(window, 37, "o")
    select_symbol(window, 0, "number:38")
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData("100"))
    gui.settle(window)
    assert list(state.layer_symbols) == ["number:38"]
    assert len(controls.layer_symbol_combos) == 1
    assert controls.layer_symbol_combos[0].currentData() == "number:38"


def test_geometry_switch_keeps_colors_and_reuses_unique_layer_prefix(gui, monkeypatch):
    window = gui.window(lattice="BCC", axis="100")
    controls, state = window.controls, window.state
    open_appearance(gui, window)
    choose_color(monkeypatch, window, 1, "#269244")
    select_symbol(window, 0, "t")  # This is the default for the newly added third layer.
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData("111"))
    gui.settle(window)
    assert state.grain_colors[1] == "#269244"
    assert state.layer_symbols[:2] == ["t", "d"]
    assert len(set(state.layer_symbols)) == state.geometry.layer_count == 3
    assert [combo.currentData() for combo in controls.layer_symbol_combos] == list(state.layer_symbols)
    controls.structure_combo.setCurrentIndex(controls.structure_combo.findData("SC"))
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData("100"))
    gui.settle(window)
    assert list(state.layer_symbols) == ["t"]
    assert state.grain_colors[1] == "#269244"
    assert len(controls.layer_symbol_combos) == 1


def test_session_restores_colors_symbols_and_appearance_controls(gui, monkeypatch, tmp_path):
    source = gui.window(lattice="SC", axis="111", angle_deg=20)
    open_appearance(gui, source)
    choose_color(monkeypatch, source, 0, "#7024bd")
    choose_color(monkeypatch, source, 1, "#269244")
    select_symbol(source, 0, "star")
    select_symbol(source, 2, "number:7")
    gui.settle(source)
    output = tmp_path / "appearance.dmap"
    source.save_session(output)
    restored = gui.window(lattice="BCC", axis="100")
    restored.load_session(output)
    gui.settle(restored)
    assert restored.state.grain_colors == source.state.grain_colors
    assert restored.state.layer_symbols == source.state.layer_symbols
    assert [combo.currentData() for combo in restored.controls.layer_symbol_combos] == list(source.state.layer_symbols)
    assert reference_arrow_colors(restored.plot.reference_axes_item) == ["#7024bd"] * 2 + ["#269244"] * 2
    for layer in range(3):
        assert restored.plot.grain_layer_items[0][layer].opts["brush"].color().name() == "#7024bd"
        assert restored.plot.grain_layer_items[1][layer].opts["pen"].color().name() == "#269244"


def test_png_export_contains_selected_grain_colors(gui, monkeypatch, tmp_path):
    window = gui.window(lattice="SC", axis="100", angle_deg=30)
    open_appearance(gui, window)
    choose_color(monkeypatch, window, 0, "#7024bd")
    choose_color(monkeypatch, window, 1, "#269244")
    gui.settle(window)
    output = tmp_path / "colored.png"
    window.plot.save(output)
    pixels = image_array(output)[:, :, :3].astype(int)
    for color in ("#7024bd", "#269244"):
        rgb = np.array(QtGui.QColor(color).getRgb()[:3])
        assert np.count_nonzero(np.max(np.abs(pixels - rgb), axis=2) <= 3) > 30
