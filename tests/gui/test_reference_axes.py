"""Floating grain-reference axes follow orientation without following data zoom."""

import re

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtCore, QtGui

pytestmark = pytest.mark.gui


def oriented_axes(angle_deg):
    """Independent unit directions for a counterclockwise planar rotation."""
    radians = np.deg2rad(angle_deg)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.array([[cosine, sine], [-sine, cosine]])


def open_view_controls(window):
    window.controls.view_performance_section.toggle.setChecked(True)
    window.controls.view_performance_tabs.setCurrentIndex(0)


def screen_metrics(item, view_box):
    bounds = item.sceneBoundingRect()
    view = view_box.sceneBoundingRect()
    return np.array([
        bounds.width(), bounds.height(),
        bounds.left() - view.left(), view.bottom() - bounds.bottom(),
    ])


def test_reference_axes_default_and_view_toggle_survive_geometry_changes(gui):
    window = gui.window(lattice="BCC", axis="100", angle_deg=0)
    controls, state, plot = window.controls, window.state, window.plot
    open_view_controls(window)
    gui.settle(window)
    checkbox = controls.reference_axes_check
    assert checkbox.text() == "Grain reference axes"
    assert checkbox.isVisible() and checkbox.isChecked()
    assert state.show_reference_axes
    assert plot.reference_axes_item.isVisible()
    checkbox.click()
    assert not state.show_reference_axes
    assert not plot.reference_axes_item.isVisible()

    controls.orientation_layers_tabs.setCurrentIndex(0)
    controls.structure_combo.setCurrentIndex(controls.structure_combo.findData("SC"))
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData("111"))
    gui.settle(window)
    assert (state.geometry.lattice, state.geometry.axis) == ("SC", "111")
    assert not checkbox.isChecked()
    assert not state.show_reference_axes
    assert not plot.reference_axes_item.isVisible()
    open_view_controls(window)
    checkbox.click()
    assert state.show_reference_axes
    assert plot.reference_axes_item.isVisible()


def test_reference_axes_follow_misorientation_display_rotation_and_signed_axis(gui):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    controls, plot = window.controls, window.plot
    item = plot.reference_axes_item
    np.testing.assert_allclose(item.directions, [np.eye(2), np.eye(2)], atol=1e-12)
    controls.angle_spin.setValue(40)
    controls.rotation_spin.setValue(31)
    gui.settle(window)
    np.testing.assert_allclose(
        item.directions, [oriented_axes(51), oriented_axes(11)], atol=1e-12,
    )
    controls.orientation_layers_tabs.setCurrentIndex(0)
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData(None))
    controls.custom_axis_edit.setText("1 -1 3")
    controls.custom_axis_apply.click()
    gui.settle(window)
    np.testing.assert_array_equal(window.state.geometry.axis_indices, [1, -1, 3])
    indices = [tuple(map(int, re.findall(r"[+-]?\d+", label))) for label in item.labels]
    assert indices == [(1, 1, 0), (-3, 3, 2)]
    # Applying another axis resets its misorientation but keeps display rotation.
    theta = window.state.angle_deg
    np.testing.assert_allclose(
        item.directions,
        [oriented_axes(31 + theta / 2), oriented_axes(31 - theta / 2)],
        atol=1e-12,
    )


def test_reference_axes_use_polar_rotation_without_shearing_the_indicator(gui):
    window = gui.window(lattice="FCC", axis="110", angle_deg=36)
    window.controls.rotation_spin.setValue(23)
    gui.settle(window)
    # Known F = R U with symmetric positive-definite U provides a reference
    # independent of the production polar decomposition and arbitrary strain searches.
    stretch = np.array([[1.2, 0.3], [0.3, 0.9]])
    window.state.deformations = (
        oriented_axes(13).T @ stretch, oriented_axes(-7).T @ stretch,
    )
    window.plot._update_reference_axes()
    directions = window.plot.reference_axes_item.directions
    np.testing.assert_allclose(
        directions, [oriented_axes(54), oriented_axes(-2)], atol=1e-12,
    )
    for frame in directions:
        np.testing.assert_allclose(frame @ frame.T, np.eye(2), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.0)


def test_reference_axes_keep_screen_size_and_lower_left_anchor_during_navigation(gui):
    window = gui.window(lattice="BCC", axis="100", angle_deg=37)
    plot = window.plot
    item = plot.reference_axes_item
    baseline = screen_metrics(item, plot.view_box)
    assert np.all(baseline > 0)
    directions = item.directions.copy()
    plot.view_box.translateBy(x=5, y=-3)
    gui.settle(window)
    np.testing.assert_allclose(screen_metrics(item, plot.view_box), baseline, atol=1)
    plot.view_box.scaleBy((0.7, 0.7))
    gui.settle(window)
    np.testing.assert_allclose(screen_metrics(item, plot.view_box), baseline, atol=1)
    for width, height in ((1040, 680), (1500, 900)):
        window.resize(width, height)
        gui.settle(window)
        assert plot.view_box.sceneBoundingRect().contains(item.sceneBoundingRect())
        np.testing.assert_allclose(screen_metrics(item, plot.view_box), baseline, atol=1)
    np.testing.assert_array_equal(item.directions, directions)


def test_reference_axes_origin_and_panel_stay_fixed_while_rotating(gui):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    controls, plot = window.controls, window.plot
    item = plot.reference_axes_item

    def placement():
        origin = item.mapToScene(item.origin)
        view = plot.view_box.sceneBoundingRect()
        return np.concatenate((
            screen_metrics(item, plot.view_box),
            [origin.x() - view.left(), view.bottom() - origin.y()],
        ))

    baseline = placement()
    # Include both sides of the old direction-dependent label alignment change.
    for misorientation, display_rotation in (
        (0, 78.4), (0, 78.5), (18.2, 78.5), (30, -120),
        (44.8, 179.9), (0, -180), (0, 0),
    ):
        controls.angle_spin.setValue(misorientation)
        controls.rotation_spin.setValue(display_rotation)
        gui.settle(window)
        np.testing.assert_allclose(placement(), baseline, atol=1e-6)
        assert plot.view_box.sceneBoundingRect().contains(item.sceneBoundingRect())


def test_reference_axes_draw_four_arrows_from_one_origin_without_headers(gui):
    window = gui.window(lattice="SC", axis="110", angle_deg=37)
    item = window.plot.reference_axes_item
    starts, texts = [], []

    class RecordingPainter(QtGui.QPainter):
        def drawLine(self, start, end):
            starts.append(self.transform().map(start))
            super().drawLine(start, end)

        def drawText(self, rect, flags, text):
            texts.append(text)
            super().drawText(rect, flags, text)

    canvas = QtGui.QImage(
        item.boundingRect().size().toSize(), QtGui.QImage.Format_RGBA8888,
    )
    canvas.fill(QtCore.Qt.GlobalColor.transparent)
    painter = RecordingPainter(canvas)
    try:
        item.paint(painter, None)
    finally:
        painter.end()
    assert len(starts) == 4
    assert all(start == item.origin for start in starts)
    assert texts == list(item.labels) * 2


def test_reference_axes_labels_stay_contained_and_move_continuously(gui):
    window = gui.window(lattice="SC", axis="110", angle_deg=37)
    item = window.plot.reference_axes_item
    threshold = np.rad2deg(np.arccos(0.2))
    transitions = [
        (base + sign * threshold) % 360
        for base in (0, 90, 180, 270) for sign in (-1, 1)
    ]
    angles = sorted([
        *range(360),
        *(angle + offset for angle in transitions for offset in (-1e-4, 1e-4)),
    ])
    # Exercise ordinary and multi-digit Miller-index labels.
    for labels in (item.labels, ("[-17 13 0]", "[-39 -51 458]")):
        item.set_reference([oriented_axes(0), oriented_axes(37)], labels)
        baseline, origin = item.boundingRect(), QtCore.QPointF(item.origin)
        for angle in angles:
            item.set_reference([oriented_axes(angle), oriented_axes(angle + 37)], labels)
            assert item.boundingRect() == baseline
            assert item.origin == origin
            assert len(item._groups) == 2
            for arrows in item._groups:
                assert len(arrows) == 2
                for tip, head, text_rect, label in arrows:
                    assert baseline.contains(tip + origin)
                    assert baseline.contains(head.boundingRect().translated(origin))
                    assert baseline.contains(text_rect.translated(origin)), (angle, label)
        for angle in transitions:
            centers = []
            for offset in (-1e-4, 1e-4):
                item.set_reference([
                    oriented_axes(angle + offset), oriented_axes(angle + offset + 37),
                ], labels)
                centers.append(np.array([
                    [rect.center().x(), rect.center().y()]
                    for arrows in item._groups for _, _, rect, _ in arrows
                ]))
            np.testing.assert_allclose(centers[0], centers[1], atol=0.01)


def test_vector_readout_clears_reference_axes_and_returns_down_when_hidden(gui):
    window = gui.window(lattice="SC", axis="100", angle_deg=0)
    controls, plot = window.controls, window.plot
    gui.select_layer(window, 0, grains=(0,))
    controls.vector_button.click()
    gui.click_plot(window, [-1, 1])
    gui.select_layer(window, 0, grains=(1,))
    gui.click_plot(window, [1, 1])
    assert [atom.grain_index for atom in window.state.selected_atoms] == [0, 1]
    window.resize(1040, 680)
    gui.settle(window)
    annotation = plot.vector_annotation
    assert annotation.isVisible()
    assert "G1 current" in annotation.toPlainText()
    assert "G2 current" in annotation.toPlainText()
    axes_bounds = plot.reference_axes_item.sceneBoundingRect()
    raised_bounds = annotation.sceneBoundingRect()
    assert not axes_bounds.intersects(raised_bounds), (axes_bounds, raised_bounds)
    assert raised_bounds.bottom() < axes_bounds.top()
    assert plot.view_box.sceneBoundingRect().contains(raised_bounds)

    open_view_controls(window)
    controls.reference_axes_check.click()
    gui.settle(window)
    assert not plot.reference_axes_item.isVisible()
    lowered_bounds = annotation.sceneBoundingRect()
    assert lowered_bounds.bottom() > raised_bounds.bottom() + axes_bounds.height() / 2
    assert plot.view_box.sceneBoundingRect().contains(lowered_bounds)
    controls.reference_axes_check.click()
    gui.settle(window)
    assert not plot.reference_axes_item.sceneBoundingRect().intersects(
        annotation.sceneBoundingRect()
    )
    assert annotation.sceneBoundingRect().bottom() == pytest.approx(
        raised_bounds.bottom(), abs=1,
    )


def test_reference_axes_are_included_in_png_export_and_obey_visibility(gui, tmp_path):
    window = gui.window(lattice="SC", axis="100", angle_deg=20)
    open_view_controls(window)
    gui.settle(window)
    images = []
    for name, visible in (("shown", True), ("hidden", False), ("restored", True)):
        if window.controls.reference_axes_check.isChecked() != visible:
            window.controls.reference_axes_check.click()
        gui.settle(window)
        path = tmp_path / (name + ".png")
        window.save(path)
        image = QtGui.QImage(str(path)).convertToFormat(QtGui.QImage.Format_RGBA8888)
        assert not image.isNull() and image.width() == 1800
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine(),
        )[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()
        images.append(pixels)
    assert images[0].shape == images[1].shape == images[2].shape
    assert np.count_nonzero(np.any(images[0] != images[1], axis=2)) > 100
    np.testing.assert_array_equal(images[0], images[2])
