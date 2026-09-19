"""Plot items, view transforms and annotation rendering.

PatternPlot owns every graphics item; the window coordinates interaction and
analysis. All scene geometry is derived from the owner's current session data.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from ._qt import QtCore, QtGui, QtWidgets, pg
from . import (
    COINCIDENCE_COLOR,
    BOUNDARY_COLOR,
    VECTOR_COLOR,
    MANUAL_CELL_COLOR,
    NEAR_COLOR,
    LOCAL_COLOR,
)
from .reference_axes import GrainReferenceAxes
from .markers import grain_edge_color, marker_symbol
from ..crystal import (
    layer_name, rotation_matrix_2d,
    in_plane_reference_directions, in_plane_reference_axes,
)
from ..state import BUFFER_FACTOR, VIEW_SCALE_MIN, VIEW_SCALE_MAX
import pyqtgraph.exporters

if TYPE_CHECKING:
    from .window import DichromaticPatternWindow


class PatternViewBox(pg.ViewBox):
    """Pan-mode ViewBox used by the pattern canvas."""

    def __init__(self) -> None:
        super().__init__(lockAspect=True, enableMouse=True, enableMenu=False)
        self.setMouseMode(self.PanMode)


class PatternPlot:
    """Own and render a single PyQtGraph canvas."""

    def __init__(self, owner: DichromaticPatternWindow):
        self.owner = owner
        self._laying_out_title = False
        self._marker_diameter = None
        self._local_overlay_pairs = None
        self._local_overlay_key = None
        self._local_midpoints_view = np.empty((0, 2))
        self._local_distances = np.empty(0)

    def _create_plot(self) -> None:
        self.view_box = PatternViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box)
        self.plot_item = self.plot_widget.getPlotItem()
        # Keep axis strokes inside the viewport, including at fractional DPI.
        self.plot_item.layout.setContentsMargins(8, 8, 12, 8)
        self.owner.setCentralWidget(self.plot_widget)
        self.plot_widget.setBackground("#f8fafc")
        self.plot_item.hideButtons()
        self.plot_item.showGrid(x=True, y=True, alpha=0.16)
        self.plot_item.setLabel("bottom", "x / a₀")
        self.plot_item.setLabel("left", "y / a₀")
        for name in ("bottom", "left"):
            self.plot_item.getAxis(name).enableAutoSIPrefix(False)
        self.plot_item.showAxis("top")
        self.plot_item.showAxis("right")
        for name in ("top", "right"):
            axis = self.plot_item.getAxis(name)
            axis.setStyle(showValues=False, tickLength=0)
            axis.setPen("#94a3b8")
        self.plot_item.getAxis("bottom").setTextPen("#475569")
        self.plot_item.getAxis("left").setTextPen("#475569")
        self.plot_item.getAxis("bottom").setPen("#94a3b8")
        self.plot_item.getAxis("left").setPen("#94a3b8")
        self.view_box.setAspectLocked(True)
        self.view_box.setBorder(pg.mkPen("#94a3b8", width=1.0))
        self.view_box.setLimits(
            minXRange=VIEW_SCALE_MIN * self.owner.state.parameters.width,
            maxXRange=VIEW_SCALE_MAX * self.owner.state.parameters.width,
        )

        diameter = self._base_marker_diameter()
        self.grain_layer_items = [[], []]
        self.coincidence_items = []

        self.boundary_item = pg.PlotCurveItem(pen=pg.mkPen(BOUNDARY_COLOR, width=2.4))
        self.vector_item = pg.PlotCurveItem(pen=pg.mkPen(VECTOR_COLOR, width=2.5))
        self.plot_item.addItem(self.boundary_item)
        self.plot_item.addItem(self.vector_item)
        self.near_cell_item = pg.PlotCurveItem(
            pen=pg.mkPen(NEAR_COLOR, width=2.4, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.plot_item.addItem(self.near_cell_item)
        self.local_match_item = self._ring_item(LOCAL_COLOR, diameter * 2.0)
        self.local_link_item = pg.PlotCurveItem(
            pen=pg.mkPen(LOCAL_COLOR, width=1.4, style=QtCore.Qt.PenStyle.DotLine),
            connect="pairs",
        )
        self.local_match_item.setZValue(3)
        self.local_link_item.setZValue(3)
        self.plot_item.addItem(self.local_match_item)
        self.plot_item.addItem(self.local_link_item)
        self.manual_cell_item = pg.PlotCurveItem(
            pen=pg.mkPen(self.owner.state.grain_colors[0], width=2.4, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.manual_grain_cell_items = [
            self.manual_cell_item,
            pg.PlotCurveItem(
                pen=pg.mkPen(self.owner.state.grain_colors[1], width=2.4, style=QtCore.Qt.PenStyle.DotLine)
            ),
        ]
        self.manual_vertex_item = self._ring_item(MANUAL_CELL_COLOR, diameter * 2.2)
        self.manual_labels = [
            self._text_item(f"C{i+1}", MANUAL_CELL_COLOR) for i in range(4)
        ]
        self.manual_annotation = self._text_item(
            "", MANUAL_CELL_COLOR, font_size=10, bordered=True
        )
        self.manual_annotation.setAnchor((1.0, 1.0))
        for item in (
            *self.manual_grain_cell_items,
            self.manual_vertex_item,
            *self.manual_labels,
            self.manual_annotation,
        ):
            item.setZValue(4)
            self.plot_item.addItem(item)
        for item in (*self.manual_labels, self.manual_annotation):
            item.hide()

        self.boundary_endpoint_item = self._ring_item("#111827", diameter * 2.2)
        self.vector_endpoint_item = self._ring_item(VECTOR_COLOR, diameter * 2.35)
        self.plot_item.addItem(self.boundary_endpoint_item)
        self.plot_item.addItem(self.vector_endpoint_item)

        arrow_pen = pg.mkPen(VECTOR_COLOR, width=1.2)
        arrow_pen.setCosmetic(True)
        self.vector_arrow = QtWidgets.QGraphicsPolygonItem()
        self.vector_arrow.setPen(arrow_pen)
        self.vector_arrow.setBrush(pg.mkBrush(VECTOR_COLOR))
        self.vector_arrow.hide()
        self.plot_item.addItem(self.vector_arrow)

        self.boundary_labels = [self._text_item("B1", "#111827") for _ in range(2)]
        self.vector_labels = [self._text_item("P1", VECTOR_COLOR) for _ in range(2)]
        self.side_labels = [
            self._text_item("L", "#18794e", font_size=15),
            self._text_item("R", "#9333a8", font_size=15),
        ]
        self.vector_annotation = self._text_item(
            "", VECTOR_COLOR, font_size=10, bordered=True
        )
        self.vector_annotation.setAnchor((0.0, 1.0))
        # Keep the complete measured arrow above atoms and manual-cell
        # outlines, especially when the vector lies along a selected cell edge.
        for item in (
            *self.vector_labels,
            self.vector_annotation,
            self.vector_endpoint_item,
        ):
            item.setZValue(7)
        for item in (self.vector_item, self.vector_arrow):
            item.setZValue(6)
        for item in (
            *self.boundary_labels,
            *self.vector_labels,
            *self.side_labels,
            self.vector_annotation,
        ):
            item.hide()
            self.plot_item.addItem(item)

        self.reference_axes_item = GrainReferenceAxes(self.view_box)

        self.legend = self.plot_item.addLegend(offset=(14, 14), colCount=3)
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 225))
        self.legend.setPen(pg.mkPen("#cbd5e1"))
        self._create_layer_items()

        self.plot_widget.scene().sigMouseClicked.connect(
            self.owner._on_scene_mouse_click
        )
        self.view_box.sigRangeChanged.connect(self.owner._on_view_range_changed)
        self.plot_widget.sigDeviceRangeChanged.connect(self._layout_title)
        self.view_box.sigResized.connect(self._layout_title)
        # Tick labels can grow during layout without resizing the ViewBox.
        # Defer a refit until the current layout has finished, since a direct
        # call here can be discarded by _layout_title's reentrancy guard.
        self._title_layout_timer = QtCore.QTimer(self.owner)
        self._title_layout_timer.setSingleShot(True)
        self._title_layout_timer.timeout.connect(self._layout_title)
        for name in ("left", "right"):
            self.plot_item.getAxis(name).geometryChanged.connect(self._title_layout_timer.start)
        self.view_box.sigResized.connect(self._update_reference_axes)

    def _update_reference_axes(self, *_args):
        state = self.owner.state
        indices = in_plane_reference_directions(state.geometry.axis)
        labels = tuple("[" + " ".join(map(str, vector)) + "]" for vector in indices)
        directions = np.stack([
            self._to_view(in_plane_reference_axes(sign * state.angle_deg / 2, deformation))
            for sign, deformation in zip((1, -1), state.deformations)
        ])
        item = self.reference_axes_item
        item.set_colors(state.grain_colors)
        item.set_reference(directions, labels)
        item.setVisible(state.show_reference_axes)
        bounds = self.view_box.boundingRect()
        # Parent directly to the ViewBox, outside its data transform. A small
        # viewport may scale the panel down, but pan/zoom never changes its size.
        scale = min(1.0, max(1.0, bounds.width() - 28) / item.boundingRect().width())
        item.setScale(scale)
        item.setPos(bounds.left() + 14, bounds.bottom() - 14 - scale * item.boundingRect().height())
        self._position_vector_annotation()

    def _position_vector_annotation(self):
        if not self.vector_annotation.isVisible():
            return
        bounds = self.view_box.boundingRect()
        self.vector_annotation.setTextWidth(-1)
        natural_width = self.vector_annotation.textItem.boundingRect().width()
        self.vector_annotation.setTextWidth(min(natural_width, max(1.0, bounds.width() - 28)))
        bottom = bounds.bottom() - 14
        if self.reference_axes_item.isVisible():
            bottom = self.reference_axes_item.pos().y() - 10
        position = self.view_box.mapToView(QtCore.QPointF(bounds.left() + 14, bottom))
        self.vector_annotation.setPos(position)

    def _set_title(self, text):
        self.plot_item.setTitle(text)
        self._layout_title()

    def _layout_title(self, *_args):
        title = self.plot_item.titleLabel
        if self._laying_out_title or not title.isVisible():
            return
        self._laying_out_title = True
        try:
            # LabelItem otherwise makes the entire plot at least as wide as
            # its unwrapped title. Measure the actual viewport, since the
            # title and ViewBox may already have grown beyond that viewport.
            layout = self.plot_item.layout
            left, _top, right, _bottom = layout.getContentsMargins()
            width = max(
                1.0,
                self.plot_widget.viewport().width() - left - right
                - self.plot_item.getAxis("left").width()
                - self.plot_item.getAxis("right").width()
                - 2 * layout.horizontalSpacing(),
            )
            document = title.item.document()
            option = document.defaultTextOption()
            option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
            option.setAlignment(QtCore.Qt.AlignHCenter)
            document.setDefaultTextOption(option)
            title.item.setTextWidth(width)
            height = max(30.0, float(np.ceil(title.item.boundingRect().height())))
            title.setMaximumHeight(height)
            layout.setRowFixedHeight(0, height)
            title.updateMin()
            # Reducing the title's minimum width alone does not shrink an
            # already enlarged central item until GraphicsView is resized.
            self.plot_item.setGeometry(self.plot_widget.sceneRect())
            layout.activate()
            title.resizeEvent(None)
        finally:
            self._laying_out_title = False

    def _create_layer_items(self):
        # Newly created items need sizing even if the view scale is unchanged.
        self._marker_diameter = None
        self.legend.clear()
        for item in [
            *self.grain_layer_items[0],
            *self.grain_layer_items[1],
            *self.coincidence_items,
        ]:
            self.plot_item.removeItem(item)
        self.grain_layer_items = [[], []]
        self.coincidence_items = []
        diameter = self._base_marker_diameter()
        for layer in range(self.owner.state.geometry.layer_count):
            symbol = marker_symbol(self.owner.state.layer_symbols[layer])
            for grain in (0, 1):
                color = self.owner.state.grain_colors[grain]
                brush_color = QtGui.QColor(color)
                brush_color.setAlpha(185 if grain == 0 else 0)
                item = pg.ScatterPlotItem(
                    size=diameter * (1 if grain == 0 else 1.12),
                    symbol=symbol,
                    pen=pg.mkPen(
                        grain_edge_color(color, grain),
                        width=0.8 if grain == 0 else 1.5,
                    ),
                    brush=pg.mkBrush(brush_color),
                    pxMode=True,
                    clickable=False,
                )
                item.setZValue(1)
                self.grain_layer_items[grain].append(item)
                self.plot_item.addItem(item)
            item = pg.ScatterPlotItem(
                size=diameter * 2.3,
                symbol=symbol,
                pen=pg.mkPen(COINCIDENCE_COLOR, width=2.2),
                brush=pg.mkBrush(0, 0, 0, 0),
                pxMode=True,
                clickable=False,
            )
            item.setZValue(2)
            self.coincidence_items.append(item)
            self.plot_item.addItem(item)
        self._rebuild_legend()

    def refresh_appearance(self):
        """Restyle existing items without regenerating atoms or changing picks."""
        state = self.owner.state
        for grain, items in enumerate(self.grain_layer_items):
            color = state.grain_colors[grain]
            brush_color = QtGui.QColor(color)
            brush_color.setAlpha(185 if grain == 0 else 0)
            pen = pg.mkPen(grain_edge_color(color, grain), width=0.8 if grain == 0 else 1.5)
            brush = pg.mkBrush(brush_color)
            for layer, item in enumerate(items):
                item.setPen(pen)
                item.setBrush(brush)
                item.setSymbol(marker_symbol(state.layer_symbols[layer]))
        for layer, item in enumerate(self.coincidence_items):
            item.setSymbol(marker_symbol(state.layer_symbols[layer]))
        for color, item, style in zip(
            state.grain_colors, self.manual_grain_cell_items,
            (QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine),
        ):
            item.setPen(pg.mkPen(color, width=2.4, style=style))
        self._local_overlay_key = None
        self._draw_manual_cell()
        self._update_local_overlay()
        self._update_reference_axes()
        self._rebuild_legend()

    def _rebuild_legend(self):
        self.legend.clear()
        layers = sorted(set().union(*self.owner.state.visible_grain_layers))[:6]
        for layer in layers:
            name = layer_name(layer)
            if layer in self.owner.state.visible_grain_layers[0]:
                self.legend.addItem(self.grain_layer_items[0][layer], f"G1 · {name}")
            if layer in self.owner.state.visible_grain_layers[1]:
                self.legend.addItem(self.grain_layer_items[1][layer], f"G2 · {name}")
            if layer in self.owner.state.visible_layers:
                self.legend.addItem(
                    self.coincidence_items[layer], f"CSL · {name}–{name}"
                )
        if (
            not hasattr(self.owner.controls, "cell_check")
            or self.owner.controls.cell_check.isChecked()
        ):
            self.legend.addItem(self.near_cell_item, "Common periodic cell")
        if self.owner.local_active:
            self.legend.addItem(self.local_match_item, "Local near pair · midpoint")

    @staticmethod
    def _ring_item(color: str, size: float) -> pg.ScatterPlotItem:
        return pg.ScatterPlotItem(
            size=size,
            symbol="o",
            pen=pg.mkPen(color, width=2.0),
            brush=pg.mkBrush(0, 0, 0, 0),
            pxMode=True,
            clickable=False,
        )

    @staticmethod
    def _text_item(
        text: str,
        color: str,
        font_size: int = 10,
        bordered: bool = False,
    ) -> pg.TextItem:
        item = pg.TextItem(
            text=text,
            color=color,
            anchor=(0.5, 1.15),
            border=pg.mkPen(color, width=1.0) if bordered else None,
            fill=pg.mkBrush(255, 255, 255, 235) if bordered else None,
        )
        font = QtGui.QFont()
        font.setPointSize(font_size)
        font.setBold(not bordered)
        item.setFont(font)
        return item

    def _base_marker_diameter(self) -> float:
        return max(5.0, float(np.sqrt(self.owner.state.parameters.marker_size) * 1.55))

    def _to_view(self, points):
        return (
            np.asarray(points)
            @ rotation_matrix_2d(self.owner.state.display_rotation_deg).T
        )

    def _from_view(self, points):
        return np.asarray(points) @ rotation_matrix_2d(
            self.owner.state.display_rotation_deg
        )

    def _model_view_bounds(self):
        x0, x1, y0, y1 = self._view_range()
        corners = self._from_view(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        low, high = corners.min(axis=0), corners.max(axis=0)
        return low[0], high[0], low[1], high[1]

    def _draw_manual_cell(self):
        self.owner._sync_manual_strain_controls()
        points = np.array(
            [v.position for v in self.owner.state.manual_vertices]
        ).reshape(-1, 2)
        if self.owner.state.manual_vertices:
            self.manual_vertex_item.setSymbol(
                marker_symbol(self.owner.state.layer_symbols[
                    self.owner.state.manual_vertices[0].layer
                ])
            )
        self._set_scatter(self.manual_vertex_item, points)
        displayed = self._to_view(points)
        for polygon, item in zip(
            self.owner._manual_grain_polygons(), self.manual_grain_cell_items
        ):
            grain_view = self._to_view(polygon)
            outline = (
                np.vstack((grain_view, grain_view[0]))
                if len(points) == 4
                else grain_view
            )
            item.setData(outline[:, 0], outline[:, 1])
        self.owner.controls.manual_fit_button.setEnabled(len(points) == 4)
        for index, label in enumerate(self.manual_labels):
            label.setVisible(index < len(points))
            if index < len(points):
                vertex = self.owner.state.manual_vertices[index]
                label.setText(
                    f"C{index+1} · {self.owner._cell_layer_label(vertex.layer).split()[0]} · "
                    f"{vertex.source}"
                )
                label.setPos(*displayed[index])
        self.manual_annotation.setVisible(len(points) == 4)
        if len(points) == 4:
            if self.owner.state.manual_counts is None:
                label = (
                    "Manual cell · count unavailable"
                    if self.owner.state.manual_count_error
                    else "Manual cell · counting…"
                )
            else:
                counts = self.owner.state.manual_counts
                layer = self.owner.state.manual_vertices[0].layer
                lines = [f"Manual cell · {self.owner._cell_layer_label(layer)} only"]
                for grain in (0, 1):
                    interior = counts.interior[grain, layer]
                    if counts.half_open_available[grain]:
                        total = counts.half_open[grain, layer]
                        edge = counts.half_open_edges[grain, layer]
                        corner = counts.half_open_corners[grain, layer]
                        lines.append(
                            f"G{grain+1}: {total} atoms · {interior} inside + "
                            f"{edge} edge + {corner} corner"
                        )
                    else:
                        boundary = counts.boundary[grain, layer]
                        lines.append(
                            f"G{grain+1}: {interior+boundary} atoms · "
                            f"{interior} inside + {boundary} boundary"
                        )
                label = "\n".join(lines)
            self.manual_annotation.setText(label)
            x_min, x_max, y_min, y_max = self._view_range()
            self.manual_annotation.setPos(
                x_max - 0.025 * (x_max - x_min),
                y_min + 0.035 * (y_max - y_min),
            )

    def _update_common_cell(self, *_args):
        cell = self.owner.common_cell
        self.owner.controls.cell_fit_button.setEnabled(cell is not None)
        if cell is None:
            self.near_cell_item.setData([], [])
        else:
            corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ cell.cell.T
            if self.owner.state.manual_strain_fit is not None:
                corners += self.owner.state.manual_strain_fit.origin
            corners = self._to_view(corners)
            self.near_cell_item.setData(corners[:, 0], corners[:, 1])
        self.near_cell_item.setVisible(
            cell is not None and self.owner.controls.cell_check.isChecked()
        )

    def _update_local_overlay(self):
        """Filter both original endpoints, never just the proposed midpoint."""
        pairs = self.owner.state.local_pairs
        session = self.owner.state
        # Pan/zoom changes the viewport, not the buffered pair geometry. Keep
        # the rendered points and distances until a physical/display filter
        # changes; setData rebuilds every scatter symbol's cached rendering.
        key = (
            self.owner.local_active,
            session.display_rotation_deg,
            tuple(sorted(session.visible_layers)),
            tuple(session.layer_symbols),
            tuple(np.asarray(session.selected_points).ravel()),
            self.owner._region_states(),
            session.grain_signature == self.owner._geometry_signature(),
            session.buffer_bounds,
            self.owner.controls.local_distance_spin.value(),
        )
        if pairs is not self._local_overlay_pairs or key != self._local_overlay_key:
            mask = self.owner._local_pair_mask()
            self._local_midpoints_view = self._to_view(pairs.midpoints[mask])
            self._local_distances = pairs.distances[mask]
            self.local_match_item.setData(
                pos=self._local_midpoints_view,
                symbol=[
                    marker_symbol(session.layer_symbols[int(layer)])
                    for layer in pairs.layers[mask]
                ],
            )
            links = np.stack(
                (pairs.first[mask], pairs.second[mask]), axis=1
            ).reshape(-1, 2)
            links = self._to_view(links)
            self.local_link_item.setData(links[:, 0], links[:, 1], connect="pairs")
            self._local_overlay_pairs = pairs
            self._local_overlay_key = key
        if not self.owner.local_active:
            return
        x0, x1, y0, y1 = self._view_range()
        midpoints = self._local_midpoints_view
        visible = (
            (midpoints[:, 0] >= x0)
            & (midpoints[:, 0] <= x1)
            & (midpoints[:, 1] >= y0)
            & (midpoints[:, 1] <= y1)
        )
        distances = self._local_distances[visible]
        if self.owner.state.local_error:
            state = "Local matching failed: " + self.owner.state.local_error
        elif self.owner.state.local_updating:
            state = f"Matching same-layer neighbors… ({self.owner.compute.worker_count} workers)"
        else:
            state = f"Visible near pairs: {len(distances)} (exact CSL excluded)"
            if len(distances):
                state += f"\nd/a₀ min / mean / max: {distances.min():.5f} / {distances.mean():.5f} / {distances.max():.5f}"
        text = (
            state
            + f"\nPair cutoff: {self.owner.controls.local_distance_spin.value():.4f} a₀."
            "\nSame-layer mutual nearest pairs; no atoms moved."
            "\nPurple marker: midpoint, shaped by layer; line: original pair."
            "\nMidpoint alignment would move each atom by d/2."
            "\nNo bulk strain, relaxation or stress calculation."
            "\nLocal matches do not define a periodic common cell."
        )
        if text != self.owner.controls.near_info.toPlainText():
            self.owner.controls.near_info.setPlainText(text)

    def _fit_model_corners(self, corners):
        corners = self._to_view(corners)
        center = (corners.min(axis=0) + corners.max(axis=0)) / 2
        extent = np.ptp(corners, axis=0) * 1.15
        _, old_width, old_height = self._view_geometry()
        aspect = old_height / old_width
        width = np.clip(
            max(extent[0], extent[1] / aspect),
            self.owner.state.parameters.width * VIEW_SCALE_MIN,
            self.owner.state.parameters.width * VIEW_SCALE_MAX,
        )
        height = width * aspect
        self.view_box.setRange(
            xRange=(center[0] - width / 2, center[0] + width / 2),
            yRange=(center[1] - height / 2, center[1] + height / 2),
            padding=0,
        )

    def _set_initial_view(self) -> None:
        width = (
            self.owner.state.parameters.width * self.owner.state.parameters.view_scale
        )
        height = (
            self.owner.state.parameters.height * self.owner.state.parameters.view_scale
        )
        self.view_box.setRange(
            xRange=(-0.5 * width, 0.5 * width),
            yRange=(-0.5 * height, 0.5 * height),
            padding=0.0,
        )

    def _view_range(self) -> tuple[float, float, float, float]:
        (x_min, x_max), (y_min, y_max) = self.view_box.viewRange()
        return float(x_min), float(x_max), float(y_min), float(y_max)

    def _view_geometry(self) -> tuple[np.ndarray, float, float]:
        x_min, x_max, y_min, y_max = self._view_range()
        return (
            np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5]),
            x_max - x_min,
            y_max - y_min,
        )

    def _buffer_geometry(
        self,
    ) -> tuple[np.ndarray, float, float, tuple[float, float, float, float]]:
        x0, x1, y0, y1 = self._model_view_bounds()
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        view_width, view_height = x1 - x0, y1 - y0
        width = BUFFER_FACTOR * view_width
        height = BUFFER_FACTOR * view_height
        if self.owner.local_active:
            # Both directions of a mutual-neighbor query need a 2*d halo.
            halo_width = 4.4 * self.owner.controls.local_distance_spin.value()
            width = max(width, view_width + halo_width)
            height = max(height, view_height + halo_width)
        bounds = (
            center[0] - 0.5 * width,
            center[0] + 0.5 * width,
            center[1] - 0.5 * height,
            center[1] + 0.5 * height,
        )
        return center, width, height, bounds

    def _buffer_contains_view(self) -> bool:
        if self.owner.state.buffer_bounds is None:
            return False
        x_min, x_max, y_min, y_max = self._model_view_bounds()
        bx_min, bx_max, by_min, by_max = self.owner.state.buffer_bounds
        margin_x = 0.10 * (bx_max - bx_min)
        margin_y = 0.10 * (by_max - by_min)
        if self.owner.local_active:
            margin_x = max(
                margin_x, 2 * self.owner.controls.local_distance_spin.value()
            )
            margin_y = max(
                margin_y, 2 * self.owner.controls.local_distance_spin.value()
            )
        return (
            x_min >= bx_min + margin_x
            and x_max <= bx_max - margin_x
            and y_min >= by_min + margin_y
            and y_max <= by_max - margin_y
        )

    def _update_marker_sizes(self) -> None:
        _center, width, _height = self._view_geometry()
        scale = max(VIEW_SCALE_MIN, width / self.owner.state.parameters.width)
        diameter = max(3.5, self._base_marker_diameter() / scale**0.22)
        if (
            self._marker_diameter is not None
            and abs(diameter - self._marker_diameter) <= 1e-12 * diameter
        ):
            return
        self._marker_diameter = diameter
        for grain, grain_items in enumerate(self.grain_layer_items):
            for layer, item in enumerate(grain_items):
                item.setSize(diameter * (1 + 0.12 * grain + 0.05 * (layer % 2)))
        for layer, item in enumerate(self.coincidence_items):
            item.setSize(diameter * (2.3 + 0.15 * (layer % 2)))
        self.boundary_endpoint_item.setSize(diameter * 2.2)
        self.vector_endpoint_item.setSize(diameter * 2.35)
        self.local_match_item.setSize(diameter * 2.0)
        self.manual_vertex_item.setSize(diameter * 2.2)

    def _set_scatter(self, item: pg.ScatterPlotItem, points: np.ndarray) -> None:
        if len(points):
            item.setData(pos=self._to_view(points))
        else:
            item.setData(x=[], y=[])

    def _draw_boundary(self) -> None:
        if self.owner.state.selected_points:
            points = np.asarray(self.owner.state.selected_points)
            self._set_scatter(self.boundary_endpoint_item, points)
            for label, point in zip(self.boundary_labels, self._to_view(points)):
                label.setPos(*point)
        if len(self.owner.state.selected_points) != 2:
            return
        first, second = self._to_view(self.owner.state.selected_points)
        intersections = self._line_box_intersections(first, second)
        if len(intersections) >= 2:
            line = np.asarray(intersections[:2])
            self.boundary_item.setData(line[:, 0], line[:, 1])
        direction = second - first
        normal = np.array([-direction[1], direction[0]])
        length = float(np.linalg.norm(normal))
        if length == 0.0:
            return
        normal /= length
        _center, width, height = self._view_geometry()
        offset = 0.10 * min(width, height)
        midpoint = 0.5 * (first + second)
        for item, point in zip(
            self.side_labels,
            (midpoint + normal * offset, midpoint - normal * offset),
            strict=True,
        ):
            item.setPos(*self._clamp_to_view(point))
            item.show()

    def _draw_vector(self) -> None:
        self._update_reference_axes()
        if self.owner.state.selected_atoms:
            points = np.array(
                [atom.position for atom in self.owner.state.selected_atoms]
            )
            self._set_scatter(self.vector_endpoint_item, points)
            for label, point in zip(self.vector_labels, self._to_view(points)):
                label.setPos(*point)
        if len(self.owner.state.selected_atoms) != 2:
            return
        first = self._to_view(self.owner.state.selected_atoms[0].position)
        second = self._to_view(self.owner.state.selected_atoms[1].position)
        self.vector_item.setData([first[0], second[0]], [first[1], second[1]])
        direction = second - first
        pixel_size = np.maximum(np.abs(self.view_box.viewPixelSize()), 1e-12)
        pixel_direction = direction / pixel_size
        pixel_length = float(np.linalg.norm(pixel_direction))
        if pixel_length <= 1e-12:
            self.vector_arrow.hide()
            return
        unit = pixel_direction / pixel_length
        _center, width, _height = self._view_geometry()
        scale = max(VIEW_SCALE_MIN, width / self.owner.state.parameters.width)
        marker_diameter = max(3.5, self._base_marker_diameter() / scale**0.22)
        # Put the tip just outside the P2 selection ring. Building the triangle
        # in pixel coordinates keeps it clear and equally legible at every zoom.
        tip = second - unit * pixel_size * (0.5 * 2.35 * marker_diameter + 2.0)
        base = tip - unit * pixel_size * 18.0
        perpendicular = np.array([-unit[1], unit[0]])
        half_width = perpendicular * pixel_size * 7.0
        polygon = QtGui.QPolygonF(
            [
                QtCore.QPointF(*tip),
                QtCore.QPointF(*(base + half_width)),
                QtCore.QPointF(*(base - half_width)),
            ]
        )
        self.vector_arrow.setPolygon(polygon)
        self.vector_arrow.show()
        self.vector_annotation.setText(self.owner._selected_vector_readout())
        self.vector_annotation.show()
        self._position_vector_annotation()
        self.owner._request_vector_annotation_repaint()

    def _line_box_intersections(
        self, first: np.ndarray, second: np.ndarray
    ) -> list[np.ndarray]:
        direction = second - first
        x_min, x_max, y_min, y_max = self._view_range()
        candidates: list[np.ndarray] = []
        if abs(direction[0]) > 1.0e-12:
            for x_value in (x_min, x_max):
                parameter = (x_value - first[0]) / direction[0]
                y_value = first[1] + parameter * direction[1]
                if y_min - 1.0e-9 <= y_value <= y_max + 1.0e-9:
                    candidates.append(np.array([x_value, y_value]))
        if abs(direction[1]) > 1.0e-12:
            for y_value in (y_min, y_max):
                parameter = (y_value - first[1]) / direction[1]
                x_value = first[0] + parameter * direction[0]
                if x_min - 1.0e-9 <= x_value <= x_max + 1.0e-9:
                    candidates.append(np.array([x_value, y_value]))
        unique: list[np.ndarray] = []
        for candidate in candidates:
            if not any(np.linalg.norm(candidate - old) < 1.0e-7 for old in unique):
                unique.append(candidate)
        return unique[:2]

    def _clamp_to_view(self, point: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max = self._view_range()
        return np.array(
            [
                np.clip(
                    point[0],
                    x_min + 0.04 * (x_max - x_min),
                    x_max - 0.04 * (x_max - x_min),
                ),
                np.clip(
                    point[1],
                    y_min + 0.06 * (y_max - y_min),
                    y_max - 0.06 * (y_max - y_min),
                ),
            ]
        )

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exporter = pyqtgraph.exporters.ImageExporter(self.plot_item)
        exporter.parameters()["width"] = 1800
        exporter.export(str(output_path))
