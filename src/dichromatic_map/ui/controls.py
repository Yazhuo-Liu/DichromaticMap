"""Control dock, reusable widgets and application theme.

ControlDock owns its widgets. Its owner coordinates user actions and results;
the control classes never generate atoms or own process pools.
"""

from __future__ import annotations
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from . import (
    GRAIN_1_COLOR,
    GRAIN_1_EDGE,
    GRAIN_2_COLOR,
    LAYER_SYMBOLS,
)
from ..crystal import layer_name
from ..state import VIEW_SCALE_MIN, VIEW_SCALE_MAX, VIEW_SCALE_STOPS
from ..strain import DEFAULT_STRAIN_PERCENT, DEFAULT_SEARCH_INDEX
from ..matching import DEFAULT_LOCAL_DISTANCE

if TYPE_CHECKING:
    from .window import DichromaticPatternWindow


APPLICATION_STYLESHEET = (
    Path(__file__).with_name("resources") / "theme.qss"
).read_text(encoding="utf-8")


class CollapsibleSection(QtWidgets.QWidget):
    """Compact control-dock section with an explicit expanded state."""

    def __init__(
        self,
        title: str,
        layout_type=QtWidgets.QVBoxLayout,
        expanded: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("sectionToggle")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.body = QtWidgets.QFrame()
        self.body.setObjectName("sectionBody")
        self.content_layout = layout_type(self.body)
        self.toggle.toggled.connect(self._set_expanded)
        outer.addWidget(self.toggle)
        outer.addWidget(self.body)
        self._set_expanded(expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if expanded
            else QtCore.Qt.ArrowType.RightArrow
        )
        self.body.setVisible(expanded)

    def isExpanded(self) -> bool:  # noqa: N802 - match Qt naming
        return self.toggle.isChecked()


class CurrentPageTabWidget(QtWidgets.QTabWidget):
    """A tab widget whose height follows the page that is currently visible."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.currentChanged.connect(self._current_page_changed)

    def _height_for_current_page(self, base: QtCore.QSize, minimum=False):
        if self.count() == 0 or self.currentWidget() is None:
            return base
        hint_name = "minimumSizeHint" if minimum else "sizeHint"
        page_heights = [
            max(0, getattr(self.widget(index), hint_name)().height())
            for index in range(self.count())
        ]
        current_height = page_heights[self.currentIndex()]
        chrome_height = max(0, base.height() - max(page_heights, default=0))
        result = QtCore.QSize(base)
        result.setHeight(chrome_height + current_height)
        return result

    def sizeHint(self):  # noqa: N802 - Qt override
        return self._height_for_current_page(super().sizeHint())

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        return self._height_for_current_page(super().minimumSizeHint(), minimum=True)

    def syncCurrentPageGeometry(self) -> None:  # noqa: N802 - Qt-style helper
        self._current_page_changed(self.currentIndex())

    def _current_page_changed(self, current_index: int) -> None:
        for index in range(self.count()):
            page = self.widget(index)
            policy = page.sizePolicy()
            policy.setVerticalPolicy(
                QtWidgets.QSizePolicy.Policy.Preferred
                if index == current_index
                else QtWidgets.QSizePolicy.Policy.Ignored
            )
            page.setSizePolicy(policy)
        self.updateGeometry()
        QtCore.QTimer.singleShot(0, self._refresh_parent_layout)

    def _refresh_parent_layout(self) -> None:
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().invalidate()
            parent.layout().activate()
            parent.updateGeometry()


def layer_marker_icon(grain: int, layer: int, size: int = 18) -> QtGui.QIcon:
    """Return a small marker matching the grain color and plotted layer shape."""

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    edge = QtGui.QColor(GRAIN_1_EDGE if grain == 0 else GRAIN_2_COLOR)
    fill = QtGui.QColor(GRAIN_1_COLOR)
    fill.setAlpha(190)
    painter.setPen(QtGui.QPen(edge, 1.5))
    painter.setBrush(QtGui.QBrush(fill) if grain == 0 else QtCore.Qt.BrushStyle.NoBrush)

    center = QtCore.QPointF(size / 2, size / 2)
    radius = size * 0.31
    symbol = LAYER_SYMBOLS[layer % len(LAYER_SYMBOLS)]

    def polygon(sides: int, rotation: float) -> QtGui.QPolygonF:
        return QtGui.QPolygonF(
            [
                QtCore.QPointF(
                    center.x() + radius * math.cos(rotation + 2 * math.pi * i / sides),
                    center.y() + radius * math.sin(rotation + 2 * math.pi * i / sides),
                )
                for i in range(sides)
            ]
        )

    if symbol == "o":
        painter.drawEllipse(center, radius, radius)
    elif symbol == "d":
        painter.drawPolygon(polygon(4, 0.0))
    elif symbol in {"t", "t1", "t2", "t3"}:
        rotations = {
            "t": -math.pi / 2,
            "t1": math.pi / 2,
            "t2": 0.0,
            "t3": math.pi,
        }
        painter.drawPolygon(polygon(3, rotations[symbol]))
    elif symbol == "s":
        painter.drawPolygon(polygon(4, math.pi / 4))
    elif symbol in {"p", "h"}:
        painter.drawPolygon(polygon(5 if symbol == "p" else 6, -math.pi / 2))
    elif symbol == "star":
        points = []
        for index in range(10):
            point_radius = radius if index % 2 == 0 else radius * 0.43
            angle = -math.pi / 2 + index * math.pi / 5
            points.append(
                QtCore.QPointF(
                    center.x() + point_radius * math.cos(angle),
                    center.y() + point_radius * math.sin(angle),
                )
            )
        painter.drawPolygon(QtGui.QPolygonF(points))
    else:
        diagonal = symbol == "x"
        offset = radius / math.sqrt(2) if diagonal else radius
        if diagonal:
            painter.drawLine(
                QtCore.QPointF(center.x() - offset, center.y() - offset),
                QtCore.QPointF(center.x() + offset, center.y() + offset),
            )
            painter.drawLine(
                QtCore.QPointF(center.x() - offset, center.y() + offset),
                QtCore.QPointF(center.x() + offset, center.y() - offset),
            )
        else:
            painter.drawLine(
                QtCore.QPointF(center.x() - radius, center.y()),
                QtCore.QPointF(center.x() + radius, center.y()),
            )
            painter.drawLine(
                QtCore.QPointF(center.x(), center.y() - radius),
                QtCore.QPointF(center.x(), center.y() + radius),
            )
    painter.end()
    return QtGui.QIcon(pixmap)


class ControlDock:
    """Own the control widgets and connect them to the window coordinator."""

    def __init__(self, owner: DichromaticPatternWindow):
        self.owner = owner

    def _create_control_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Controls", self.owner)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setMinimumWidth(320)
        dock.setMaximumWidth(410)
        self.owner.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        panel = QtWidgets.QWidget()
        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        scroll.setWidget(panel)
        dock.setWidget(scroll)

        self.heading = QtWidgets.QLabel()
        layout.addWidget(self.heading)

        crystal_box = QtWidgets.QGroupBox("CRYSTAL / AXIS")
        crystal_layout = QtWidgets.QFormLayout(crystal_box)
        self.structure_combo = QtWidgets.QComboBox()
        for lattice in ("FCC", "BCC"):
            self.structure_combo.addItem(lattice, lattice)
        self.structure_combo.setCurrentIndex(
            self.structure_combo.findData(self.owner.state.geometry.lattice)
        )
        self.axis_combo = QtWidgets.QComboBox()
        for axis in ("100", "110", "111", "112"):
            self.axis_combo.addItem(f"⟨{axis}⟩", axis)
        self.axis_combo.addItem("Custom [h k l]", None)
        axis_index = self.axis_combo.findData(self.owner.state.geometry.axis)
        self.axis_combo.setCurrentIndex(axis_index if axis_index >= 0 else 4)
        crystal_layout.addRow("Structure", self.structure_combo)
        crystal_layout.addRow("Tilt / viewing axis", self.axis_combo)
        self.custom_axis_row = QtWidgets.QWidget()
        custom_layout = QtWidgets.QHBoxLayout(self.custom_axis_row)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_axis_edit = QtWidgets.QLineEdit(
            " ".join(map(str, self.owner.state.geometry.axis_indices))
        )
        self.custom_axis_edit.setPlaceholderText("e.g. 1 -1 3")
        self.custom_axis_edit.setToolTip(
            "Integer [h k l]; separate signed or multi-digit indices with spaces."
        )
        self.custom_axis_apply = QtWidgets.QPushButton("Apply axis")
        custom_layout.addWidget(self.custom_axis_edit, 1)
        custom_layout.addWidget(self.custom_axis_apply)
        crystal_layout.addRow(self.custom_axis_row)
        self.custom_axis_row.setVisible(axis_index < 0)
        self.axis_error = QtWidgets.QLabel()
        self.axis_error.setStyleSheet("color: #b42318;")
        self.axis_error.setWordWrap(True)
        self.axis_error.hide()
        crystal_layout.addRow(self.axis_error)
        self.layer_info = QtWidgets.QLabel()
        self.layer_info.setWordWrap(True)
        self.layer_info.setObjectName("mutedLabel")
        crystal_layout.addRow(self.layer_info)
        self.layer_combo = QtWidgets.QComboBox()
        self._populate_layer_combo()
        self.axial_layer_label = QtWidgets.QLabel("Visible axial layers")
        crystal_layout.addRow(self.axial_layer_label, self.layer_combo)
        self.layer_combo.currentIndexChanged.connect(self.owner._on_layer_changed)
        self.axis_combo.currentIndexChanged.connect(self.owner._on_geometry_changed)
        self.structure_combo.currentIndexChanged.connect(
            lambda: self.owner._on_geometry_changed(apply_custom=True)
        )
        self.custom_axis_apply.clicked.connect(
            lambda: self.owner._on_geometry_changed(apply_custom=True)
        )
        self.custom_axis_edit.returnPressed.connect(
            lambda: self.owner._on_geometry_changed(apply_custom=True)
        )
        layout.addWidget(crystal_box)

        self.interaction_section = CollapsibleSection(
            "GB / VECTOR", QtWidgets.QGridLayout, expanded=True
        )
        interaction_box = self.interaction_section
        interaction_layout = interaction_box.content_layout
        self.pick_gb_button = QtWidgets.QPushButton("Pick GB   R")
        self.vector_button = QtWidgets.QPushButton("Measure vector   V")
        self.vector_button.setToolTip(
            "Pick P1/P2 in either grain. Cross-grain picks show both G1 and G2 representations. "
            "Current (polar): actual components / a0 in the grain's polar-rotated orthonormal frame. "
            "Lattice [uvw]: coefficients in its deformed conventional basis; these can remain unchanged under strain."
        )
        self.pick_gb_button.setCheckable(True)
        self.vector_button.setCheckable(True)
        self.pick_gb_button.clicked.connect(self.owner._start_new_boundary)
        self.vector_button.clicked.connect(self.owner._start_vector_measurement)
        interaction_layout.addWidget(self.pick_gb_button, 0, 0)
        interaction_layout.addWidget(self.vector_button, 0, 1)

        self.full_button = QtWidgets.QPushButton("Full pattern   F")
        self.center_button = QtWidgets.QPushButton("Center view   C")
        self.first_bicrystal_button = QtWidgets.QPushButton("G1:L  /  G2:R")
        self.second_bicrystal_button = QtWidgets.QPushButton("G1:R  /  G2:L")
        self.full_button.clicked.connect(
            lambda: self.owner._set_region_states((True, True, True, True))
        )
        self.center_button.clicked.connect(self.owner._reset_view)
        self.first_bicrystal_button.clicked.connect(
            lambda: self.owner._set_region_states((True, False, False, True))
        )
        self.second_bicrystal_button.clicked.connect(
            lambda: self.owner._set_region_states((False, True, True, False))
        )
        interaction_layout.addWidget(self.full_button, 1, 0)
        interaction_layout.addWidget(self.center_button, 1, 1)
        interaction_layout.addWidget(self.first_bicrystal_button, 2, 0)
        interaction_layout.addWidget(self.second_bicrystal_button, 2, 1)
        layout.addWidget(interaction_box)

        self.layers_section = CollapsibleSection(
            "LAYERS", QtWidgets.QGridLayout, expanded=True
        )
        visibility_box = self.layers_section
        visibility_layout = visibility_box.content_layout
        self.region_checks: list[QtWidgets.QCheckBox] = []
        check_specs = (
            ("G1 · Left", GRAIN_1_COLOR),
            ("G1 · Right", GRAIN_1_COLOR),
            ("G2 · Left", GRAIN_2_COLOR),
            ("G2 · Right", GRAIN_2_COLOR),
        )
        for index, (label, color) in enumerate(check_specs):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(True)
            check.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            check.stateChanged.connect(self.owner._update_visible_points)
            visibility_layout.addWidget(check, index // 2, index % 2)
            self.region_checks.append(check)
        layout.addWidget(visibility_box)

        self.orientation_section = CollapsibleSection(
            "ORIENTATION", QtWidgets.QVBoxLayout, expanded=True
        )
        orientation_box = self.orientation_section
        orientation_layout = orientation_box.content_layout
        orientation_layout.addWidget(QtWidgets.QLabel("CSL preset"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Custom angle", None)
        for index, preset in enumerate(self.owner.state.presets):
            self.preset_combo.addItem(preset.label, index)
        self.preset_combo.currentIndexChanged.connect(self.owner._on_preset_changed)
        orientation_layout.addWidget(self.preset_combo)

        angle_row = QtWidgets.QHBoxLayout()
        angle_row.addWidget(QtWidgets.QLabel("Misorientation"))
        angle_row.addStretch(1)
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(0.0, 90.0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setSingleStep(0.1)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setValue(self.owner.state.angle_deg)
        self.angle_spin.valueChanged.connect(self.owner._queue_angle_update)
        angle_row.addWidget(self.angle_spin)
        orientation_layout.addLayout(angle_row)

        self.angle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.angle_slider.setRange(0, 9000)
        self.angle_slider.setSingleStep(1)
        self.angle_slider.setPageStep(100)
        self.angle_slider.setValue(round(self.owner.state.angle_deg * 100.0))
        self.angle_slider.valueChanged.connect(
            lambda value: self.owner._queue_angle_update(value / 100.0)
        )
        self.angle_slider.sliderReleased.connect(self.owner._finish_angle_update)
        orientation_layout.addWidget(self.angle_slider)
        self.angle_exact_label = QtWidgets.QLabel()
        self.angle_exact_label.setObjectName("mutedLabel")
        orientation_layout.addWidget(self.angle_exact_label)
        rotation_row = QtWidgets.QHBoxLayout()
        rotation_row.addWidget(QtWidgets.QLabel("Display rotation"))
        self.rotation_spin = QtWidgets.QDoubleSpinBox()
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setSingleStep(1)
        self.rotation_spin.valueChanged.connect(self.owner._on_display_rotation)
        rotation_row.addWidget(self.rotation_spin)
        rotation_reset = QtWidgets.QToolButton()
        rotation_reset.setText("0°")
        rotation_reset.clicked.connect(lambda: self.rotation_spin.setValue(0))
        rotation_row.addWidget(rotation_reset)
        orientation_layout.addLayout(rotation_row)
        self.rotation_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-1800, 1800)
        self.rotation_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.rotation_slider.setTickInterval(450)
        self.rotation_slider.setToolTip(
            "Rotate the displayed pattern only; misorientation, strain and Miller indices stay unchanged"
        )
        self.rotation_slider.valueChanged.connect(
            lambda value: self.owner._on_display_rotation(value / 10)
        )
        orientation_layout.addWidget(self.rotation_slider)
        layout.addWidget(orientation_box)

        self.view_section = CollapsibleSection(
            "VIEW", QtWidgets.QGridLayout, expanded=False
        )
        display_box = self.view_section
        display_layout = display_box.content_layout
        display_layout.addWidget(QtWidgets.QLabel("Field size"), 0, 0)
        self.view_scale_label = QtWidgets.QLabel(
            f"{self.owner.state.parameters.view_scale:.2f}×"
        )
        self.view_scale_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        display_layout.addWidget(self.view_scale_label, 0, 1)
        self.view_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.view_slider.setRange(
            round(VIEW_SCALE_MIN * 100.0), round(VIEW_SCALE_MAX * 100.0)
        )
        self.view_slider.setSingleStep(10)
        self.view_slider.setPageStep(50)
        self.view_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.view_slider.setTickInterval(50)
        self.view_slider.setValue(round(self.owner.state.parameters.view_scale * 100.0))
        self.view_slider.valueChanged.connect(self.owner._on_field_size_changed)
        display_layout.addWidget(self.view_slider, 1, 0, 1, 2)

        scale_stops_layout = QtWidgets.QGridLayout()
        scale_stops_layout.setSpacing(3)
        self.scale_stop_buttons: list[QtWidgets.QToolButton] = []
        for index, scale in enumerate(VIEW_SCALE_STOPS):
            button = QtWidgets.QToolButton()
            button.setText(f"{scale:g}×")
            button.setToolTip(f"Set field size to {scale:g}×")
            button.clicked.connect(
                lambda _checked=False, value=scale: self.view_slider.setValue(
                    round(value * 100.0)
                )
            )
            scale_stops_layout.addWidget(button, index // 3, index % 3)
            self.scale_stop_buttons.append(button)
        display_layout.addLayout(scale_stops_layout, 2, 0, 1, 2)

        self.axial_image_label = QtWidgets.QLabel("P2 axial image")
        display_layout.addWidget(self.axial_image_label, 3, 0)
        self.axial_spin = QtWidgets.QSpinBox()
        self.axial_spin.setRange(-4, 4)
        self.axial_spin.setPrefix("+")
        self.axial_spin.setValue(0)
        self.axial_spin.valueChanged.connect(self.owner._on_axial_repeat_changed)
        display_layout.addWidget(self.axial_spin, 3, 1)
        self.worker_label = QtWidgets.QLabel("CPU workers")
        display_layout.addWidget(self.worker_label, 4, 0)
        self.worker_spin = QtWidgets.QSpinBox()
        self.worker_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.worker_spin.setValue(self.owner.compute.worker_count)
        self.worker_spin.setToolTip(
            "Processes used for buffered grain generation and batched same-layer CSL searches"
        )
        self.worker_spin.valueChanged.connect(self.owner._on_worker_count_changed)
        display_layout.addWidget(self.worker_spin, 4, 1)
        self.cell_check = QtWidgets.QCheckBox("Show automatic cell")
        self.cell_check.setChecked(True)
        self.cell_check.setToolTip(
            "Layer-preserving CSL / Near-CSL periodic cell, not a GB structural unit"
        )
        self.cell_check.toggled.connect(self.owner.plot._update_common_cell)
        self.cell_check.toggled.connect(self.owner.plot._rebuild_legend)
        display_layout.addWidget(self.cell_check, 5, 0)
        self.cell_fit_button = QtWidgets.QPushButton("Fit cell")
        self.cell_fit_button.setToolTip("Fit the exact or strained common cell in view")
        self.cell_fit_button.clicked.connect(self.owner._fit_near_cell)
        display_layout.addWidget(self.cell_fit_button, 5, 1)
        layout.addWidget(display_box)

        # Keep crystallographic controls together at the top. Layer visibility,
        # GB-side clipping, vector-only options, and view scaling each have a
        # separate scope and should not share a generic display panel.
        layout.removeWidget(crystal_box)
        orientation_layout.insertWidget(0, crystal_box)

        crystal_layout.removeWidget(self.layer_info)
        crystal_layout.removeWidget(self.axial_layer_label)
        crystal_layout.removeWidget(self.layer_combo)
        self.axial_layer_label.hide()
        self.layer_combo.hide()
        visibility_layout.addWidget(self.layer_info, 0, 0, 1, 2)
        layer_buttons = QtWidgets.QHBoxLayout()
        self.all_layers_button = QtWidgets.QPushButton("All layers")
        self.no_layers_button = QtWidgets.QPushButton("No layers")
        self.all_layers_button.clicked.connect(
            lambda: self.owner._set_all_layers_visible(True)
        )
        self.no_layers_button.clicked.connect(
            lambda: self.owner._set_all_layers_visible(False)
        )
        layer_buttons.addWidget(self.all_layers_button)
        layer_buttons.addWidget(self.no_layers_button)
        visibility_layout.addLayout(layer_buttons, 1, 0, 1, 2)
        self.axial_layer_checks_widget = QtWidgets.QWidget()
        self.axial_layer_checks_layout = QtWidgets.QGridLayout(
            self.axial_layer_checks_widget
        )
        self.axial_layer_checks_layout.setContentsMargins(0, 0, 0, 0)
        self.axial_layer_scroll = QtWidgets.QScrollArea()
        self.axial_layer_scroll.setWidgetResizable(True)
        self.axial_layer_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.axial_layer_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.axial_layer_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.axial_layer_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.axial_layer_scroll.setWidget(self.axial_layer_checks_widget)
        visibility_layout.addWidget(self.axial_layer_scroll, 2, 0, 1, 2)
        self.grain_layer_checks: list[list[QtWidgets.QCheckBox]] = [[], []]
        # Compatibility toggles preserve the former one-checkbox-per-layer
        # programmatic interface. The visible controls below are per grain.
        self.layer_checks: list[QtWidgets.QCheckBox] = []
        self._rebuild_layer_checks()

        self.gb_region_widget = QtWidgets.QWidget()
        gb_region_layout = QtWidgets.QVBoxLayout(self.gb_region_widget)
        gb_region_layout.setContentsMargins(0, 7, 0, 0)
        gb_region_layout.setSpacing(6)
        gb_region_layout.addWidget(QtWidgets.QLabel("GB side visibility"))
        gb_checks_layout = QtWidgets.QGridLayout()
        for index, check in enumerate(self.region_checks):
            visibility_layout.removeWidget(check)
            gb_checks_layout.addWidget(check, index // 2, index % 2)
        gb_region_layout.addLayout(gb_checks_layout)
        gb_presets_layout = QtWidgets.QGridLayout()
        for button in (
            self.full_button,
            self.first_bicrystal_button,
            self.second_bicrystal_button,
        ):
            interaction_layout.removeWidget(button)
        self.full_button.setText("Show all sides   F")
        gb_presets_layout.addWidget(self.full_button, 0, 0, 1, 2)
        gb_presets_layout.addWidget(self.first_bicrystal_button, 1, 0)
        gb_presets_layout.addWidget(self.second_bicrystal_button, 1, 1)
        gb_region_layout.addLayout(gb_presets_layout)
        interaction_layout.addWidget(self.gb_region_widget, 1, 0, 1, 2)

        self.vector_options_widget = QtWidgets.QWidget()
        vector_options_layout = QtWidgets.QFormLayout(self.vector_options_widget)
        vector_options_layout.setContentsMargins(0, 7, 0, 0)
        display_layout.removeWidget(self.axial_image_label)
        display_layout.removeWidget(self.axial_spin)
        self.axial_image_label.setText("P2 axial periodic image")
        axial_help = (
            "Vector readout only: choose which periodic atom along the viewing axis "
            "the projected P2 column represents. This does not add thickness, duplicate "
            "the structure, or change plotted atoms."
        )
        self.axial_image_label.setToolTip(axial_help)
        self.axial_spin.setToolTip(axial_help)
        vector_options_layout.addRow(self.axial_image_label, self.axial_spin)
        interaction_layout.addWidget(self.vector_options_widget, 2, 0, 1, 2)

        interaction_layout.removeWidget(self.center_button)
        display_layout.addWidget(self.center_button, 3, 0, 1, 2)
        display_layout.removeWidget(self.cell_check)
        display_layout.removeWidget(self.cell_fit_button)
        self.cell_check.setText("Automatic common cell")
        self.cell_check.setChecked(False)
        self.owner.plot._rebuild_legend()
        visibility_layout.addWidget(self.cell_check, 3, 0)
        visibility_layout.addWidget(self.cell_fit_button, 3, 1)

        display_layout.removeWidget(self.worker_label)
        display_layout.removeWidget(self.worker_spin)
        self.performance_section = CollapsibleSection(
            "PERFORMANCE", QtWidgets.QFormLayout, expanded=False
        )
        self.performance_section.content_layout.addRow(
            self.worker_label, self.worker_spin
        )
        layout.addWidget(self.performance_section)

        # Orientation and layer visibility share one tabbed card because they
        # are normally adjusted at different times. Orientation is active by
        # default, so the layer page starts folded behind its neighboring tab.
        for section in (
            orientation_box,
            visibility_box,
            interaction_box,
            display_box,
            self.performance_section,
        ):
            layout.removeWidget(section)
        self.orientation_layers_tabs = CurrentPageTabWidget()
        self.orientation_layers_tabs.setObjectName("controlTabs")
        self.orientation_layers_tabs.addTab(orientation_box.body, "ORIENTATION")
        self.orientation_layers_tabs.addTab(visibility_box.body, "LAYERS")
        self.orientation_layers_tabs.setCurrentIndex(0)
        self.orientation_layers_tabs.syncCurrentPageGeometry()
        layout.insertWidget(1, self.orientation_layers_tabs)
        layout.insertWidget(2, interaction_box)

        self.view_performance_section = CollapsibleSection(
            "VIEW / PERFORMANCE", QtWidgets.QVBoxLayout, expanded=False
        )
        self.view_performance_tabs = CurrentPageTabWidget()
        self.view_performance_tabs.setObjectName("controlTabs")
        self.view_performance_tabs.addTab(display_box.body, "VIEW")
        self.view_performance_tabs.addTab(self.performance_section.body, "PERFORMANCE")
        self.view_performance_tabs.syncCurrentPageGeometry()
        self.view_performance_section.content_layout.addWidget(
            self.view_performance_tabs
        )
        layout.insertWidget(3, self.view_performance_section)
        orientation_box.deleteLater()
        visibility_box.deleteLater()
        display_box.deleteLater()
        self.performance_section.deleteLater()
        self.owner._sync_context_controls()

        self.manual_section = CollapsibleSection(
            "MANUAL COMMON CELL", QtWidgets.QVBoxLayout, expanded=False
        )
        manual_box = self.manual_section
        manual_layout = manual_box.content_layout
        self.manual_pick_button = QtWidgets.QPushButton("Pick 4 CSL vertices   M")
        self.manual_pick_button.setCheckable(True)
        self.manual_pick_button.clicked.connect(self.owner._start_manual_cell)
        manual_layout.addWidget(self.manual_pick_button)
        manual_help = QtWidgets.QLabel(
            "Pick 4 same-layer/symbol sites around the perimeter. Blue/red cells use their own atom vertices; count the picked layer only."
        )
        manual_help.setWordWrap(True)
        manual_help.setObjectName("mutedLabel")
        manual_layout.addWidget(manual_help)
        manual_actions = QtWidgets.QHBoxLayout()
        self.manual_undo_button = QtWidgets.QPushButton("Undo vertex")
        self.manual_undo_button.clicked.connect(self.owner._undo_manual_vertex)
        self.manual_clear_button = QtWidgets.QPushButton("Clear")
        self.manual_clear_button.clicked.connect(self.owner._clear_manual_cell)
        self.manual_fit_button = QtWidgets.QPushButton("Fit")
        self.manual_fit_button.setEnabled(False)
        self.manual_fit_button.clicked.connect(self.owner._fit_manual_cell)
        for button in (
            self.manual_undo_button,
            self.manual_clear_button,
            self.manual_fit_button,
        ):
            manual_actions.addWidget(button)
        manual_layout.addLayout(manual_actions)
        self.manual_visible_check = QtWidgets.QCheckBox(
            "Apply GB side visibility to counts"
        )
        self.manual_visible_check.setToolTip(
            "Counts always use the vertices' layer. Optionally apply G1/G2 side switches; the Layers display checkboxes never change the counted layer."
        )
        self.manual_visible_check.toggled.connect(self.owner._queue_manual_count)
        manual_layout.addWidget(self.manual_visible_check)
        self.manual_info = QtWidgets.QPlainTextEdit()
        self.manual_info.setReadOnly(True)
        self.manual_info.setMinimumHeight(125)
        self.manual_info.setMaximumHeight(190)
        self.manual_info.setPlainText(
            "No manual cell. Pick one layer; each grain is counted inside its own four actual atom vertices. Periodicity is not assumed."
        )
        manual_layout.addWidget(self.manual_info)
        self.manual_strain_button = QtWidgets.QPushButton(
            "Apply bulk strain to selected cell"
        )
        self.manual_strain_button.setEnabled(False)
        self.manual_strain_button.clicked.connect(self.owner._toggle_manual_strain)
        self.manual_strain_button.setToolTip(
            "Available after picking four same-layer vertices in Local matching, including at least one near pair. "
            "Align all four pairs by bulk strain and bounded small rotations, then recompute exact CSL."
        )
        manual_layout.addWidget(self.manual_strain_button)
        manual_strain_form = QtWidgets.QFormLayout()
        self.manual_strain_limit = QtWidgets.QDoubleSpinBox()
        self.manual_strain_limit.setRange(0.01, 10.0)
        self.manual_strain_limit.setDecimals(3)
        self.manual_strain_limit.setValue(DEFAULT_STRAIN_PERCENT)
        self.manual_strain_limit.setSuffix(" %")
        self.manual_strain_limit.setSingleStep(0.1)
        self.manual_strain_limit.setEnabled(False)
        manual_strain_form.addRow(
            "Selected-cell strain limit", self.manual_strain_limit
        )
        self.manual_rotation_limit = QtWidgets.QDoubleSpinBox()
        self.manual_rotation_limit.setRange(0, 5.0)
        self.manual_rotation_limit.setDecimals(3)
        self.manual_rotation_limit.setValue(1.0)
        self.manual_rotation_limit.setSuffix("°")
        self.manual_rotation_limit.setSingleStep(0.1)
        self.manual_rotation_limit.setEnabled(False)
        self.manual_rotation_limit.setToolTip(
            "Maximum polar rigid rotation per grain, separate from strain. 0° restricts the fit to pure symmetric strain."
        )
        manual_strain_form.addRow("Rotation limit / grain", self.manual_rotation_limit)
        manual_layout.addLayout(manual_strain_form)
        self.manual_strain_note = QtWidgets.QLabel(
            "Select a local near-CSL cell first. Both grains share the strain; no individual atoms are snapped."
        )
        self.manual_strain_note.setWordWrap(True)
        self.manual_strain_note.setObjectName("mutedLabel")
        manual_layout.addWidget(self.manual_strain_note)
        self.manual_strain_details = QtWidgets.QPlainTextEdit()
        self.manual_strain_details.setObjectName("manualStrainDetails")
        self.manual_strain_details.setAccessibleName(
            "Selected-cell strain and rotation details"
        )
        self.manual_strain_details.setReadOnly(True)
        self.manual_strain_details.setMinimumHeight(220)
        self.manual_strain_details.setMaximumHeight(320)
        self.manual_strain_details.hide()
        manual_layout.addWidget(self.manual_strain_details)
        layout.addWidget(manual_box)

        self.near_section = CollapsibleSection(
            "NEAR-CSL", QtWidgets.QVBoxLayout, expanded=False
        )
        near_box = self.near_section
        near_layout = near_box.content_layout
        near_layout.addWidget(QtWidgets.QLabel("Method"))
        self.near_method_combo = QtWidgets.QComboBox()
        self.near_method_combo.setToolTip("Choose a Near-CSL method")
        self.near_method_combo.addItem("Local matching · no bulk strain", "local")
        self.near_method_combo.addItem("Homogeneous strain + periodic cell", "strain")
        self.near_method_combo.currentIndexChanged.connect(
            self.owner._on_near_method_changed
        )
        near_layout.addWidget(self.near_method_combo)
        self.near_button = QtWidgets.QPushButton("Enable Near-CSL")
        self.near_button.setCheckable(True)
        self.near_button.toggled.connect(self.owner._toggle_near)
        near_layout.addWidget(self.near_button)
        near_form = QtWidgets.QFormLayout()
        self.near_form = near_form
        self.local_distance_spin = QtWidgets.QDoubleSpinBox()
        self.local_distance_spin.setDecimals(4)
        self.local_distance_spin.setRange(0.0001, 0.5)
        self.local_distance_spin.setSingleStep(0.01)
        self.local_distance_spin.setValue(DEFAULT_LOCAL_DISTANCE)
        self.local_distance_spin.setSuffix(" a₀")
        self.local_distance_spin.setToolTip(
            "Maximum pair separation d, not strain. A midpoint alignment would move each atom by d/2."
        )
        self.local_distance_spin.setEnabled(False)
        self.local_distance_spin.valueChanged.connect(self.owner._queue_near_search)
        near_form.addRow("Local pair distance", self.local_distance_spin)
        self.strain_spin = QtWidgets.QDoubleSpinBox()
        self.strain_spin.setRange(0.01, 10.0)
        self.strain_spin.setValue(DEFAULT_STRAIN_PERCENT)
        self.strain_spin.setSuffix(" %")
        self.strain_spin.setSingleStep(0.1)
        self.strain_spin.setToolTip("Limit on |principal stretch - 1| in each grain")
        self.search_index_spin = QtWidgets.QSpinBox()
        self.search_index_spin.setRange(2, 40)
        self.search_index_spin.setValue(DEFAULT_SEARCH_INDEX)
        near_form.addRow("Max principal strain", self.strain_spin)
        near_form.addRow("Search index bound", self.search_index_spin)
        near_layout.addLayout(near_form)
        for control in (self.strain_spin, self.search_index_spin):
            control.setEnabled(False)
            control.valueChanged.connect(self.owner._queue_near_search)
        self.near_combo = QtWidgets.QComboBox()
        self.near_combo.setEnabled(False)
        self.near_combo.currentIndexChanged.connect(self.owner._select_near_cell)
        near_layout.addWidget(self.near_combo)
        self.near_info = QtWidgets.QPlainTextEdit()
        self.near_info.setReadOnly(True)
        self.near_info.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        self.near_info.setMinimumHeight(175)
        self.near_info.setPlainText(
            "Off. Choose a method and enable Near-CSL.\n"
            "Local: same-layer mutual nearest pairs; no atoms moved.\n"
            "Strain: solve homogeneous strains and a common periodic cell."
        )
        near_layout.addWidget(self.near_info)
        layout.addWidget(near_box)

        # Near-CSL precedes the manual-cell workflow; both start collapsed.
        layout.removeWidget(near_box)
        layout.removeWidget(manual_box)
        layout.insertWidget(4, near_box)
        layout.insertWidget(5, manual_box)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("statusCard")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(112)
        layout.addWidget(self.status_label)

        export_button = QtWidgets.QPushButton("Export plot as PNG…")
        export_button.clicked.connect(self.owner._choose_export_path)
        layout.addWidget(export_button)
        layout.addStretch(1)
        self.owner._sync_near_controls()
        self.owner._sync_angle_controls()

    def _populate_layer_combo(self):
        with QtCore.QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            self.layer_combo.addItem("All layers", -1)
            for layer in range(self.owner.state.geometry.layer_count):
                self.layer_combo.addItem(
                    f"{layer_name(layer)} · z/a₀ = {layer*self.owner.state.geometry.layer_spacing:.6g}",
                    layer,
                )
            self.layer_combo.setCurrentIndex(self.owner.state.selected_layer + 1)

    def _rebuild_layer_checks(self) -> None:
        if not hasattr(self, "axial_layer_checks_layout"):
            return
        for check in getattr(self, "layer_checks", []):
            check.deleteLater()
        while self.axial_layer_checks_layout.count():
            item = self.axial_layer_checks_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.grain_layer_checks = [[], []]
        self.layer_checks = []
        for layer in range(self.owner.state.geometry.layer_count):
            for grain in (0, 1):
                check = QtWidgets.QCheckBox(f"G{grain + 1} {layer_name(layer)}")
                check.setIcon(layer_marker_icon(grain, layer))
                check.setIconSize(QtCore.QSize(18, 18))
                check.setToolTip(
                    f"Show G{grain + 1} layer {layer_name(layer)} atoms · "
                    f"z/a₀ = {layer*self.owner.state.geometry.layer_spacing:.6g}"
                )
                check.setChecked(layer in self.owner.state.visible_grain_layers[grain])
                check.toggled.connect(
                    lambda checked, grain_index=grain, layer_index=layer: (
                        self.owner._on_grain_layer_toggled(
                            grain_index, layer_index, checked
                        )
                    )
                )
                self.axial_layer_checks_layout.addWidget(check, layer, grain)
                self.grain_layer_checks[grain].append(check)

            compatibility_check = QtWidgets.QCheckBox(self.owner)
            compatibility_check.hide()
            compatibility_check.setChecked(layer in self.owner.state.visible_layers)
            compatibility_check.toggled.connect(
                lambda checked, index=layer: self.owner._on_axial_layer_toggled(
                    index, checked
                )
            )
            self.layer_checks.append(compatibility_check)

        self.axial_layer_checks_layout.setColumnStretch(0, 1)
        self.axial_layer_checks_layout.setColumnStretch(1, 1)
        self.axial_layer_checks_layout.activate()
        content_height = self.axial_layer_checks_layout.sizeHint().height()
        scroll_height = min(150, max(34, content_height + 2))
        self.axial_layer_scroll.setFixedHeight(scroll_height)
        if hasattr(self, "orientation_layers_tabs"):
            self.orientation_layers_tabs.updateGeometry()


def create_application() -> QtWidgets.QApplication:
    pg.setConfigOptions(antialias=True, foreground="#334155")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Tilt GB Dichromatic Pattern")
    application.setStyle("Fusion")
    assets = Path(__file__).with_name("resources") / "icons"
    stylesheet = APPLICATION_STYLESHEET.replace(
        "__CHEVRON_DOWN__", (assets / "chevron-down.svg").as_posix()
    ).replace("__CHEVRON_UP__", (assets / "chevron-up.svg").as_posix())
    application.setStyleSheet(stylesheet)
    return application
