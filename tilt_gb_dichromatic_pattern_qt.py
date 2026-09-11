#!/usr/bin/env python3
"""Standalone cubic FCC/BCC tilt-GB viewer, with coordinates in units of a0.

This Qt/PyQtGraph application is independent of GBClaw. The older Matplotlib
viewer is deprecated; this is the actively maintained interface.

Controls
--------
* Drag anywhere in the plot to pan; use the wheel to zoom.
* Press "Pick GB" and click B1/B2 to define a trial boundary.
* Press "Measure vector" and click P1/P2 to label their lattice vector.
* Atom selections survive panning and zooming.
* Display rotation turns the drawing, without changing misorientation or strain.
* Pick four same-layer common sites around a manual cell and count its atoms.
* Choose FCC/BCC and a preset or custom integer tilt axis; axial layers are computed.
* Every plot coordinate is in a0: coordinate 1 means one lattice constant.
* Gold rings mark coincidences only when both position and layer agree.
* Near-CSL offers unstrained local pair matching or homogeneous strain/cell search.
* Show common cell / Fit cell also work for unstrained, exact CSL rotations.

Only NumPy, PySide6, and PyQtGraph are required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import multiprocessing
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6 import QtCore, QtGui, QtWidgets
from tilt_gb_near_csl import (
    NearSearch,
    DEFAULT_STRAIN_PERCENT,
    DEFAULT_SEARCH_INDEX,
    NEAR_COLOR,
    tensor_readout,
    worker_initializer,
    exact_csl_cell,
    LocalPairs,
    local_match_layers_worker,
    DEFAULT_LOCAL_DISTANCE,
    LOCAL_COLOR,
    strain_selected_cell,
)
from tilt_gb_crystallography import (
    ProjectedGrain,
    CSLPreset,
    get_geometry,
    projected_columns,
    csl_angle_deg,
    csl_presets,
    matching_csl_preset,
    layer_name,
    GeometryLimitError,
    validate_cell_vertices,
    count_cell_atoms,
)


SIDE_TOLERANCE = 1.0e-9
COINCIDENCE_TOLERANCE_FACTOR = 1.0e-6
BUFFER_FACTOR = 2.0
VIEW_SCALE_MIN = 0.1
VIEW_SCALE_MAX = 5.0
VIEW_SCALE_STOPS = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0)

GRAIN_1_COLOR = "#1677d2"
GRAIN_1_EDGE = "#075796"
GRAIN_2_COLOR = "#e35d35"
COINCIDENCE_COLOR = "#e5a50a"
BOUNDARY_COLOR = "#17212b"
VECTOR_COLOR = "#8736a6"
LAYER_SYMBOLS = ("o", "d", "t", "s", "p", "h", "star", "+", "x", "t1", "t2", "t3")
MANUAL_CELL_COLOR = "#4755b8"


# Backward-compatible names for scripts importing the original FCC [110] API.
CSL_PRESETS = csl_presets("110")
DEFAULT_ANGLE_DEG = CSL_PRESETS[3].angle_deg


def default_angle_deg(axis: str) -> float:
    presets = csl_presets(axis)
    sigma = {"110": 9, "100": 5}.get(axis)
    if sigma is not None:
        return min(p.angle_deg for p in presets if p.sigma == sigma)
    return (
        min(presets, key=lambda p: (p.sigma, p.angle_deg)).angle_deg if presets else 0.0
    )


@dataclass(frozen=True)
class SelectedAtom:
    position: np.ndarray
    grain_index: int
    layer: int
    half_indices: np.ndarray


@dataclass(frozen=True)
class CellVertex:
    position: np.ndarray  # unrotated model coordinates
    layer: int
    source: str  # exact CSL or local near-pair midpoint
    grain_positions: np.ndarray | None = None  # (G1 blue, G2 red) actual endpoints


def cell_count_worker(*args):
    return os.getpid(), count_cell_atoms(*args)


@dataclass(frozen=True)
class PatternParameters:
    angle_deg: float | None = None
    lattice_constant: float = 3.52
    width: float = 12.0  # units of a0, not Angstrom
    height: float = 9.0
    marker_size: float = 32.0
    view_scale: float = 1.0
    lattice: str = "FCC"
    axis: str = "110"

    def validate(self) -> None:
        get_geometry(self.lattice, self.axis)
        if self.angle_deg is not None and not 0.0 <= self.angle_deg <= 90.0:
            raise ValueError("angle_deg must be between 0 and 90 degrees")
        if self.lattice_constant <= 0.0:
            raise ValueError("lattice_constant must be positive")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("width and height must be positive")
        if self.marker_size <= 0.0:
            raise ValueError("marker_size must be positive")
        if not VIEW_SCALE_MIN <= self.view_scale <= VIEW_SCALE_MAX:
            raise ValueError(
                f"view_scale must be between {VIEW_SCALE_MIN} and {VIEW_SCALE_MAX}"
            )


def rotation_matrix_2d(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def fcc_110_projected_columns(
    lattice_constant: float,
    width: float,
    height: float,
    rotation_deg: float,
    center: tuple[float, float] = (0.0, 0.0),
    deformation: np.ndarray | None = None,
) -> ProjectedGrain:
    """Legacy physical-unit adapter; the Qt viewer uses projected_columns in a0."""
    grain = projected_columns(
        width / lattice_constant,
        height / lattice_constant,
        rotation_deg,
        center=tuple(np.asarray(center) / lattice_constant),
        deformation=deformation,
        lattice="FCC",
        axis="110",
    )
    return ProjectedGrain(
        grain.positions * lattice_constant, grain.layers, grain.half_indices
    )


def selected_region_mask(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    keep_left: bool,
    keep_right: bool,
) -> np.ndarray:
    direction = second - first
    signed_cross_product = direction[0] * (points[:, 1] - first[1]) - direction[1] * (
        points[:, 0] - first[0]
    )
    keep = np.zeros(len(points), dtype=bool)
    if keep_left:
        keep |= signed_cross_product >= -SIDE_TOLERANCE
    if keep_right:
        keep |= signed_cross_product <= SIDE_TOLERANCE
    return keep


def same_layer_coincidence_sites(
    grain_1: ProjectedGrain,
    grain_2: ProjectedGrain,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    """Find coincidences separately in every computed axial phase."""

    if tolerance <= 0.0:
        raise ValueError("coincidence tolerance must be positive")
    sites_by_layer: list[np.ndarray] = []
    tolerance_squared = tolerance * tolerance

    def packed_cell_keys(cells: np.ndarray) -> np.ndarray:
        """Pack two signed 32-bit cell coordinates into one sortable key."""

        mask = np.uint64(0xFFFFFFFF)
        x_bits = cells[:, 0].astype(np.uint64) & mask
        y_bits = cells[:, 1].astype(np.uint64) & mask
        return (x_bits << np.uint64(32)) | y_bits

    count = max(
        getattr(grain_1, "layer_count", 2),
        getattr(grain_2, "layer_count", 2),
        int(np.max(grain_1.layers, initial=-1)) + 1,
        int(np.max(grain_2.layers, initial=-1)) + 1,
    )
    for layer in range(count):
        first = grain_1.positions[grain_1.layers == layer]
        second = grain_2.positions[grain_2.layers == layer]
        if len(first) == 0 or len(second) == 0:
            sites_by_layer.append(np.empty((0, 2)))
            continue

        second_cells = np.floor(second / tolerance).astype(np.int64)
        second_keys = packed_cell_keys(second_cells)
        order = np.argsort(second_keys)
        sorted_keys = second_keys[order]
        first_cells = np.floor(first / tolerance).astype(np.int64)
        matched_second = np.full(len(first), -1, dtype=np.int64)

        # A genuine match can lie in the same cell or one of eight neighbors.
        # Searching all first sites in a batch keeps large CSL views responsive.
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                unresolved = matched_second < 0
                if not np.any(unresolved):
                    break
                first_indices = np.flatnonzero(unresolved)
                neighbor_cells = first_cells[first_indices] + (delta_x, delta_y)
                neighbor_keys = packed_cell_keys(neighbor_cells)
                locations = np.searchsorted(sorted_keys, neighbor_keys)
                within = locations < len(sorted_keys)
                safe_locations = np.minimum(locations, len(sorted_keys) - 1)
                key_matches = within & (sorted_keys[safe_locations] == neighbor_keys)
                if not np.any(key_matches):
                    continue
                trial_first = first_indices[key_matches]
                trial_second = order[safe_locations[key_matches]]
                distances_squared = np.sum(
                    (first[trial_first] - second[trial_second]) ** 2, axis=1
                )
                accepted = distances_squared <= tolerance_squared
                matched_second[trial_first[accepted]] = trial_second[accepted]
            if not np.any(matched_second < 0):
                break

        matched_first = np.flatnonzero(matched_second >= 0)
        sites_by_layer.append(
            0.5 * (first[matched_first] + second[matched_second[matched_first]])
        )
    return tuple(sites_by_layer)


def generate_grain_worker(
    width: float,
    height: float,
    rotation_deg: float,
    center: tuple[float, float],
    deformation: np.ndarray | None = None,
    lattice: str = "FCC",
    axis: str = "110",
    translation: np.ndarray | None = None,
) -> tuple[int, ProjectedGrain]:
    """Process-pool entry point for one projected grain."""

    return (
        os.getpid(),
        projected_columns(
            width, height, rotation_deg, center, deformation, lattice, axis, translation
        ),
    )


def coincidence_layer_worker(
    grain_1: ProjectedGrain,
    grain_2: ProjectedGrain,
    tolerance: float,
    layer: int,
) -> tuple[int, np.ndarray]:
    """Process-pool entry point for one stacking-layer coincidence search."""

    only_first = grain_1.layers == layer
    only_second = grain_2.layers == layer
    first = ProjectedGrain(
        grain_1.positions[only_first],
        np.zeros(np.count_nonzero(only_first), dtype=np.int16),
        grain_1.half_indices[only_first],
    )
    second = ProjectedGrain(
        grain_2.positions[only_second],
        np.zeros(np.count_nonzero(only_second), dtype=np.int16),
        grain_2.half_indices[only_second],
    )
    return os.getpid(), same_layer_coincidence_sites(first, second, tolerance)[0]


def coincidence_layers_worker(grain_1, grain_2, tolerance, layers):
    """A bounded batch of phases, avoiding one full-array transfer per layer."""
    return os.getpid(), tuple(
        (
            int(layer),
            coincidence_layer_worker(grain_1, grain_2, tolerance, int(layer))[1],
        )
        for layer in layers
    )


class PatternViewBox(pg.ViewBox):
    """Pan-mode ViewBox used by the pattern canvas."""

    def __init__(self) -> None:
        super().__init__(lockAspect=True, enableMouse=True, enableMenu=False)
        self.setMouseMode(self.PanMode)


class DichromaticPatternWindow(QtWidgets.QMainWindow):
    """Native Qt window for cubic dichromatic patterns in normalized a0 units."""

    def __init__(
        self, parameters: PatternParameters, worker_count: int | None = None
    ) -> None:
        super().__init__()
        parameters.validate()
        self.parameters = parameters
        self.geometry = get_geometry(parameters.lattice, parameters.axis)
        self.presets = csl_presets(self.geometry.axis)
        initial_angle = parameters.angle_deg
        if initial_angle is None:
            initial_angle = default_angle_deg(self.geometry.axis)
        initial_preset = matching_csl_preset(initial_angle, axis=self.geometry.axis)
        self.angle_deg = (
            initial_preset.angle_deg if initial_preset is not None else initial_angle
        )
        self.interaction_mode = "idle"
        self.display_rotation_deg = 0.0
        self.manual_vertices = []
        self.manual_counts = None
        self.manual_count_error = None
        self.manual_count_key = None
        self.manual_count_future = None
        self.manual_count_pending = None
        self.manual_count_running_key = None
        self.manual_local_cutoff = None
        self.manual_strain_fit = None
        self.manual_unstrained_vertices = None
        self.manual_unstrained_cutoff = None
        self.axial_repeat = 0
        self.grains: list[ProjectedGrain] = []
        self.grain_signature = None
        self.visible_atom_masks: list[np.ndarray] = []
        self.coincident_points = self._empty_coincidences()
        self.visible_atom_counts = [0, 0]
        self.visible_coincidence_counts = [0] * self.geometry.layer_count
        self.selected_layer = -1
        self.selected_points: list[np.ndarray] = []
        self.selected_atoms: list[SelectedAtom] = []
        self.buffer_bounds: tuple[float, float, float, float] | None = None
        self.pending_angle = self.angle_deg
        self.angle_update_active = False
        self.csl_updating = False
        self.render_error = None
        self.near_enabled = False
        self.near_method = "local"
        self.local_pairs = LocalPairs.empty()
        self.local_updating = False
        self.local_error = None
        self.local_thread_executor = None
        self.near_search = None
        self.near_solutions = []
        self.near_cell = None
        self.deformations = (np.eye(2), np.eye(2))
        self.translations = np.zeros((2, 2))
        available_cpus = max(1, os.cpu_count() or 1)
        automatic_workers = min(4, available_cpus)
        self.worker_count = int(worker_count or automatic_workers)
        self.worker_count = int(np.clip(self.worker_count, 1, available_cpus))
        self.executor: concurrent.futures.ProcessPoolExecutor | None = None
        self.parallel_stage: str | None = None
        self.parallel_futures: list[concurrent.futures.Future] = []
        self.parallel_payload: dict[str, object] = {}
        self.parallel_generation = 0
        self.worker_process_ids: set[int] = set()
        self._configure_worker_pool(self.worker_count)

        self.setWindowTitle("Cubic Tilt GB · Dichromatic Pattern")
        self.resize(1380, 860)
        self.setMinimumSize(1040, 680)

        self._create_plot()
        self._create_control_dock()
        self._create_timers()
        self._create_shortcuts()
        self._update_geometry_labels()
        self._set_initial_view()
        self._regenerate_buffer(compute_coincidences=True)
        self.view_refresh_timer.stop()
        self._set_mode("idle")

    # ---------- construction ----------

    def _create_plot(self) -> None:
        self.view_box = PatternViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box)
        self.plot_item = self.plot_widget.getPlotItem()
        self.setCentralWidget(self.plot_widget)
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
            minXRange=VIEW_SCALE_MIN * self.parameters.width,
            maxXRange=VIEW_SCALE_MAX * self.parameters.width,
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
            pen=pg.mkPen(GRAIN_1_COLOR, width=2.4, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.manual_grain_cell_items = [
            self.manual_cell_item,
            pg.PlotCurveItem(
                pen=pg.mkPen(GRAIN_2_COLOR, width=2.4, style=QtCore.Qt.PenStyle.DotLine)
            ),
        ]
        self.manual_vertex_item = self._ring_item(MANUAL_CELL_COLOR, diameter * 2.2)
        self.manual_labels = [
            self._text_item(f"C{i+1}", MANUAL_CELL_COLOR) for i in range(4)
        ]
        self.manual_annotation = self._text_item(
            "", MANUAL_CELL_COLOR, font_size=10, bordered=True
        )
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

        self.vector_arrow = pg.ArrowItem(
            angle=0,
            headLen=15,
            tipAngle=28,
            brush=pg.mkBrush(VECTOR_COLOR),
            pen=pg.mkPen(VECTOR_COLOR),
            pxMode=True,
        )
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
        for item in (
            *self.boundary_labels,
            *self.vector_labels,
            *self.side_labels,
            self.vector_annotation,
        ):
            item.hide()
            self.plot_item.addItem(item)

        self.legend = self.plot_item.addLegend(offset=(14, 14), colCount=3)
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 225))
        self.legend.setPen(pg.mkPen("#cbd5e1"))
        self._create_layer_items()

        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_mouse_click)
        self.view_box.sigRangeChanged.connect(self._on_view_range_changed)

    def _create_layer_items(self):
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
        for layer in range(self.geometry.layer_count):
            symbol = LAYER_SYMBOLS[layer % len(LAYER_SYMBOLS)]
            for grain in (0, 1):
                item = pg.ScatterPlotItem(
                    size=diameter * (1 if grain == 0 else 1.12),
                    symbol=symbol,
                    pen=pg.mkPen(
                        GRAIN_1_EDGE if grain == 0 else GRAIN_2_COLOR,
                        width=0.8 if grain == 0 else 1.5,
                    ),
                    brush=(
                        pg.mkBrush(22, 119, 210, 185)
                        if grain == 0
                        else pg.mkBrush(0, 0, 0, 0)
                    ),
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

    def _rebuild_legend(self):
        self.legend.clear()
        layers = (
            [self.selected_layer]
            if self.selected_layer >= 0
            else range(min(self.geometry.layer_count, 6))
        )
        for layer in layers:
            name = layer_name(layer)
            self.legend.addItem(self.grain_layer_items[0][layer], f"G1 · {name}")
            self.legend.addItem(self.grain_layer_items[1][layer], f"G2 · {name}")
            self.legend.addItem(self.coincidence_items[layer], f"CSL · {name}–{name}")
        self.legend.addItem(self.near_cell_item, "Common periodic cell")
        if self.local_active:
            self.legend.addItem(self.local_match_item, "Local near pair · midpoint")

    def _create_control_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setMinimumWidth(310)
        dock.setMaximumWidth(370)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        scroll.setWidget(panel)
        dock.setWidget(scroll)

        self.heading = QtWidgets.QLabel()
        layout.addWidget(self.heading)

        crystal_box = QtWidgets.QGroupBox("CRYSTAL")
        crystal_layout = QtWidgets.QFormLayout(crystal_box)
        self.structure_combo = QtWidgets.QComboBox()
        for lattice in ("FCC", "BCC"):
            self.structure_combo.addItem(lattice, lattice)
        self.structure_combo.setCurrentIndex(
            self.structure_combo.findData(self.geometry.lattice)
        )
        self.axis_combo = QtWidgets.QComboBox()
        for axis in ("100", "110", "111", "112"):
            self.axis_combo.addItem(f"⟨{axis}⟩", axis)
        self.axis_combo.addItem("Custom [h k l]", None)
        axis_index = self.axis_combo.findData(self.geometry.axis)
        self.axis_combo.setCurrentIndex(axis_index if axis_index >= 0 else 4)
        crystal_layout.addRow("Structure", self.structure_combo)
        crystal_layout.addRow("Tilt / viewing axis", self.axis_combo)
        self.custom_axis_row = QtWidgets.QWidget()
        custom_layout = QtWidgets.QHBoxLayout(self.custom_axis_row)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_axis_edit = QtWidgets.QLineEdit(
            " ".join(map(str, self.geometry.axis_indices))
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
        crystal_layout.addRow("Visible axial layers", self.layer_combo)
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        self.axis_combo.currentIndexChanged.connect(self._on_geometry_changed)
        self.structure_combo.currentIndexChanged.connect(
            lambda: self._on_geometry_changed(apply_custom=True)
        )
        self.custom_axis_apply.clicked.connect(
            lambda: self._on_geometry_changed(apply_custom=True)
        )
        self.custom_axis_edit.returnPressed.connect(
            lambda: self._on_geometry_changed(apply_custom=True)
        )
        layout.addWidget(crystal_box)

        interaction_box = QtWidgets.QGroupBox("INTERACTION")
        interaction_layout = QtWidgets.QGridLayout(interaction_box)
        self.pick_gb_button = QtWidgets.QPushButton("Pick GB   R")
        self.vector_button = QtWidgets.QPushButton("Measure vector   V")
        self.pick_gb_button.setCheckable(True)
        self.vector_button.setCheckable(True)
        self.pick_gb_button.clicked.connect(self._start_new_boundary)
        self.vector_button.clicked.connect(self._start_vector_measurement)
        interaction_layout.addWidget(self.pick_gb_button, 0, 0)
        interaction_layout.addWidget(self.vector_button, 0, 1)

        self.full_button = QtWidgets.QPushButton("Full pattern   F")
        self.center_button = QtWidgets.QPushButton("Center view   C")
        self.first_bicrystal_button = QtWidgets.QPushButton("G1:L  /  G2:R")
        self.second_bicrystal_button = QtWidgets.QPushButton("G1:R  /  G2:L")
        self.full_button.clicked.connect(
            lambda: self._set_region_states((True, True, True, True))
        )
        self.center_button.clicked.connect(self._reset_view)
        self.first_bicrystal_button.clicked.connect(
            lambda: self._set_region_states((True, False, False, True))
        )
        self.second_bicrystal_button.clicked.connect(
            lambda: self._set_region_states((False, True, True, False))
        )
        interaction_layout.addWidget(self.full_button, 1, 0)
        interaction_layout.addWidget(self.center_button, 1, 1)
        interaction_layout.addWidget(self.first_bicrystal_button, 2, 0)
        interaction_layout.addWidget(self.second_bicrystal_button, 2, 1)
        layout.addWidget(interaction_box)

        visibility_box = QtWidgets.QGroupBox("VISIBLE REGIONS")
        visibility_layout = QtWidgets.QGridLayout(visibility_box)
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
            check.stateChanged.connect(self._update_visible_points)
            visibility_layout.addWidget(check, index // 2, index % 2)
            self.region_checks.append(check)
        layout.addWidget(visibility_box)

        orientation_box = QtWidgets.QGroupBox("ORIENTATION")
        orientation_layout = QtWidgets.QVBoxLayout(orientation_box)
        orientation_layout.addWidget(QtWidgets.QLabel("CSL preset"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Custom angle", None)
        for index, preset in enumerate(self.presets):
            self.preset_combo.addItem(preset.label, index)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        orientation_layout.addWidget(self.preset_combo)

        angle_row = QtWidgets.QHBoxLayout()
        angle_row.addWidget(QtWidgets.QLabel("Misorientation"))
        angle_row.addStretch(1)
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(0.0, 90.0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setSingleStep(0.1)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setValue(self.angle_deg)
        self.angle_spin.valueChanged.connect(self._queue_angle_update)
        angle_row.addWidget(self.angle_spin)
        orientation_layout.addLayout(angle_row)

        self.angle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.angle_slider.setRange(0, 9000)
        self.angle_slider.setSingleStep(1)
        self.angle_slider.setPageStep(100)
        self.angle_slider.setValue(round(self.angle_deg * 100.0))
        self.angle_slider.valueChanged.connect(
            lambda value: self._queue_angle_update(value / 100.0)
        )
        self.angle_slider.sliderReleased.connect(self._finish_angle_update)
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
        self.rotation_spin.valueChanged.connect(self._on_display_rotation)
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
            lambda value: self._on_display_rotation(value / 10)
        )
        orientation_layout.addWidget(self.rotation_slider)
        layout.addWidget(orientation_box)

        display_box = QtWidgets.QGroupBox("DISPLAY")
        display_layout = QtWidgets.QGridLayout(display_box)
        display_layout.addWidget(QtWidgets.QLabel("Field size"), 0, 0)
        self.view_scale_label = QtWidgets.QLabel(f"{self.parameters.view_scale:.2f}×")
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
        self.view_slider.setValue(round(self.parameters.view_scale * 100.0))
        self.view_slider.valueChanged.connect(self._on_field_size_changed)
        display_layout.addWidget(self.view_slider, 1, 0, 1, 2)

        scale_stops_layout = QtWidgets.QHBoxLayout()
        scale_stops_layout.setSpacing(3)
        self.scale_stop_buttons: list[QtWidgets.QToolButton] = []
        for scale in VIEW_SCALE_STOPS:
            button = QtWidgets.QToolButton()
            button.setText(f"{scale:g}×")
            button.setToolTip(f"Set field size to {scale:g}×")
            button.clicked.connect(
                lambda _checked=False, value=scale: self.view_slider.setValue(
                    round(value * 100.0)
                )
            )
            scale_stops_layout.addWidget(button)
            self.scale_stop_buttons.append(button)
        display_layout.addLayout(scale_stops_layout, 2, 0, 1, 2)

        display_layout.addWidget(QtWidgets.QLabel("P2 axial image"), 3, 0)
        self.axial_spin = QtWidgets.QSpinBox()
        self.axial_spin.setRange(-4, 4)
        self.axial_spin.setPrefix("+")
        self.axial_spin.setValue(0)
        self.axial_spin.valueChanged.connect(self._on_axial_repeat_changed)
        display_layout.addWidget(self.axial_spin, 3, 1)
        display_layout.addWidget(QtWidgets.QLabel("CPU workers"), 4, 0)
        self.worker_spin = QtWidgets.QSpinBox()
        self.worker_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.worker_spin.setValue(self.worker_count)
        self.worker_spin.setToolTip(
            "Processes used for buffered grain generation and batched same-layer CSL searches"
        )
        self.worker_spin.valueChanged.connect(self._on_worker_count_changed)
        display_layout.addWidget(self.worker_spin, 4, 1)
        self.cell_check = QtWidgets.QCheckBox("Show automatic cell")
        self.cell_check.setChecked(True)
        self.cell_check.setToolTip(
            "Layer-preserving CSL / Near-CSL periodic cell, not a GB structural unit"
        )
        self.cell_check.toggled.connect(self._update_common_cell)
        display_layout.addWidget(self.cell_check, 5, 0)
        self.cell_fit_button = QtWidgets.QPushButton("Fit cell")
        self.cell_fit_button.setToolTip("Fit the exact or strained common cell in view")
        self.cell_fit_button.clicked.connect(self._fit_near_cell)
        display_layout.addWidget(self.cell_fit_button, 5, 1)
        layout.addWidget(display_box)

        manual_box = QtWidgets.QGroupBox("MANUAL COMMON CELL")
        manual_layout = QtWidgets.QVBoxLayout(manual_box)
        self.manual_pick_button = QtWidgets.QPushButton("Pick 4 CSL vertices   M")
        self.manual_pick_button.setCheckable(True)
        self.manual_pick_button.clicked.connect(self._start_manual_cell)
        manual_layout.addWidget(self.manual_pick_button)
        manual_help = QtWidgets.QLabel(
            "Pick 4 same-layer/symbol sites around the perimeter. Blue/red cells use their own atom vertices; count the picked layer only."
        )
        manual_help.setWordWrap(True)
        manual_help.setObjectName("mutedLabel")
        manual_layout.addWidget(manual_help)
        manual_actions = QtWidgets.QHBoxLayout()
        self.manual_undo_button = QtWidgets.QPushButton("Undo vertex")
        self.manual_undo_button.clicked.connect(self._undo_manual_vertex)
        self.manual_clear_button = QtWidgets.QPushButton("Clear")
        self.manual_clear_button.clicked.connect(self._clear_manual_cell)
        self.manual_fit_button = QtWidgets.QPushButton("Fit")
        self.manual_fit_button.setEnabled(False)
        self.manual_fit_button.clicked.connect(self._fit_manual_cell)
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
            "Counts always use the vertices' layer. Optionally apply G1/G2 side switches; the visible-layer selector never changes the counted layer."
        )
        self.manual_visible_check.toggled.connect(self._queue_manual_count)
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
        self.manual_strain_button.clicked.connect(self._toggle_manual_strain)
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
        layout.addWidget(manual_box)

        near_box = QtWidgets.QGroupBox("NEAR-CSL · METHOD")
        near_layout = QtWidgets.QVBoxLayout(near_box)
        self.near_method_combo = QtWidgets.QComboBox()
        self.near_method_combo.addItem("Local matching · no bulk strain", "local")
        self.near_method_combo.addItem("Homogeneous strain + periodic cell", "strain")
        self.near_method_combo.currentIndexChanged.connect(self._on_near_method_changed)
        near_layout.addWidget(self.near_method_combo)
        self.near_button = QtWidgets.QPushButton("Enable Near-CSL")
        self.near_button.setCheckable(True)
        self.near_button.toggled.connect(self._toggle_near)
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
        self.local_distance_spin.valueChanged.connect(self._queue_near_search)
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
            control.valueChanged.connect(self._queue_near_search)
        self.near_combo = QtWidgets.QComboBox()
        self.near_combo.setEnabled(False)
        self.near_combo.currentIndexChanged.connect(self._select_near_cell)
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

        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("statusCard")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(112)
        layout.addWidget(self.status_label)

        export_button = QtWidgets.QPushButton("Export plot as PNG…")
        export_button.clicked.connect(self._choose_export_path)
        layout.addWidget(export_button)
        layout.addStretch(1)
        self._sync_near_controls()
        self._sync_angle_controls()

    def _create_timers(self) -> None:
        self.angle_preview_timer = QtCore.QTimer(self)
        self.angle_preview_timer.setSingleShot(True)
        self.angle_preview_timer.setInterval(18)
        self.angle_preview_timer.timeout.connect(self._apply_pending_angle)
        self.coincidence_timer = QtCore.QTimer(self)
        self.coincidence_timer.setSingleShot(True)
        self.coincidence_timer.setInterval(140)
        self.coincidence_timer.timeout.connect(self._finish_angle_update)
        self.view_refresh_timer = QtCore.QTimer(self)
        self.view_refresh_timer.setSingleShot(True)
        self.view_refresh_timer.setInterval(90)
        self.view_refresh_timer.timeout.connect(self._refresh_view_buffer)
        self.parallel_poll_timer = QtCore.QTimer(self)
        self.parallel_poll_timer.setInterval(16)
        self.parallel_poll_timer.timeout.connect(self._poll_parallel_work)
        self.near_debounce_timer = QtCore.QTimer(self)
        self.near_debounce_timer.setSingleShot(True)
        self.near_debounce_timer.setInterval(250)
        self.near_debounce_timer.timeout.connect(self._start_near_search)
        self.near_poll_timer = QtCore.QTimer(self)
        self.near_poll_timer.setInterval(25)
        self.near_poll_timer.timeout.connect(self._poll_near_search)
        self.manual_count_timer = QtCore.QTimer(self)
        self.manual_count_timer.setInterval(30)
        self.manual_count_timer.timeout.connect(self._poll_manual_count)

    def _create_shortcuts(self) -> None:
        shortcuts = (
            ("R", self._start_new_boundary),
            ("V", self._start_vector_measurement),
            ("M", self._start_manual_cell),
            ("C", self._reset_view),
            ("F", lambda: self._set_region_states((True, True, True, True))),
            ("1", lambda: self._set_region_states((True, False, False, True))),
            ("2", lambda: self._set_region_states((False, True, True, False))),
            ("Escape", lambda: self._set_mode("idle")),
        )
        self.shortcuts: list[QtGui.QShortcut] = []
        for key, callback in shortcuts:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

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
        return max(5.0, float(np.sqrt(self.parameters.marker_size) * 1.55))

    # ---------- view and rendering ----------

    def _to_view(self, points):
        return np.asarray(points) @ rotation_matrix_2d(self.display_rotation_deg).T

    def _from_view(self, points):
        return np.asarray(points) @ rotation_matrix_2d(self.display_rotation_deg)

    def _model_view_bounds(self):
        x0, x1, y0, y1 = self._view_range()
        corners = self._from_view(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        low, high = corners.min(axis=0), corners.max(axis=0)
        return low[0], high[0], low[1], high[1]

    def _on_display_rotation(self, angle):
        angle = float(angle)
        if abs(angle - self.display_rotation_deg) < 1e-10:
            return
        center, width, height = self._view_geometry()
        model_center = self._from_view(center)
        self.display_rotation_deg = angle
        with QtCore.QSignalBlocker(self.rotation_spin), QtCore.QSignalBlocker(
            self.rotation_slider
        ):
            self.rotation_spin.setValue(angle)
            self.rotation_slider.setValue(round(angle * 10))
        center = self._to_view(model_center)
        self.view_box.setRange(
            xRange=(center[0] - width / 2, center[0] + width / 2),
            yRange=(center[1] - height / 2, center[1] + height / 2),
            padding=0,
        )
        self._update_visible_points()
        self._update_common_cell()
        self._draw_boundary()
        self._draw_vector()
        self._draw_manual_cell()
        self._update_title()
        if not self._buffer_contains_view():
            self.view_refresh_timer.start()

    def _start_manual_cell(self, *_args):
        if self.manual_strain_fit is not None:
            self._manual_pick_warning(
                "Restore the local structure before editing cell vertices."
            )
            return
        if self.interaction_mode == "cell":
            self._set_mode("idle")
            return
        if len(self.manual_vertices) == 4:
            self._clear_manual_cell()
        self._set_mode("cell")

    def _clear_manual_cell(self, *_args):
        self.manual_vertices = []
        self.manual_counts = None
        self.manual_count_error = None
        self.manual_count_key = None
        self.manual_count_pending = None
        self.manual_local_cutoff = None
        if self.manual_count_future is not None and self.manual_count_future.cancel():
            self.manual_count_future = None
        self.manual_info.setPlainText(
            "No manual cell. Counts use the picked layer and each grain's own atom vertices; periodicity is not assumed."
        )
        self._draw_manual_cell()
        if self.interaction_mode == "cell":
            self._set_mode("idle")

    def _undo_manual_vertex(self, *_args):
        if self.manual_strain_fit is not None:
            return
        if not self.manual_vertices:
            return
        self.manual_vertices.pop()
        self.manual_counts = None
        self.manual_count_error = None
        self.manual_count_key = None
        self.manual_count_pending = None
        self.manual_info.setPlainText("Continue picking vertices in perimeter order.")
        self._draw_manual_cell()
        self._set_mode("cell")

    def _invalidate_manual_local_source(self):
        if any(vertex.source == "local" for vertex in self.manual_vertices) and (
            not self.local_active
            or self.manual_local_cutoff != self.local_distance_spin.value()
        ):
            self._clear_manual_cell()

    def _sync_manual_strain_controls(self):
        active = self.manual_strain_fit is not None
        with QtCore.QSignalBlocker(self.near_method_combo):
            self.near_method_combo.setItemText(
                0,
                (
                    "Local cell · bulk strain applied"
                    if active
                    else "Local matching · no bulk strain"
                ),
            )
        ready = (
            self.local_active
            and len(self.manual_vertices) == 4
            and any(vertex.source == "local" for vertex in self.manual_vertices)
        )
        self.manual_strain_button.setEnabled(active or ready)
        self.manual_strain_limit.setEnabled(ready and not active)
        self.manual_rotation_limit.setEnabled(ready and not active)
        self.manual_strain_button.setText(
            "Restore original local structure"
            if active
            else "Apply bulk strain to selected cell"
        )
        for button in (
            self.manual_pick_button,
            self.manual_clear_button,
            self.manual_undo_button,
        ):
            button.setEnabled(not active)
        if not active and len(self.manual_vertices) < 4:
            self.manual_strain_note.setText(
                "Select a local near-CSL cell first. Both grains share the deformation; strain and small rotations have separate limits."
            )

    def _toggle_manual_strain(self, *_args):
        if self.manual_strain_fit is not None:
            vertices = self.manual_unstrained_vertices
            cutoff = self.manual_unstrained_cutoff
            self._cancel_parallel_work()
            self._clear_near_cell()
            self._sync_near_controls()
            self.manual_vertices = vertices
            self.manual_local_cutoff = cutoff
            self._draw_manual_cell()
            self._start_parallel_regeneration(True)
            self._queue_manual_count()
            self.manual_strain_note.setText(
                "Original local structure restored; no bulk strain."
            )
            return
        if not (
            self.local_active
            and len(self.manual_vertices) == 4
            and any(vertex.source == "local" for vertex in self.manual_vertices)
        ):
            self._manual_pick_warning(
                "First select four same-layer vertices using Local matching."
            )
            return
        try:
            fit = strain_selected_cell(
                self._manual_grain_polygons(),
                self.angle_deg,
                self.manual_strain_limit.value(),
                self.geometry.lattice,
                self.geometry.axis,
                self.manual_vertices[0].layer,
                self.manual_rotation_limit.value(),
            )
        except ValueError as error:
            self.manual_strain_note.setText(str(error))
            self._update_status(str(error))
            return
        vertices = self.manual_vertices
        cutoff = self.manual_local_cutoff
        self.near_debounce_timer.stop()
        self.near_poll_timer.stop()
        if self.near_search is not None:
            self.near_search.cancel()
        self._cancel_parallel_work()
        self._clear_lattice_selections()
        self.manual_unstrained_vertices = vertices
        self.manual_unstrained_cutoff = cutoff
        self.manual_strain_fit = fit
        self.near_cell = fit.cell
        self.deformations = (fit.cell.f1, fit.cell.f2)
        self.translations = fit.translations.copy()
        self.manual_vertices = [
            CellVertex(pair.mean(axis=0), vertex.layer, "CSL", pair.copy())
            for vertex, pair in zip(vertices, fit.vertices.transpose(1, 0, 2))
        ]
        self._clear_local_pairs()
        self._sync_near_controls()
        self._draw_manual_cell()
        self.manual_strain_note.setText(
            f"Applied: max principal strain {100*fit.cell.max_strain:.6f}%; "
            f"rotations G1 {fit.rotations_deg[0]:+.6f}°, G2 {fit.rotations_deg[1]:+.6f}°; "
            f"four-pair residual {fit.residual:.2e} a₀. Full-lattice CSL is recomputed, layer by layer."
        )
        self.near_info.setPlainText(
            "SELECTED-CELL BULK STRAIN (not stress-free)\n"
            "All four original pairs aligned. Polar decomposition F = R U:\n"
            + "\n".join(
                f"G{g+1}: rotation {fit.rotations_deg[g]:+.8f}°; principal strains "
                + np.array2string(100 * (fit.stretches[g] - 1), precision=8)
                + " %"
                for g in (0, 1)
            )
            + f"\nReference θ = {self.angle_deg:.8f}°; polar-frame θ = {self.angle_deg+fit.rotations_deg[0]-fit.rotations_deg[1]:.8f}°.\n"
            "Uniform shifts align the two cell centroids at their original midpoint.\n"
            "t1/t2 / a₀ (analysis x,y):\n"
            + np.array2string(fit.translations, precision=8)
            + "\n"
            + tensor_readout(fit.cell, self.angle_deg, selected=True)
        )
        self._start_parallel_regeneration(True)
        self._queue_manual_count()
        self._update_title()

    def _common_site_candidates(self):
        """All displayed markers: reject wrong-layer clicks *after* hit-testing."""
        if self.grain_signature != self._geometry_signature():
            return []
        candidates = []
        states = self._region_states()
        if not self.csl_updating:
            for layer, points in enumerate(self.coincident_points):
                if self.selected_layer >= 0 and layer != self.selected_layer:
                    continue
                keep = np.ones(len(points), dtype=bool)
                if len(self.selected_points) == 2:
                    for grain in (0, 1):
                        keep &= selected_region_mask(
                            points,
                            *self.selected_points,
                            states[2 * grain],
                            states[2 * grain + 1],
                        )
                candidates.extend(
                    CellVertex(point.copy(), layer, "CSL") for point in points[keep]
                )
        if self.local_active and not self.local_updating:
            keep = self._local_pair_mask()
            candidates.extend(
                CellVertex(point.copy(), int(layer), "local", np.stack((first, second)))
                for point, layer, first, second in zip(
                    self.local_pairs.midpoints[keep],
                    self.local_pairs.layers[keep],
                    self.local_pairs.first[keep],
                    self.local_pairs.second[keep],
                )
            )
        return candidates

    @staticmethod
    def _cell_layer_label(layer):
        symbol = LAYER_SYMBOLS[layer % len(LAYER_SYMBOLS)]
        name = {"o": "circle", "d": "diamond", "t": "triangle", "s": "square"}.get(
            symbol, symbol
        )
        return f"{layer_name(layer)} ({name})"

    def _resolve_cell_vertex(self, candidate):
        if candidate.grain_positions is not None:
            return candidate
        # Exact CSL stores marker centers; resolve its two real atoms once at
        # selection time, not later from a possibly panned-away drawing buffer.
        endpoints = []
        for grain in self.grains:
            points = grain.positions[grain.layers == candidate.layer]
            if not len(points):
                raise ValueError(
                    "Common-site atoms are not available; wait for the lattice update"
                )
            distances = np.linalg.norm(points - candidate.position, axis=1)
            index = int(np.argmin(distances))
            if distances[index] > COINCIDENCE_TOLERANCE_FACTOR:
                raise ValueError(
                    "Common-site atoms no longer match this marker; pick again after updating"
                )
            endpoints.append(points[index].copy())
        return replace(candidate, grain_positions=np.array(endpoints))

    def _manual_grain_polygons(self):
        if not self.manual_vertices:
            return np.empty((2, 0, 2))
        return np.stack(
            [vertex.grain_positions for vertex in self.manual_vertices], axis=1
        )

    def _manual_pick_warning(self, message):
        # Keep rejection feedback beside the picking controls, even if the
        # general status label is outside the currently scrolled panel.
        self.manual_info.setPlainText(message)
        self._update_status(message)

    def _pick_manual_vertex(self, position):
        candidates = self._common_site_candidates()
        if not candidates:
            self._update_status(
                "No visible common sites; enable Near-CSL or choose a CSL angle."
            )
            return
        positions = np.array([candidate.position for candidate in candidates])
        pixel_size = np.asarray(self.view_box.viewPixelSize())
        delta = self._to_view(positions - position) / np.maximum(pixel_size, 1e-12)
        distance = np.linalg.norm(delta, axis=1)
        index = int(np.argmin(distance))
        if distance[index] > max(14.0, self._base_marker_diameter()):
            self._update_status(
                "Click a gold CSL or purple near-pair marker in the first vertex's layer."
            )
            return
        candidate = candidates[index]
        hit_layers = {
            candidates[i].layer
            for i in np.flatnonzero(np.abs(distance - distance[index]) < 1e-6)
        }
        if len(hit_layers) > 1:
            self._manual_pick_warning(
                "Overlapping markers from different layers; isolate one layer with Visible axial layers, then pick again. Point not selected."
            )
            return
        if self.manual_vertices and candidate.layer != self.manual_vertices[0].layer:
            self._manual_pick_warning(
                f"Wrong layer/symbol: expected {self._cell_layer_label(self.manual_vertices[0].layer)}, "
                f"clicked {self._cell_layer_label(candidate.layer)}. Point not selected."
            )
            return
        if any(
            np.linalg.norm(candidate.position - v.position) < 1e-8
            for v in self.manual_vertices
        ):
            self._update_status("Choose a different common-site vertex.")
            return
        try:
            candidate = self._resolve_cell_vertex(candidate)
        except ValueError as error:
            self._update_status(str(error))
            return
        trial = self.manual_vertices + [candidate]
        if len(trial) == 4:
            for grain_index in (0, 1):
                try:
                    validate_cell_vertices(
                        [vertex.grain_positions[grain_index] for vertex in trial]
                    )
                except ValueError as error:
                    self._manual_pick_warning(
                        f"G{grain_index+1}: {error}; Undo vertex if needed. Point not selected."
                    )
                    return
        self.manual_vertices = trial
        if candidate.source == "local":
            self.manual_local_cutoff = self.local_distance_spin.value()
        self._draw_manual_cell()
        if len(trial) == 4:
            self._set_mode("idle")
            self._queue_manual_count()
        else:
            self.manual_info.setPlainText(
                f"Selected {len(trial)}/4 vertices in layer {self._cell_layer_label(candidate.layer)}. "
                "Continue around the perimeter in the same layer."
            )
            self._update_status()

    def _draw_manual_cell(self):
        self._sync_manual_strain_controls()
        points = np.array([v.position for v in self.manual_vertices]).reshape(-1, 2)
        if self.manual_vertices:
            self.manual_vertex_item.setSymbol(
                LAYER_SYMBOLS[self.manual_vertices[0].layer % len(LAYER_SYMBOLS)]
            )
        self._set_scatter(self.manual_vertex_item, points)
        displayed = self._to_view(points)
        for polygon, item in zip(
            self._manual_grain_polygons(), self.manual_grain_cell_items
        ):
            grain_view = self._to_view(polygon)
            outline = (
                np.vstack((grain_view, grain_view[0]))
                if len(points) == 4
                else grain_view
            )
            item.setData(outline[:, 0], outline[:, 1])
        self.manual_fit_button.setEnabled(len(points) == 4)
        for index, label in enumerate(self.manual_labels):
            label.setVisible(index < len(points))
            if index < len(points):
                vertex = self.manual_vertices[index]
                label.setText(
                    f"C{index+1} · {layer_name(vertex.layer)} · {vertex.source}"
                )
                label.setPos(*displayed[index])
        self.manual_annotation.setVisible(len(points) == 4)
        if len(points) == 4:
            if self.manual_counts is None:
                label = (
                    "Manual cell · count unavailable"
                    if self.manual_count_error
                    else "Manual cell · counting…"
                )
            else:
                counts = self.manual_counts
                layer = self.manual_vertices[0].layer
                lines = [f"Manual cell · layer {self._cell_layer_label(layer)} only"]
                for grain in (0, 1):
                    half_open = counts.half_open_available[grain]
                    value = (
                        counts.half_open[grain, layer]
                        if half_open
                        else (counts.interior + counts.boundary)[grain, layer]
                    )
                    convention = "half-open" if half_open else "closed"
                    lines.append(
                        f"G{grain+1}: {value} {convention} · {counts.interior[grain,layer]} interior"
                    )
                label = "\n".join(lines)
            self.manual_annotation.setText(label)
            self.manual_annotation.setPos(*displayed.mean(axis=0))

    def _queue_manual_count(self, *_args):
        if len(self.manual_vertices) != 4:
            return
        points = self._manual_grain_polygons()
        filtered = self.manual_visible_check.isChecked()
        boundary = (
            np.array(self.selected_points)
            if filtered and len(self.selected_points) == 2
            else None
        )
        states = self._region_states() if filtered else (True, True, True, True)
        layer = self.manual_vertices[0].layer
        key = (
            tuple(points.ravel()),
            self._geometry_signature(),
            filtered,
            layer,
            states,
            tuple(boundary.ravel()) if boundary is not None else None,
        )
        if key == self.manual_count_key:
            return
        self.manual_count_key = key
        self.manual_counts = None
        self.manual_count_error = None
        self.manual_count_pending = (
            points,
            self.angle_deg,
            tuple(f.copy() for f in self.deformations),
            self.geometry.lattice,
            self.geometry.axis,
            boundary,
            states,
            layer,
            self.translations.copy(),
        )
        if self.manual_count_future is not None and self.manual_count_future.cancel():
            self.manual_count_future = None
        self.manual_info.setPlainText(
            "Counting the entire cell in the background (independent of viewport)…"
        )
        self._draw_manual_cell()
        self.manual_count_timer.start()
        self._poll_manual_count()

    def _poll_manual_count(self):
        if self.manual_count_future is not None:
            if not self.manual_count_future.done():
                return
            try:
                pid, counts = self.manual_count_future.result()
                self.worker_process_ids.add(pid)
                if self.manual_count_running_key == self.manual_count_key:
                    self.manual_counts = counts
                    self._show_manual_counts()
            except Exception as error:
                if self.manual_count_running_key == self.manual_count_key:
                    self.manual_count_error = str(error)
                    self.manual_info.setPlainText(
                        f"Cell count unavailable: {error}\nNo partial count is reported."
                    )
                    self.manual_annotation.setText("Manual cell · count unavailable")
            self.manual_count_future = None
        if self.manual_count_pending is not None:
            args, self.manual_count_pending = self.manual_count_pending, None
            executor = self.executor
            if executor is None:
                if self.local_thread_executor is None:
                    self.local_thread_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1
                    )
                executor = self.local_thread_executor
            self.manual_count_running_key = self.manual_count_key
            try:
                self.manual_count_future = executor.submit(cell_count_worker, *args)
            except Exception as error:
                self.manual_count_error = str(error)
                self.manual_info.setPlainText(f"Cell count unavailable: {error}")
                self._draw_manual_cell()
        if self.manual_count_future is None:
            self.manual_count_timer.stop()

    def _show_manual_counts(self):
        counts = self.manual_counts
        layer = self.manual_vertices[0].layer
        lines = [
            (
                "Selected-cell bulk strain; common translations verified."
                if self.manual_strain_fit is not None
                else "Manual region; periodicity not verified."
            ),
            f"Only picked layer {self._cell_layer_label(layer)}; not an all-layer total.",
            "Each grain uses its own four current atom vertices.",
            (
                "GB side visibility applied."
                if self.manual_visible_check.isChecked()
                else "Both grains; independent of view and display-layer filter."
            ),
        ]
        for grain in (0, 1):
            interior = counts.interior[grain, layer]
            boundary = counts.boundary[grain, layer]
            lines.append(
                f"G{grain+1} ({'blue' if grain == 0 else 'red'}): area {counts.areas[grain]:.6g} a₀²"
            )
            lines.append(
                f"  Layer {layer_name(layer)}: interior {interior}, boundary {boundary}, closed {interior+boundary}"
            )
            if counts.half_open_available[grain]:
                lines.append(
                    f"  Half-open count: {counts.half_open[grain,layer]} (upper edges excluded)"
                )
            else:
                lines.append(
                    "  Not a parallelogram: no half-open count for this grain."
                )
        lines.append("G1/G2 counted separately; overlapping atoms are not merged.")
        self.manual_info.setPlainText("\n".join(lines))
        self._draw_manual_cell()

    def _fit_manual_cell(self, *_args):
        if len(self.manual_vertices) == 4:
            self._fit_model_corners(self._manual_grain_polygons().reshape(-1, 2))

    def _empty_coincidences(self):
        return tuple(np.empty((0, 2)) for _ in range(self.geometry.layer_count))

    def _populate_layer_combo(self):
        with QtCore.QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            self.layer_combo.addItem("All layers", -1)
            for layer in range(self.geometry.layer_count):
                self.layer_combo.addItem(
                    f"{layer_name(layer)} · z/a₀ = {layer*self.geometry.layer_spacing:.6g}",
                    layer,
                )
            self.layer_combo.setCurrentIndex(self.selected_layer + 1)

    def _on_layer_changed(self, *_args):
        self.selected_layer = self.layer_combo.currentData()
        self._rebuild_legend()
        self._update_visible_points()

    def _geometry_signature(self):
        """Identify physical positions independently of the view rectangle."""
        return (
            self.geometry.lattice,
            self.geometry.axis,
            self.angle_deg,
            tuple(self.deformations[0].ravel()),
            tuple(self.deformations[1].ravel()),
            tuple(self.translations.ravel()),
        )

    def _update_geometry_labels(self):
        model = self.geometry
        name = f"{model.lattice} ⟨{model.axis}⟩ Tilt GB"
        self.setWindowTitle(name + " · Dichromatic Pattern")
        self.heading.setText(
            f"<div style='font-size:17px;font-weight:700;color:#17212b'>{name}</div>"
            "<div style='color:#64748b;margin-top:3px'>Coordinates in a₀ · 1 = one lattice constant</div>"
        )
        self.plot_item.setLabel("bottom", f"{model.x_label} / a₀", units="")
        self.plot_item.setLabel("left", f"{model.y_label} / a₀", units="")
        self.layer_info.setText(
            f"{model.layer_count} axial layers · spacing = {model.layer_spacing:.6g} a₀\n"
            f"Axial repeat = {model.axial_period:.6g} a₀"
            + (
                "\nLegend shows first 6 layers; use the layer selector to inspect others."
                if model.layer_count > 6
                else ""
            )
        )
        self.axial_spin.setToolTip(
            "P2 shifts by "
            + self._format_lattice_vector(model.axial_repeat_half_indices)
            + f" per step along {model.axis_label}"
        )

    def _on_geometry_changed(self, *_args, apply_custom=False):
        lattice = self.structure_combo.currentData()
        axis = self.axis_combo.currentData()
        self.custom_axis_row.setVisible(axis is None)
        if axis is None:
            if not apply_custom:
                return
            axis = self.custom_axis_edit.text()
        try:
            geometry = get_geometry(lattice, axis)
        except ValueError as error:
            self.axis_error.setText(str(error))
            self.axis_error.show()
            with QtCore.QSignalBlocker(self.structure_combo):
                self.structure_combo.setCurrentIndex(
                    self.structure_combo.findData(self.geometry.lattice)
                )
            return
        self.axis_error.clear()
        self.axis_error.hide()
        axis = geometry.axis
        if (lattice, axis) == (self.geometry.lattice, self.geometry.axis):
            return
        axis_changed = axis != self.geometry.axis
        for timer in (
            self.angle_preview_timer,
            self.coincidence_timer,
            self.view_refresh_timer,
            self.near_debounce_timer,
            self.near_poll_timer,
        ):
            timer.stop()
        self._cancel_parallel_work()
        if self.near_search is not None:
            self.near_search.cancel()
        self._clear_lattice_selections()
        self.geometry = geometry
        self.render_error = None
        self.parameters = replace(self.parameters, lattice=lattice, axis=axis)
        self.presets = csl_presets(axis)
        if axis_changed:
            self.angle_deg = default_angle_deg(axis)
        self.pending_angle = self.angle_deg
        self.angle_update_active = False
        self._clear_near_cell()
        with QtCore.QSignalBlocker(self.preset_combo):
            self.preset_combo.clear()
            self.preset_combo.addItem("Custom angle", None)
            for index, preset in enumerate(self.presets):
                self.preset_combo.addItem(preset.label, index)
        self.axial_repeat = 0
        with QtCore.QSignalBlocker(self.axial_spin):
            self.axial_spin.setValue(0)
            self.axial_spin.setPrefix("+")
        self._sync_angle_controls()
        self._update_geometry_labels()
        # Hide the previous geometry while its replacement is generated.
        self.grains = []
        self.grain_signature = None
        self.visible_atom_masks = []
        self.visible_atom_counts = [0, 0]
        self.visible_coincidence_counts = [0] * geometry.layer_count
        self.coincident_points = self._empty_coincidences()
        self.selected_layer = -1
        self._populate_layer_combo()
        self._create_layer_items()
        self._update_marker_sizes()
        self.buffer_bounds = None
        self._update_title()
        if self.near_enabled:
            self._queue_near_search()
        else:
            self._start_parallel_regeneration(True)
            self.near_info.setPlainText("Off. Original unstrained lattices displayed.")
        self._update_status(
            f"Changed to {lattice} {self.geometry.axis_label}; previous selections cleared."
        )

    @property
    def common_cell(self):
        return (
            self.near_cell
            if self.near_cell is not None
            else exact_csl_cell(
                self.angle_deg, lattice=self.geometry.lattice, axis=self.geometry.axis
            )
        )

    def _update_common_cell(self, *_args):
        cell = self.common_cell
        self.cell_fit_button.setEnabled(cell is not None)
        if cell is None:
            self.near_cell_item.setData([], [])
        else:
            corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ cell.cell.T
            if self.manual_strain_fit is not None:
                corners += self.manual_strain_fit.origin
            corners = self._to_view(corners)
            self.near_cell_item.setData(corners[:, 0], corners[:, 1])
        self.near_cell_item.setVisible(cell is not None and self.cell_check.isChecked())

    def _clear_near_cell(self):
        if self.near_cell is not None:
            self._clear_lattice_selections()
        self.near_cell = None
        self.near_solutions = []
        self.deformations = (np.eye(2), np.eye(2))
        self.translations = np.zeros((2, 2))
        self.manual_strain_fit = None
        self.manual_unstrained_vertices = None
        self.manual_unstrained_cutoff = None
        self.manual_strain_note.setText(
            "Select a local near-CSL cell first. Both grains share the strain; no individual atoms are snapped."
        )
        # A geometry switch can be between updating its model and rebuilding
        # per-layer plot items here; do not rebuild the legend in that interval.
        self.local_distance_spin.setEnabled(self.local_active)
        self._sync_manual_strain_controls()
        self._update_common_cell()
        with QtCore.QSignalBlocker(self.near_combo):
            self.near_combo.clear()

    @property
    def local_active(self):
        return (
            self.near_enabled
            and self.near_method == "local"
            and self.manual_strain_fit is None
        )

    def _clear_local_pairs(self):
        self.local_pairs = LocalPairs.empty()
        self.local_error = None
        self.local_updating = self.local_active
        self._set_scatter(self.local_match_item, np.empty((0, 2)))
        self.local_link_item.setData([], [])

    def _sync_near_controls(self):
        strain_method = self.near_method == "strain"
        strained = self.near_enabled and strain_method
        for control in (self.strain_spin, self.search_index_spin, self.near_combo):
            control.setEnabled(strained)
            control.setVisible(strain_method)
            if control is not self.near_combo:
                self.near_form.labelForField(control).setVisible(strain_method)
        self.local_distance_spin.setEnabled(self.local_active)
        self.local_distance_spin.setVisible(not strain_method)
        self.near_form.labelForField(self.local_distance_spin).setVisible(
            not strain_method
        )
        self._rebuild_legend()
        self._sync_manual_strain_controls()

    def _on_near_method_changed(self, *_args):
        self.near_method = self.near_method_combo.currentData()
        self._sync_near_controls()
        if self.near_enabled:
            self._queue_near_search()
        else:
            self.near_info.setPlainText(
                "Off. Enable Near-CSL to run the selected method."
            )
        self._update_title()

    def _toggle_near(self, enabled):
        self.near_enabled = bool(enabled)
        self._invalidate_manual_local_source()
        self._sync_near_controls()
        self.near_button.setText("Disable Near-CSL" if enabled else "Enable Near-CSL")
        if enabled:
            self._queue_near_search()
        else:
            self.near_debounce_timer.stop()
            self.near_poll_timer.stop()
            if self.near_search is not None:
                self.near_search.cancel()
            self._clear_local_pairs()
            self._clear_near_cell()
            self._cancel_parallel_work()
            self._start_parallel_regeneration(True)
            self.near_info.setPlainText("Off. Original unstrained lattices restored.")
        self._update_title()

    def _queue_near_search(self, *_args):
        if not self.near_enabled:
            return
        if self.manual_strain_fit is not None and self.near_method == "local":
            return  # Keep selected-cell strain on worker-count changes / fallback.
        self._invalidate_manual_local_source()
        self.near_debounce_timer.stop()
        self.near_poll_timer.stop()
        if self.near_search is not None:
            self.near_search.cancel()
        self._clear_local_pairs()
        self._clear_near_cell()
        if self.local_active:
            # A threshold change only changes annotations, preserving picks.
            # Rebuild only if switching away from strain or needing a halo.
            if self.parallel_stage == "local":
                self._cancel_parallel_work()
            if (
                self.grain_signature != self._geometry_signature()
                or not self._buffer_contains_view()
            ):
                self._start_parallel_regeneration(True)
            else:
                self.near_debounce_timer.start()
            self._update_local_overlay()
            self._update_title()
            return
        self._cancel_parallel_work()
        self._start_parallel_regeneration(True)
        self.near_info.setPlainText("Waiting for parameters to settle…")
        self.near_debounce_timer.start()

    def _start_near_search(self):
        if not self.near_enabled or self.manual_strain_fit is not None:
            return
        if self.local_active:
            self._start_local_matching()
            return
        if self.near_search is None:
            self.near_search = NearSearch(self.worker_count, executor=self.executor)
        self.near_search.request(
            self.angle_deg,
            self.strain_spin.value(),
            self.search_index_spin.value(),
            lattice=self.geometry.lattice,
            axis=self.geometry.axis,
        )
        self.near_poll_timer.start()

    def _poll_near_search(self):
        if (
            self.near_search is None
            or not self.near_enabled
            or self.near_method != "strain"
        ):
            self.near_poll_timer.stop()
            return
        result = self.near_search.poll()
        if self.near_search.error:
            self.near_info.setPlainText(
                "Near-CSL search failed: " + self.near_search.error
            )
            self.near_poll_timer.stop()
        elif result is not None:
            self.near_poll_timer.stop()
            self.near_solutions = result
            with QtCore.QSignalBlocker(self.near_combo):
                self.near_combo.clear()
                self.near_combo.addItems([cell.label() for cell in result])
            if result:
                self._select_near_cell(0)
            else:
                self.near_info.setPlainText(
                    "No compatible cell found within this bounded search.\n"
                    "Increase the index bound or principal strain limit.\nUnstrained lattices remain displayed."
                )
        elif self.near_search.busy:
            self.near_info.setPlainText(
                f"Searching strained periodic cells…\n"
                f"{self.near_search.completed}/{self.near_search.total} chunks complete\n"
                f"{self.worker_count} workers; navigation remains available."
            )

    def _select_near_cell(self, index):
        if (
            not self.near_enabled
            or self.local_active
            or not 0 <= index < len(self.near_solutions)
        ):
            return
        self._cancel_parallel_work()
        self._clear_lattice_selections()
        self.near_cell = self.near_solutions[index]
        self.deformations = (self.near_cell.f1, self.near_cell.f2)
        self.near_info.setPlainText(tensor_readout(self.near_cell, self.angle_deg))
        self._update_common_cell()
        self._start_parallel_regeneration(True)

    def _start_local_matching(self):
        if not self.local_active or len(self.grains) != 2:
            return
        if self.parallel_stage in ("grains", "coincidences"):
            # Exact detection will start matching after the new atoms arrive.
            if self.parallel_stage == "grains":
                self.parallel_payload["compute_coincidences"] = True
            return
        if self.grain_signature != self._geometry_signature():
            return
        self.near_debounce_timer.stop()
        self._cancel_parallel_work()
        self._clear_local_pairs()
        executor = self.executor
        if executor is None:
            if self.local_thread_executor is None:
                self.local_thread_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1
                )
            executor = self.local_thread_executor
        groups = np.array_split(
            np.arange(self.geometry.layer_count),
            min(self.worker_count, self.geometry.layer_count),
        )
        distance = self.local_distance_spin.value()
        self.parallel_stage = "local"
        self.parallel_payload = {
            "geometry_signature": self.grain_signature,
            "distance": distance,
        }
        try:
            self.parallel_futures = [
                executor.submit(
                    local_match_layers_worker,
                    self.grains[0],
                    self.grains[1],
                    distance,
                    layers,
                )
                for layers in groups
            ]
        except Exception as error:
            self._parallel_failure(error)
            return
        self.parallel_poll_timer.start()
        self._update_local_overlay()
        self._update_status()

    def _local_pair_mask(self):
        """Visibility in physical coordinates; also used for manual picking."""
        pairs = self.local_pairs
        mask = np.full(len(pairs.layers), self.local_active, dtype=bool)
        if self.grain_signature != self._geometry_signature():
            mask[:] = False
        if self.selected_layer >= 0:
            mask &= pairs.layers == self.selected_layer
        if len(self.selected_points) == 2:
            states = self._region_states()
            for grain, points in enumerate((pairs.first, pairs.second)):
                mask &= selected_region_mask(
                    points,
                    *self.selected_points,
                    states[2 * grain],
                    states[2 * grain + 1],
                )
        midpoints = pairs.midpoints
        # Reject buffer-edge assignments lacking the full mutual-neighbor halo.
        # All displayed view points have a 2*d halo after buffered regeneration.
        if self.local_active and self.buffer_bounds is not None:
            left, right, bottom, top = self.buffer_bounds
            halo = 2 * self.local_distance_spin.value()
            mask &= (
                (midpoints[:, 0] >= left + halo)
                & (midpoints[:, 0] <= right - halo)
                & (midpoints[:, 1] >= bottom + halo)
                & (midpoints[:, 1] <= top - halo)
            )
        return mask

    def _update_local_overlay(self):
        """Filter both original endpoints, never just the proposed midpoint."""
        pairs = self.local_pairs
        mask = self._local_pair_mask()
        midpoints = pairs.midpoints
        self.local_match_item.setData(
            pos=self._to_view(midpoints[mask]),
            symbol=[
                LAYER_SYMBOLS[int(layer) % len(LAYER_SYMBOLS)]
                for layer in pairs.layers[mask]
            ],
        )
        links = np.stack((pairs.first[mask], pairs.second[mask]), axis=1).reshape(-1, 2)
        links = self._to_view(links)
        self.local_link_item.setData(links[:, 0], links[:, 1], connect="pairs")
        if not self.local_active:
            return
        x0, x1, y0, y1 = self._view_range()
        midpoints = self._to_view(midpoints)
        visible = mask & (
            (midpoints[:, 0] >= x0)
            & (midpoints[:, 0] <= x1)
            & (midpoints[:, 1] >= y0)
            & (midpoints[:, 1] <= y1)
        )
        distances = pairs.distances[visible]
        if self.local_error:
            state = "Local matching failed: " + self.local_error
        elif self.local_updating:
            state = f"Matching same-layer neighbors… ({self.worker_count} workers)"
        else:
            state = f"Visible near pairs: {len(distances)} (exact CSL excluded)"
            if len(distances):
                state += f"\nd/a₀ min / mean / max: {distances.min():.5f} / {distances.mean():.5f} / {distances.max():.5f}"
        self.near_info.setPlainText(
            state + f"\nPair cutoff: {self.local_distance_spin.value():.4f} a₀."
            "\nSame-layer mutual nearest pairs; no atoms moved."
            "\nPurple marker: midpoint, shaped by layer; line: original pair."
            "\nMidpoint alignment would move each atom by d/2."
            "\nNo bulk strain, relaxation or stress calculation."
            "\nLocal matches do not define a periodic common cell."
        )

    def _fit_near_cell(self):
        """Fit whichever periodic cell is displayed (exact CSL or Near-CSL)."""
        cell = self.common_cell
        if cell is None:
            return
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ cell.cell.T
        if self.manual_strain_fit is not None:
            corners += self.manual_strain_fit.origin
        self._fit_model_corners(corners)

    def _fit_model_corners(self, corners):
        corners = self._to_view(corners)
        center = (corners.min(axis=0) + corners.max(axis=0)) / 2
        extent = np.ptp(corners, axis=0) * 1.15
        _, old_width, old_height = self._view_geometry()
        aspect = old_height / old_width
        width = np.clip(
            max(extent[0], extent[1] / aspect),
            self.parameters.width * VIEW_SCALE_MIN,
            self.parameters.width * VIEW_SCALE_MAX,
        )
        height = width * aspect
        self.view_box.setRange(
            xRange=(center[0] - width / 2, center[0] + width / 2),
            yRange=(center[1] - height / 2, center[1] + height / 2),
            padding=0,
        )

    def _set_initial_view(self) -> None:
        width = self.parameters.width * self.parameters.view_scale
        height = self.parameters.height * self.parameters.view_scale
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
        if self.local_active:
            # Both directions of a mutual-neighbor query need a 2*d halo.
            halo_width = 4.4 * self.local_distance_spin.value()
            width = max(width, view_width + halo_width)
            height = max(height, view_height + halo_width)
        bounds = (
            center[0] - 0.5 * width,
            center[0] + 0.5 * width,
            center[1] - 0.5 * height,
            center[1] + 0.5 * height,
        )
        return center, width, height, bounds

    def _configure_worker_pool(self, worker_count: int) -> None:
        old_executor = self.executor
        self.executor = None
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=True)
        self.worker_count = int(worker_count)
        if self.worker_count > 1:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initializer,
            )

    def _on_worker_count_changed(self, worker_count: int) -> None:
        recount = len(self.manual_vertices) == 4 and self.manual_counts is None
        self._cancel_parallel_work()
        self.near_poll_timer.stop()
        if self.near_search is not None:
            self.near_search.close()
            self.near_search = None
        self._configure_worker_pool(worker_count)
        self._start_parallel_regeneration(True)
        if self.near_enabled:
            self._queue_near_search()
        self._update_status(f"Compute pool changed to {self.worker_count} workers.")
        if recount:
            self.manual_count_key = None
            self._queue_manual_count()

    def _cancel_parallel_work(self) -> None:
        self.parallel_generation += 1
        for future in self.parallel_futures:
            future.cancel()
        self.parallel_futures = []
        self.parallel_stage = None
        self.parallel_payload = {}
        if hasattr(self, "parallel_poll_timer"):
            self.parallel_poll_timer.stop()

    def _regenerate_buffer(self, compute_coincidences: bool) -> None:
        if self.parallel_stage == "local":
            self._cancel_parallel_work()
        self._clear_local_pairs()
        try:
            self._regenerate_buffer_checked(compute_coincidences)
        except GeometryLimitError as error:
            self.csl_updating = False
            self.angle_update_active = False
            self.render_error = str(error)
            if self.local_active:
                self.local_updating = False
                self.local_error = str(error)
                self._update_local_overlay()
            self._update_status(str(error))

    def _regenerate_buffer_checked(self, compute_coincidences: bool) -> None:
        center, width, height, bounds = self._buffer_geometry()
        half_angle = 0.5 * self.angle_deg
        self.grains = [
            projected_columns(
                width,
                height,
                +half_angle,
                center=tuple(center),
                deformation=self.deformations[0],
                lattice=self.geometry.lattice,
                axis=self.geometry.axis,
                translation=self.translations[0],
            ),
            projected_columns(
                width,
                height,
                -half_angle,
                center=tuple(center),
                deformation=self.deformations[1],
                lattice=self.geometry.lattice,
                axis=self.geometry.axis,
                translation=self.translations[1],
            ),
        ]
        self.buffer_bounds = bounds
        self.render_error = None
        self.grain_signature = self._geometry_signature()
        if compute_coincidences:
            self._compute_coincidences()
        else:
            self.coincident_points = self._empty_coincidences()
            self.csl_updating = True
        self._update_visible_points()
        self._update_title()

    def _start_parallel_regeneration(self, compute_coincidences: bool) -> None:
        if self.executor is None:
            self._regenerate_buffer(compute_coincidences)
            return
        self._cancel_parallel_work()
        self._clear_local_pairs()
        center, width, height, bounds = self._buffer_geometry()
        half_angle = 0.5 * self.angle_deg
        try:
            self.parallel_futures = [
                self.executor.submit(
                    generate_grain_worker,
                    width,
                    height,
                    sign * half_angle,
                    tuple(center),
                    self.deformations[index],
                    self.geometry.lattice,
                    self.geometry.axis,
                    self.translations[index],
                )
                for index, sign in enumerate((1.0, -1.0))
            ]
        except Exception as error:  # pragma: no cover - platform-specific failure
            self._parallel_failure(error)
            return
        self.parallel_payload = {
            "bounds": bounds,
            "compute_coincidences": compute_coincidences,
            "geometry_signature": self._geometry_signature(),
        }
        self.coincident_points = self._empty_coincidences()
        self.csl_updating = True
        for item in self.coincidence_items:
            self._set_scatter(item, np.empty((0, 2)))
        self.parallel_stage = "grains"
        self._update_status()
        self.parallel_poll_timer.start()

    def _start_parallel_coincidences(self) -> None:
        if self.parallel_stage == "grains":
            self.parallel_payload["compute_coincidences"] = True
            return
        if self.executor is None:
            self._compute_coincidences()
            self.angle_update_active = False
            self._update_visible_points()
            return
        if len(self.grains) != 2:
            return
        self._cancel_parallel_work()
        tolerance = COINCIDENCE_TOLERANCE_FACTOR
        layer_groups = np.array_split(
            np.arange(self.geometry.layer_count),
            min(self.worker_count, self.geometry.layer_count),
        )
        try:
            self.parallel_futures = [
                self.executor.submit(
                    coincidence_layers_worker,
                    self.grains[0],
                    self.grains[1],
                    tolerance,
                    layers,
                )
                for layers in layer_groups
            ]
        except Exception as error:  # pragma: no cover - platform-specific failure
            self._parallel_failure(error)
            return
        self.parallel_stage = "coincidences"
        self.csl_updating = True
        self.parallel_poll_timer.start()
        self._update_status()

    def _poll_parallel_work(self) -> None:
        if not self.parallel_futures or not all(
            future.done() for future in self.parallel_futures
        ):
            return
        stage = self.parallel_stage
        try:
            results = [future.result() for future in self.parallel_futures]
        except Exception as error:  # pragma: no cover - platform-specific failure
            self._parallel_failure(error)
            return

        self.worker_process_ids.update(process_id for process_id, _result in results)
        self.parallel_futures = []
        self.parallel_stage = None
        if stage == "grains":
            self.grains = [grain for _process_id, grain in results]
            self.render_error = None
            self.grain_signature = self.parallel_payload["geometry_signature"]
            self.buffer_bounds = self.parallel_payload["bounds"]  # type: ignore[assignment]
            compute_coincidences = bool(
                self.parallel_payload.get("compute_coincidences", True)
            )
            self.parallel_payload = {}
            self._update_visible_points()
            self._update_title()
            if compute_coincidences:
                self._start_parallel_coincidences()
            else:
                self.parallel_poll_timer.stop()
            if not self._buffer_contains_view():
                self.view_refresh_timer.start()
            return

        if stage == "coincidences":
            by_layer = {
                layer: points for _pid, batch in results for layer, points in batch
            }
            self.coincident_points = tuple(
                by_layer[layer] for layer in range(self.geometry.layer_count)
            )
            self.csl_updating = False
            self.angle_update_active = False
            self.parallel_poll_timer.stop()
            self._update_visible_points()
            self._update_title()
            if self.local_active:
                self._start_local_matching()
            return

        if stage == "local":
            self.parallel_poll_timer.stop()
            current = (
                self.local_active
                and self.parallel_payload.get("geometry_signature")
                == self._geometry_signature()
                and self.parallel_payload.get("distance")
                == self.local_distance_spin.value()
            )
            self.parallel_payload = {}
            if current:
                self.local_pairs = LocalPairs.concatenate(
                    batch for _pid, batch in results
                )
                self.local_updating = False
                self._update_local_overlay()
                self._update_status()

    def _parallel_failure(self, error: Exception) -> None:
        local_failure = self.parallel_stage == "local"
        self._cancel_parallel_work()
        if local_failure:
            self.local_updating = False
            self.local_error = str(error)
            self._update_local_overlay()
            self._update_status(
                "Local matching failed; original lattices are unchanged."
            )
            return
        if isinstance(error, GeometryLimitError):
            self.csl_updating = False
            self.angle_update_active = False
            self.render_error = str(error)
            if self.local_active:
                self.local_updating = False
                self.local_error = str(error)
                self._update_local_overlay()
            self._update_status(str(error))
            return
        self.near_poll_timer.stop()
        if self.near_search is not None:
            self.near_search.close()
            self.near_search = None
        failed_executor = self.executor
        self.executor = None
        if failed_executor is not None:
            failed_executor.shutdown(wait=False, cancel_futures=True)
        self.worker_count = 1
        if hasattr(self, "worker_spin"):
            blocker = QtCore.QSignalBlocker(self.worker_spin)
            self.worker_spin.setValue(1)
            del blocker
        self.angle_update_active = False
        self._regenerate_buffer(compute_coincidences=True)
        if self.near_enabled:
            self._queue_near_search()
        self._update_status(
            f"Parallel calculation failed; using one process ({error})."
        )

    def _compute_coincidences(self) -> None:
        if len(self.grains) != 2:
            return
        tolerance = COINCIDENCE_TOLERANCE_FACTOR
        self.coincident_points = same_layer_coincidence_sites(
            self.grains[0], self.grains[1], tolerance
        )
        self.csl_updating = False
        if self.local_active:
            self._start_local_matching()

    def _refresh_view_buffer(self) -> None:
        self._start_parallel_regeneration(
            compute_coincidences=not self.angle_update_active
        )
        if self.angle_update_active:
            self.coincidence_timer.start()

    def _buffer_contains_view(self) -> bool:
        if self.buffer_bounds is None:
            return False
        x_min, x_max, y_min, y_max = self._model_view_bounds()
        bx_min, bx_max, by_min, by_max = self.buffer_bounds
        margin_x = 0.10 * (bx_max - bx_min)
        margin_y = 0.10 * (by_max - by_min)
        if self.local_active:
            margin_x = max(margin_x, 2 * self.local_distance_spin.value())
            margin_y = max(margin_y, 2 * self.local_distance_spin.value())
        return (
            x_min >= bx_min + margin_x
            and x_max <= bx_max - margin_x
            and y_min >= by_min + margin_y
            and y_max <= by_max - margin_y
        )

    def _on_view_range_changed(self, *_args) -> None:
        self._sync_view_scale_control()
        self._update_marker_sizes()
        self._draw_boundary()
        self._draw_vector()
        self._draw_manual_cell()
        self._refresh_visible_counts()
        if not self._buffer_contains_view() and self.parallel_stage != "grains":
            self.view_refresh_timer.start()

    def _on_field_size_changed(self, raw_value: int) -> None:
        scale = raw_value / 100.0
        center, _width, _height = self._view_geometry()
        width = self.parameters.width * scale
        height = self.parameters.height * scale
        self.view_box.setRange(
            xRange=(center[0] - 0.5 * width, center[0] + 0.5 * width),
            yRange=(center[1] - 0.5 * height, center[1] + 0.5 * height),
            padding=0.0,
        )

    def _sync_view_scale_control(self) -> None:
        _center, width, _height = self._view_geometry()
        scale = float(
            np.clip(width / self.parameters.width, VIEW_SCALE_MIN, VIEW_SCALE_MAX)
        )
        blocker = QtCore.QSignalBlocker(self.view_slider)
        self.view_slider.setValue(round(scale * 100.0))
        del blocker
        self.view_scale_label.setText(f"{scale:.2f}×")

    def _update_marker_sizes(self) -> None:
        _center, width, _height = self._view_geometry()
        scale = max(VIEW_SCALE_MIN, width / self.parameters.width)
        diameter = max(3.5, self._base_marker_diameter() / scale**0.22)
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

    def _region_states(self) -> tuple[bool, bool, bool, bool]:
        return tuple(check.isChecked() for check in self.region_checks)  # type: ignore[return-value]

    def _update_visible_points(self, *_args) -> None:
        if len(self.grains) != 2:
            return
        states = self._region_states()
        boundary_ready = len(self.selected_points) == 2
        self.visible_atom_masks = []
        for grain_index, grain in enumerate(self.grains):
            if boundary_ready:
                mask = selected_region_mask(
                    grain.positions,
                    self.selected_points[0],
                    self.selected_points[1],
                    states[2 * grain_index],
                    states[2 * grain_index + 1],
                )
            else:
                mask = np.ones(len(grain.positions), dtype=bool)
            if self.selected_layer >= 0:
                mask &= grain.layers == self.selected_layer
            self.visible_atom_masks.append(mask)
            for layer, item in enumerate(self.grain_layer_items[grain_index]):
                layer_mask = grain.layers == layer
                self._set_scatter(item, grain.positions[mask & layer_mask])

        for layer, (points, item) in enumerate(
            zip(self.coincident_points, self.coincidence_items, strict=True)
        ):
            if boundary_ready:
                first_mask = selected_region_mask(
                    points,
                    self.selected_points[0],
                    self.selected_points[1],
                    states[0],
                    states[1],
                )
                second_mask = selected_region_mask(
                    points,
                    self.selected_points[0],
                    self.selected_points[1],
                    states[2],
                    states[3],
                )
                visible_points = points[first_mask & second_mask]
            else:
                visible_points = points
            if self.selected_layer >= 0 and self.selected_layer != layer:
                visible_points = np.empty((0, 2))
            self._set_scatter(item, visible_points)
        self._refresh_visible_counts()
        self._queue_manual_count()

    def _refresh_visible_counts(self) -> None:
        self._update_local_overlay()
        if len(self.grains) != 2 or len(self.visible_atom_masks) != 2:
            return
        x_min, x_max, y_min, y_max = self._view_range()

        def in_view(points: np.ndarray) -> np.ndarray:
            points = self._to_view(points)
            return (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
            )

        for index, grain in enumerate(self.grains):
            self.visible_atom_counts[index] = int(
                np.count_nonzero(
                    self.visible_atom_masks[index] & in_view(grain.positions)
                )
            )

        states = self._region_states()
        boundary_ready = len(self.selected_points) == 2
        for layer, points in enumerate(self.coincident_points):
            mask = in_view(points)
            if self.selected_layer >= 0 and self.selected_layer != layer:
                mask[:] = False
            if boundary_ready:
                mask &= selected_region_mask(
                    points,
                    self.selected_points[0],
                    self.selected_points[1],
                    states[0],
                    states[1],
                )
                mask &= selected_region_mask(
                    points,
                    self.selected_points[0],
                    self.selected_points[1],
                    states[2],
                    states[3],
                )
            self.visible_coincidence_counts[layer] = int(np.count_nonzero(mask))
        self._update_status()

    def _update_title(self) -> None:
        self._update_common_cell()
        preset = matching_csl_preset(self.angle_deg, axis=self.geometry.axis)
        suffix = f" · {preset.name}" if preset is not None else ""
        if self.manual_strain_fit is not None:
            fit = self.manual_strain_fit
            suffix = f"<br/>SELECTED-CELL BULK STRAIN · {100*self.near_cell.max_strain:.6f}% · ΔR₁/ΔR₂ = {fit.rotations_deg[0]:+.4f}°/{fit.rotations_deg[1]:+.4f}°"
        elif self.near_cell is not None:
            suffix = f" · STRAINED near-CSL · {100*self.near_cell.max_strain:.3f}%"
        elif self.local_active:
            suffix += f" · LOCAL near-CSL · d ≤ {self.local_distance_spin.value():.4f} a₀ · unstrained"
        if self.display_rotation_deg:
            suffix += f" · display rotation {self.display_rotation_deg:.1f}°"
        angle_label = "reference θ" if self.manual_strain_fit is not None else "θ"
        self.plot_item.setTitle(
            f"<span style='font-size:15px;color:#17212b'>"
            f"{self.geometry.lattice} ⟨{self.geometry.axis}⟩ tilt dichromatic pattern · {angle_label} = {self.angle_deg:.2f}°"
            f"{suffix}</span>"
        )
        if self.manual_strain_fit is not None:
            self.plot_item.titleLabel.setMaximumHeight(48)
            self.plot_item.layout.setRowFixedHeight(0, 48)
        prefix = "Reference" if self.manual_strain_fit is not None else "Exact"
        exact = f"{prefix} θ = {self.angle_deg:.8f}°"
        if preset is not None:
            tail = ", ".join(
                str(int(preset.n * index)) for index in self.geometry.axis_indices
            )
            exact += f"  ·  quaternion ({preset.m}, {tail})"
        self.angle_exact_label.setText(exact)

    # ---------- angle updates ----------

    def _queue_angle_update(self, angle_deg: float) -> None:
        preset = matching_csl_preset(float(angle_deg), axis=self.geometry.axis)
        self.pending_angle = (
            preset.angle_deg if preset is not None else float(angle_deg)
        )
        self._sync_angle_controls(self.pending_angle)
        if abs(self.pending_angle - self.angle_deg) > 1.0e-12:
            if self.near_enabled:
                if self.near_search is not None:
                    self.near_search.cancel()
                self.near_debounce_timer.stop()
                self._clear_near_cell()
            self._cancel_parallel_work()
            if not self.angle_update_active:
                self._clear_lattice_selections()
            self.angle_update_active = True
        # Throttle previews rather than debouncing them: while the handle is
        # moving, the newest orientation is rendered at most once per frame.
        if not self.angle_preview_timer.isActive():
            self.angle_preview_timer.start()
        self.coincidence_timer.start()

    def _apply_pending_angle(self) -> None:
        if abs(self.pending_angle - self.angle_deg) <= 1.0e-12:
            return
        self.angle_deg = self.pending_angle
        self._regenerate_buffer(compute_coincidences=False)

    def _finish_angle_update(self) -> None:
        self.coincidence_timer.stop()
        self.angle_preview_timer.stop()
        self._apply_pending_angle()
        self._start_parallel_coincidences()
        self._update_title()
        if self.near_enabled:
            self._queue_near_search()

    def _on_preset_changed(self, combo_index: int) -> None:
        preset_index = self.preset_combo.itemData(combo_index)
        if preset_index is None:
            return
        self._cancel_parallel_work()
        preset = self.presets[int(preset_index)]
        if self.near_enabled:
            if self.near_search is not None:
                self.near_search.cancel()
            self._clear_near_cell()
        self.coincidence_timer.stop()
        self.pending_angle = preset.angle_deg
        self.angle_update_active = True
        self._clear_lattice_selections()
        self._sync_angle_controls(preset.angle_deg)
        self._finish_angle_update()

    def _sync_angle_controls(self, angle: float | None = None) -> None:
        angle = self.angle_deg if angle is None else angle
        slider_blocker = QtCore.QSignalBlocker(self.angle_slider)
        spin_blocker = QtCore.QSignalBlocker(self.angle_spin)
        combo_blocker = QtCore.QSignalBlocker(self.preset_combo)
        self.angle_slider.setValue(round(angle * 100.0))
        self.angle_spin.setValue(angle)
        preset = matching_csl_preset(angle, axis=self.geometry.axis)
        combo_index = 0
        if preset is not None:
            combo_index = self.presets.index(preset) + 1
        self.preset_combo.setCurrentIndex(combo_index)
        del slider_blocker, spin_blocker, combo_blocker

    # ---------- picking and annotations ----------

    def _set_mode(self, mode: str) -> None:
        self.interaction_mode = mode
        gb_blocker = QtCore.QSignalBlocker(self.pick_gb_button)
        vector_blocker = QtCore.QSignalBlocker(self.vector_button)
        self.pick_gb_button.setChecked(mode == "boundary")
        self.vector_button.setChecked(mode == "vector")
        with QtCore.QSignalBlocker(self.manual_pick_button):
            self.manual_pick_button.setChecked(mode == "cell")
        del gb_blocker, vector_blocker
        cursor = (
            QtCore.Qt.CursorShape.CrossCursor
            if mode != "idle"
            else QtCore.Qt.CursorShape.OpenHandCursor
        )
        self.plot_widget.viewport().setCursor(cursor)
        self._update_status()

    def _start_new_boundary(self, *_args) -> None:
        self.selected_points.clear()
        self._set_scatter(self.boundary_endpoint_item, np.empty((0, 2)))
        self.boundary_item.setData([], [])
        for item in (*self.boundary_labels, *self.side_labels):
            item.hide()
        self._set_mode("boundary")
        self._update_visible_points()

    def _start_vector_measurement(self, *_args) -> None:
        self.selected_atoms.clear()
        self._set_scatter(self.vector_endpoint_item, np.empty((0, 2)))
        self.vector_item.setData([], [])
        self.vector_arrow.hide()
        self.vector_annotation.hide()
        for item in self.vector_labels:
            item.hide()
        self._set_mode("vector")

    def _clear_lattice_selections(self) -> None:
        self._clear_manual_cell()
        self.selected_points.clear()
        self.selected_atoms.clear()
        self.boundary_item.setData([], [])
        self.vector_item.setData([], [])
        self._set_scatter(self.boundary_endpoint_item, np.empty((0, 2)))
        self._set_scatter(self.vector_endpoint_item, np.empty((0, 2)))
        self.vector_arrow.hide()
        for item in (
            *self.boundary_labels,
            *self.vector_labels,
            *self.side_labels,
            self.vector_annotation,
        ):
            item.hide()
        self._set_mode("idle")

    def _handle_view_click(self, position: np.ndarray) -> None:
        if self.interaction_mode == "idle":
            return
        if self.grain_signature != self._geometry_signature():
            self._update_status(
                "Lattice positions are updating; pick again when the new atoms appear."
            )
            return
        position = self._from_view(position)
        if self.interaction_mode == "cell":
            self._pick_manual_vertex(position)
            return
        atom, pixel_distance = self._nearest_atom(position)
        if atom is None or pixel_distance > 13.0:
            self._update_status("No atom nearby; the current selection is preserved.")
            return
        if self.interaction_mode == "boundary":
            self._pick_boundary_atom(atom)
        else:
            self._pick_vector_atom(atom)

    def _on_scene_mouse_click(self, event) -> None:
        """Route a true click to picking; drag gestures never reach this slot."""

        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        point = self.view_box.mapSceneToView(event.scenePos())
        self._handle_view_click(np.array([point.x(), point.y()], dtype=float))

    def _nearest_atom(self, position: np.ndarray) -> tuple[SelectedAtom | None, float]:
        if self.grain_signature != self._geometry_signature():
            return None, float("inf")
        x_min, x_max, y_min, y_max = self._view_range()
        scene_rect = self.view_box.sceneBoundingRect()
        pixels_per_x = scene_rect.width() / max(x_max - x_min, 1.0e-12)
        pixels_per_y = scene_rect.height() / max(y_max - y_min, 1.0e-12)
        best_atom = None
        best_distance_squared = np.inf
        for grain_index, (grain, visible_mask) in enumerate(
            zip(self.grains, self.visible_atom_masks, strict=True)
        ):
            candidate_indices = np.flatnonzero(visible_mask)
            if not len(candidate_indices):
                continue
            candidate_points = grain.positions[candidate_indices]
            deltas = self._to_view(candidate_points - position)
            distances_squared = (deltas[:, 0] * pixels_per_x) ** 2 + (
                deltas[:, 1] * pixels_per_y
            ) ** 2
            local_index = int(np.argmin(distances_squared))
            if distances_squared[local_index] < best_distance_squared:
                index = int(candidate_indices[local_index])
                best_distance_squared = float(distances_squared[local_index])
                best_atom = SelectedAtom(
                    position=grain.positions[index].copy(),
                    grain_index=grain_index,
                    layer=int(grain.layers[index]),
                    half_indices=grain.half_indices[index].copy(),
                )
        return best_atom, float(np.sqrt(best_distance_squared))

    @staticmethod
    def _atom_name(atom: SelectedAtom) -> str:
        return f"G{atom.grain_index + 1}-{layer_name(atom.layer)}"

    def _pick_boundary_atom(self, atom: SelectedAtom) -> None:
        if (
            self.selected_points
            and np.linalg.norm(atom.position - self.selected_points[0]) < 1.0e-8
        ):
            self._update_status("Choose a different second atom.")
            return
        self.selected_points.append(atom.position.copy())
        self._set_scatter(self.boundary_endpoint_item, np.asarray(self.selected_points))
        index = len(self.selected_points) - 1
        label = self.boundary_labels[index]
        label.setText(f"B{index + 1}  {self._atom_name(atom)}")
        label.setPos(*self._to_view(atom.position))
        label.show()
        if len(self.selected_points) == 2:
            self._set_mode("idle")
            self._draw_boundary()
            self._update_visible_points()
        else:
            self._update_status()

    def _pick_vector_atom(self, atom: SelectedAtom) -> None:
        if (
            self.selected_atoms
            and np.linalg.norm(atom.position - self.selected_atoms[0].position) < 1.0e-8
        ):
            self._update_status("Choose a different second atom.")
            return
        self.selected_atoms.append(atom)
        points = np.asarray([selected.position for selected in self.selected_atoms])
        self._set_scatter(self.vector_endpoint_item, points)
        index = len(self.selected_atoms) - 1
        label = self.vector_labels[index]
        label.setText(f"P{index + 1}  {self._atom_name(atom)}")
        label.setPos(*self._to_view(atom.position))
        label.show()
        if len(self.selected_atoms) == 2:
            self._set_mode("idle")
            self._draw_vector()
        self._update_status()

    def _draw_boundary(self) -> None:
        if self.selected_points:
            points = np.asarray(self.selected_points)
            self._set_scatter(self.boundary_endpoint_item, points)
            for label, point in zip(self.boundary_labels, self._to_view(points)):
                label.setPos(*point)
        if len(self.selected_points) != 2:
            return
        first, second = self._to_view(self.selected_points)
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
        if self.selected_atoms:
            points = np.array([atom.position for atom in self.selected_atoms])
            self._set_scatter(self.vector_endpoint_item, points)
            for label, point in zip(self.vector_labels, self._to_view(points)):
                label.setPos(*point)
        if len(self.selected_atoms) != 2:
            return
        first = self._to_view(self.selected_atoms[0].position)
        second = self._to_view(self.selected_atoms[1].position)
        self.vector_item.setData([first[0], second[0]], [first[1], second[1]])
        direction = second - first
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])) - 180.0)
        self.vector_arrow.setStyle(angle=angle)
        self.vector_arrow.setPos(*second)
        self.vector_arrow.show()
        normal = np.array([-direction[1], direction[0]])
        normal_length = float(np.linalg.norm(normal))
        if normal_length:
            normal /= normal_length
        _center, width, height = self._view_geometry()
        text_position = self._clamp_to_view(
            0.5 * (first + second) + normal * 0.10 * min(width, height)
        )
        self.vector_annotation.setText(self._selected_vector_readout())
        self.vector_annotation.setPos(*text_position)
        self.vector_annotation.show()

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

    @staticmethod
    def _format_lattice_vector(half_indices: np.ndarray) -> str:
        nonzero = np.abs(half_indices[half_indices != 0])
        common_factor = int(np.gcd.reduce(nonzero)) if len(nonzero) else 1
        direction = half_indices // common_factor
        if common_factor == 1:
            scale = "a₀/2"
        elif common_factor == 2:
            scale = "a₀"
        elif common_factor % 2 == 0:
            scale = f"{common_factor // 2}a₀"
        else:
            scale = f"{common_factor}a₀/2"
        indices = " ".join(str(int(component)) for component in direction)
        return f"{scale}[{indices}]"

    def _selected_vector_readout(self) -> str:
        if len(self.selected_atoms) != 2:
            return ""
        first, second = self.selected_atoms
        projected = second.position - first.position
        if first.grain_index != second.grain_index:
            projected = self._to_view(projected)
            return (
                "Cross-grain · no unique [hkl]\n"
                f"proj/a₀({self.geometry.x_label},{self.geometry.y_label}) = "
                f"({projected[0]:.4f}, {projected[1]:.4f})"
            )
        delta = second.half_indices - first.half_indices
        delta = delta + self.axial_repeat * self.geometry.axial_repeat_half_indices
        vector = self._format_lattice_vector(delta)
        magnitude = 0.5 * float(np.linalg.norm(delta))
        strained = ""
        if self.near_cell is not None:
            actual = np.sqrt(
                np.dot(projected, projected)
                + (0.5 * delta @ self.geometry.frame[:, 2]) ** 2
            )
            strained = f"\nStrained |Δr|/a₀ = {actual:.4f} (indices reference lattice)"
        return (
            f"Miller vector: {vector}\n"
            f"G{first.grain_index + 1} · axial {self.axial_repeat:+d} · ref |Δr|/a₀ = {magnitude:.4f}"
            + strained
        )

    # ---------- controls and status ----------

    def _set_region_states(self, states: tuple[bool, bool, bool, bool]) -> None:
        blockers = [QtCore.QSignalBlocker(check) for check in self.region_checks]
        for check, state in zip(self.region_checks, states, strict=True):
            check.setChecked(state)
        del blockers
        self._update_visible_points()

    def _reset_view(self, *_args) -> None:
        _center, width, height = self._view_geometry()
        self.view_box.setRange(
            xRange=(-0.5 * width, 0.5 * width),
            yRange=(-0.5 * height, 0.5 * height),
            padding=0.0,
        )
        self.view_refresh_timer.start(0)

    def _on_axial_repeat_changed(self, value: int) -> None:
        self.axial_repeat = int(value)
        self.axial_spin.setPrefix("+" if value >= 0 else "")
        self._draw_vector()
        self._update_status()

    def _update_status(self, message: str | None = None) -> None:
        if self.render_error is not None:
            message = self.render_error
        if message is None:
            if self.interaction_mode == "boundary":
                message = (
                    "GB mode · click B1"
                    if not self.selected_points
                    else "B1 selected · click B2"
                )
            elif self.interaction_mode == "cell":
                layer = (
                    f" · layer {layer_name(self.manual_vertices[0].layer)}"
                    if self.manual_vertices
                    else ""
                )
                message = f"Manual cell · pick C{len(self.manual_vertices)+1}/4 around the perimeter{layer}"
            elif self.interaction_mode == "vector":
                message = (
                    "Vector mode · click P1"
                    if not self.selected_atoms
                    else "P1 selected · click P2"
                )
            else:
                message = "Ready · choose an interaction tool"
        csl_state = (
            "updating…"
            if self.csl_updating
            else f"{sum(self.visible_coincidence_counts)} total · "
            + ", ".join(
                f"{layer_name(layer)}: {count}"
                for layer, count in enumerate(self.visible_coincidence_counts)
                if layer == self.selected_layer
                or (self.selected_layer < 0 and layer < 6)
            )
            + (
                ", …"
                if self.selected_layer < 0 and self.geometry.layer_count > 6
                else ""
            )
        )
        compute_state = (
            f"{self.worker_count} workers · {self.parallel_stage} running"
            if self.parallel_stage is not None
            else f"{self.worker_count} worker{'s' if self.worker_count != 1 else ''}"
        )
        self.status_label.setText(
            "<div style='font-weight:700;color:#17212b'>"
            f"{html.escape(message)}</div>"
            "<div style='margin-top:7px;color:#526174'>Drag to pan · wheel to zoom<br>"
            "Selections stay active while navigating</div>"
            "<div style='margin-top:7px;color:#526174'>"
            f"Visible G1 / G2: {self.visible_atom_counts[0]} / {self.visible_atom_counts[1]}<br>"
            f"Same-layer CSL: {csl_state}<br>"
            f"Compute: {compute_state}</div>"
        )

    def _choose_export_path(self) -> None:
        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export dichromatic pattern",
            "dichromatic_pattern.png",
            "PNG image (*.png)",
        )
        if filename:
            self.save(Path(filename))

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exporter = pyqtgraph.exporters.ImageExporter(self.plot_item)
        exporter.parameters()["width"] = 1800
        exporter.export(str(output_path))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.manual_count_timer.stop()
        self.manual_count_pending = None
        if self.manual_count_future is not None:
            self.manual_count_future.cancel()
        self.near_debounce_timer.stop()
        self.near_poll_timer.stop()
        if self.near_search is not None:
            self.near_search.close()
        self._cancel_parallel_work()
        executor = self.executor
        self.executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if self.local_thread_executor is not None:
            self.local_thread_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


APPLICATION_STYLESHEET = """
QMainWindow, QDockWidget, QScrollArea, QWidget {
    background: #eef2f7;
    color: #1f2937;
    font-family: "Inter", "Noto Sans", "DejaVu Sans";
    font-size: 10pt;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d8e0ea;
    border-radius: 8px;
    margin-top: 11px;
    padding: 12px 9px 9px 9px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #64748b;
    font-size: 8pt;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    min-height: 29px;
    padding: 2px 8px;
}
QPushButton:hover { background: #f1f5f9; border-color: #94a3b8; }
QPushButton:pressed, QPushButton:checked {
    background: #e0edff;
    border-color: #3b82f6;
    color: #174ea6;
    font-weight: 700;
}
QToolButton {
    background: #f8fafc;
    border: 1px solid #d5dee9;
    border-radius: 4px;
    min-height: 21px;
    padding: 0 5px;
    color: #526174;
    font-size: 8pt;
}
QToolButton:hover { background: #e0edff; border-color: #75a7ed; color: #174ea6; }
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    min-height: 27px;
    padding: 1px 7px;
}
QComboBox::drop-down { border: none; width: 24px; }
QPushButton:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {
    color: #94a3b8;
    background: #f1f5f9;
    border-color: #e2e8f0;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #dbe4ee;
    border-radius: 2px;
}
QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #3b82f6;
    width: 15px;
    height: 15px;
    margin: -6px 0;
    border-radius: 8px;
}
QLabel#mutedLabel { color: #64748b; font-size: 8pt; }
QLabel#statusCard {
    background: #ffffff;
    border: 1px solid #d8e0ea;
    border-radius: 8px;
    padding: 10px;
}
QScrollBar:vertical { width: 9px; background: transparent; }
QScrollBar::handle:vertical { background: #cbd5e1; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone FCC/BCC integer-axis tilt-GB viewer; coordinates in a0"
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="degrees; default is Sigma9 for 110, Sigma5 for 100, lowest-Sigma preset otherwise",
    )
    parser.add_argument("--lattice", choices=("FCC", "BCC"), default="FCC")
    parser.add_argument(
        "--axis",
        default="110",
        help='Tilt axis: 100, 110, 111, 112, or an integer triple, e.g. "1 -1 3"',
    )
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=3.52,
        help="reference a0 in Angstrom; display is normalized by a0",
    )
    parser.add_argument(
        "--width", type=float, default=12.0, help="base view width in a0 (not Angstrom)"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=9.0,
        help="base view height in a0 (not Angstrom)",
    )
    parser.add_argument("--marker-size", type=float, default=32.0)
    parser.add_argument(
        "--view-scale",
        type=float,
        default=1.0,
        help="initial field-size multiplier, 0.1 to 5 (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help="calculation processes; use 1 to disable multiprocessing (default: %(default)s)",
    )
    parser.add_argument("--save", type=Path, metavar="PNG")
    return parser.parse_args()


def create_application() -> QtWidgets.QApplication:
    pg.setConfigOptions(antialias=True, foreground="#334155")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Tilt GB Dichromatic Pattern")
    application.setStyle("Fusion")
    application.setStyleSheet(APPLICATION_STYLESHEET)
    return application


def main() -> None:
    multiprocessing.freeze_support()
    arguments = parse_arguments()
    parameters = PatternParameters(
        angle_deg=arguments.angle,
        lattice_constant=arguments.lattice_constant,
        width=arguments.width,
        height=arguments.height,
        marker_size=arguments.marker_size,
        view_scale=arguments.view_scale,
        lattice=arguments.lattice,
        axis=arguments.axis,
    )
    application = create_application()
    window = DichromaticPatternWindow(parameters, worker_count=arguments.workers)
    window.show()
    application.processEvents()
    if arguments.save is not None:
        window.save(arguments.save)
        print(f"Saved dichromatic pattern to {arguments.save}")
        window.close()
        return
    raise SystemExit(application.exec())


if __name__ == "__main__":
    main()
