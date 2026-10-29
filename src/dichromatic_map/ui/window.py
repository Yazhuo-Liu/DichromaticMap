"""Window composition and interactive workflows.

State, workers, controls and drawing each have a dedicated owner. The legacy
window attributes below are properties onto those owners for API compatibility.
"""

from __future__ import annotations
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from contextlib import ExitStack
import html
import os
from pathlib import Path
from dataclasses import replace
import numpy as np
from ._qt import QtCore, QtGui, QtWidgets
from ..appearance import (
    DEFAULT_GRAIN_COLORS,
    MAX_LAYER_SIZE_SCALE,
    MIN_LAYER_SIZE_SCALE,
    default_layer_size_scales,
    default_layer_symbols,
    resize_layer_size_scales,
    resize_layer_symbols,
)
from .session import SessionController
from .completion import CellCompletionDialog
from ..completion import cell_completion_candidates
from ..crystal import (
    CSLPreset,
    axial_displacement,
    bicrystal,
    projected_columns,
    layer_name,
    GeometryLimitError,
    matching_angle_preset,
    crystal_vector_coordinates,
    format_direction_components,
)
from ..cells import validate_cell_vertices, selected_region_mask
from ..displacements import (
    MAX_DISPLACEMENT_OPERATIONS, normalize_grain_displacements,
    whole_grain_displacement, set_whole_grain_displacement, set_region_displacement,
)
from ..matching import exact_csl_cell, LocalPairs, COINCIDENCE_TOLERANCE_FACTOR
from ..state import (
    PatternParameters,
    SelectedAtom,
    CellVertex,
    default_crystal_angle_deg,
    VIEW_SCALE_MIN,
    VIEW_SCALE_MAX,
    PatternState,
)
from ..strain import strain_selected_cell, selected_cell_strain_readout, tensor_readout
from ..compute import (
    ComputeSession,
    NearSearch,
    cell_count_worker,
    generate_grain_worker,
    coincidence_layers_worker,
    local_match_layers_worker,
    grain_layer_subset,
    projected_coincidence_worker,
    projected_local_match_worker,
)
from ..matching import projected_coincidence_sites, same_layer_coincidence_sites
from .controls import ControlDock
from .plot import PatternPlot

CITATION_DOI = "https://doi.org/10.48550/arXiv.2609.14673"
CITATION_TEXT = (
    "K.Q. Ding, Y.Z. Liu, Y. Zhang, L.H. Wang, X.D. Han, T. Zhu, "
    "Misfit-dislocation hierarchy governs sliding of asymmetric non-CSL grain "
    "boundaries, arXiv preprint arXiv:2609.14673 (2026)."
)


def _component_property(component, name):
    """Compatibility view onto a component's field, without copying state."""

    def get(window):
        return getattr(getattr(window, component), name)

    def set_value(window, value):
        setattr(getattr(window, component), name, value)

    return property(get, set_value)


class DichromaticPatternWindow(QtWidgets.QMainWindow):
    """Coordinate user actions, session changes and background calculations."""

    angle_deg = _component_property("state", "angle_deg")
    angle_update_active = _component_property("state", "angle_update_active")
    axial_repeat = _component_property("state", "axial_repeat")
    buffer_bounds = _component_property("state", "buffer_bounds")
    coincident_points = _component_property("state", "coincident_points")
    csl_updating = _component_property("state", "csl_updating")
    deformations = _component_property("state", "deformations")
    display_rotation_deg = _component_property("state", "display_rotation_deg")
    geometry = _component_property("state", "geometry")
    grain_signature = _component_property("state", "grain_signature")
    grains = _component_property("state", "grains")
    interaction_mode = _component_property("state", "interaction_mode")
    local_error = _component_property("state", "local_error")
    local_pairs = _component_property("state", "local_pairs")
    local_updating = _component_property("state", "local_updating")
    manual_count_error = _component_property("state", "manual_count_error")
    manual_count_key = _component_property("state", "manual_count_key")
    manual_counts = _component_property("state", "manual_counts")
    manual_local_cutoff = _component_property("state", "manual_local_cutoff")
    manual_strain_fit = _component_property("state", "manual_strain_fit")
    manual_unstrained_cutoff = _component_property("state", "manual_unstrained_cutoff")
    manual_unstrained_vertices = _component_property(
        "state", "manual_unstrained_vertices"
    )
    manual_vertices = _component_property("state", "manual_vertices")
    near_cell = _component_property("state", "near_cell")
    near_enabled = _component_property("state", "near_enabled")
    near_method = _component_property("state", "near_method")
    near_solutions = _component_property("state", "near_solutions")
    parameters = _component_property("state", "parameters")
    pending_angle = _component_property("state", "pending_angle")
    presets = _component_property("state", "presets")
    render_error = _component_property("state", "render_error")
    selected_atoms = _component_property("state", "selected_atoms")
    selected_layer = _component_property("state", "selected_layer")
    selected_points = _component_property("state", "selected_points")
    translations = _component_property("state", "translations")
    visible_atom_counts = _component_property("state", "visible_atom_counts")
    visible_atom_masks = _component_property("state", "visible_atom_masks")
    visible_coincidence_counts = _component_property(
        "state", "visible_coincidence_counts"
    )
    visible_grain_layers = _component_property("state", "visible_grain_layers")
    visible_layers = _component_property("state", "visible_layers")
    executor = _component_property("compute", "executor")
    local_thread_executor = _component_property("compute", "local_thread_executor")
    manual_count_future = _component_property("compute", "manual_count_future")
    manual_count_pending = _component_property("compute", "manual_count_pending")
    manual_count_running_key = _component_property(
        "compute", "manual_count_running_key"
    )
    near_search = _component_property("compute", "near_search")
    parallel_futures = _component_property("compute", "parallel_futures")
    parallel_generation = _component_property("compute", "parallel_generation")
    parallel_payload = _component_property("compute", "parallel_payload")
    parallel_stage = _component_property("compute", "parallel_stage")
    worker_count = _component_property("compute", "worker_count")
    worker_process_ids = _component_property("compute", "worker_process_ids")
    all_layers_button = _component_property("controls", "all_layers_button")
    angle_exact_label = _component_property("controls", "angle_exact_label")
    angle_slider = _component_property("controls", "angle_slider")
    angle_spin = _component_property("controls", "angle_spin")
    axial_image_label = _component_property("controls", "axial_image_label")
    axial_layer_checks_layout = _component_property(
        "controls", "axial_layer_checks_layout"
    )
    axial_layer_checks_widget = _component_property(
        "controls", "axial_layer_checks_widget"
    )
    axial_layer_label = _component_property("controls", "axial_layer_label")
    axial_layer_scroll = _component_property("controls", "axial_layer_scroll")
    axial_spin = _component_property("controls", "axial_spin")
    axis_combo = _component_property("controls", "axis_combo")
    axis_error = _component_property("controls", "axis_error")
    cell_check = _component_property("controls", "cell_check")
    cell_fit_button = _component_property("controls", "cell_fit_button")
    center_button = _component_property("controls", "center_button")
    custom_axis_apply = _component_property("controls", "custom_axis_apply")
    custom_axis_edit = _component_property("controls", "custom_axis_edit")
    custom_axis_row = _component_property("controls", "custom_axis_row")
    first_bicrystal_button = _component_property("controls", "first_bicrystal_button")
    full_button = _component_property("controls", "full_button")
    gb_region_widget = _component_property("controls", "gb_region_widget")
    grain_layer_checks = _component_property("controls", "grain_layer_checks")
    heading = _component_property("controls", "heading")
    interaction_section = _component_property("controls", "interaction_section")
    layer_checks = _component_property("controls", "layer_checks")
    layer_combo = _component_property("controls", "layer_combo")
    layer_info = _component_property("controls", "layer_info")
    layers_section = _component_property("controls", "layers_section")
    local_distance_spin = _component_property("controls", "local_distance_spin")
    manual_clear_button = _component_property("controls", "manual_clear_button")
    manual_fit_button = _component_property("controls", "manual_fit_button")
    manual_info = _component_property("controls", "manual_info")
    manual_pick_button = _component_property("controls", "manual_pick_button")
    manual_rotation_limit = _component_property("controls", "manual_rotation_limit")
    manual_section = _component_property("controls", "manual_section")
    manual_strain_button = _component_property("controls", "manual_strain_button")
    manual_strain_limit = _component_property("controls", "manual_strain_limit")
    manual_strain_note = _component_property("controls", "manual_strain_note")
    manual_undo_button = _component_property("controls", "manual_undo_button")
    manual_visible_check = _component_property("controls", "manual_visible_check")
    near_button = _component_property("controls", "near_button")
    near_combo = _component_property("controls", "near_combo")
    near_form = _component_property("controls", "near_form")
    near_info = _component_property("controls", "near_info")
    near_method_combo = _component_property("controls", "near_method_combo")
    near_section = _component_property("controls", "near_section")
    no_layers_button = _component_property("controls", "no_layers_button")
    orientation_layers_tabs = _component_property("controls", "orientation_layers_tabs")
    orientation_section = _component_property("controls", "orientation_section")
    performance_section = _component_property("controls", "performance_section")
    pick_gb_button = _component_property("controls", "pick_gb_button")
    preset_combo = _component_property("controls", "preset_combo")
    region_checks = _component_property("controls", "region_checks")
    rotation_slider = _component_property("controls", "rotation_slider")
    rotation_spin = _component_property("controls", "rotation_spin")
    scale_stop_buttons = _component_property("controls", "scale_stop_buttons")
    search_index_spin = _component_property("controls", "search_index_spin")
    second_bicrystal_button = _component_property("controls", "second_bicrystal_button")
    status_label = _component_property("controls", "status_label")
    strain_spin = _component_property("controls", "strain_spin")
    structure_combo = _component_property("controls", "structure_combo")
    vector_button = _component_property("controls", "vector_button")
    vector_options_widget = _component_property("controls", "vector_options_widget")
    view_performance_section = _component_property(
        "controls", "view_performance_section"
    )
    view_performance_tabs = _component_property("controls", "view_performance_tabs")
    view_scale_label = _component_property("controls", "view_scale_label")
    view_section = _component_property("controls", "view_section")
    view_slider = _component_property("controls", "view_slider")
    worker_label = _component_property("controls", "worker_label")
    worker_spin = _component_property("controls", "worker_spin")
    boundary_endpoint_item = _component_property("plot", "boundary_endpoint_item")
    boundary_item = _component_property("plot", "boundary_item")
    boundary_labels = _component_property("plot", "boundary_labels")
    coincidence_items = _component_property("plot", "coincidence_items")
    grain_layer_items = _component_property("plot", "grain_layer_items")
    legend = _component_property("plot", "legend")
    local_link_item = _component_property("plot", "local_link_item")
    local_match_item = _component_property("plot", "local_match_item")
    manual_annotation = _component_property("plot", "manual_annotation")
    manual_cell_item = _component_property("plot", "manual_cell_item")
    manual_grain_cell_items = _component_property("plot", "manual_grain_cell_items")
    manual_labels = _component_property("plot", "manual_labels")
    manual_vertex_item = _component_property("plot", "manual_vertex_item")
    near_cell_item = _component_property("plot", "near_cell_item")
    plot_item = _component_property("plot", "plot_item")
    plot_widget = _component_property("plot", "plot_widget")
    side_labels = _component_property("plot", "side_labels")
    vector_annotation = _component_property("plot", "vector_annotation")
    vector_arrow = _component_property("plot", "vector_arrow")
    vector_endpoint_item = _component_property("plot", "vector_endpoint_item")
    vector_item = _component_property("plot", "vector_item")
    vector_labels = _component_property("plot", "vector_labels")
    view_box = _component_property("plot", "view_box")
    _base_marker_diameter = _component_property("plot", "_base_marker_diameter")
    _buffer_contains_view = _component_property("plot", "_buffer_contains_view")
    _buffer_geometry = _component_property("plot", "_buffer_geometry")
    _clamp_to_view = _component_property("plot", "_clamp_to_view")
    _create_control_dock = _component_property("controls", "_create_control_dock")
    _create_layer_items = _component_property("plot", "_create_layer_items")
    _create_plot = _component_property("plot", "_create_plot")
    _draw_boundary = _component_property("plot", "_draw_boundary")
    _draw_manual_cell = _component_property("plot", "_draw_manual_cell")
    _draw_vector = _component_property("plot", "_draw_vector")
    _fit_model_corners = _component_property("plot", "_fit_model_corners")
    _from_view = _component_property("plot", "_from_view")
    _line_box_intersections = _component_property("plot", "_line_box_intersections")
    _model_view_bounds = _component_property("plot", "_model_view_bounds")
    _populate_layer_combo = _component_property("controls", "_populate_layer_combo")
    _rebuild_layer_checks = _component_property("controls", "_rebuild_layer_checks")
    _rebuild_legend = _component_property("plot", "_rebuild_legend")
    _ring_item = _component_property("plot", "_ring_item")
    _set_initial_view = _component_property("plot", "_set_initial_view")
    _set_scatter = _component_property("plot", "_set_scatter")
    _text_item = _component_property("plot", "_text_item")
    _to_view = _component_property("plot", "_to_view")
    _update_common_cell = _component_property("plot", "_update_common_cell")
    _update_local_overlay = _component_property("plot", "_update_local_overlay")
    _update_marker_sizes = _component_property("plot", "_update_marker_sizes")
    _view_geometry = _component_property("plot", "_view_geometry")
    _view_range = _component_property("plot", "_view_range")
    save = _component_property("plot", "save")

    def __init__(
        self,
        parameters: PatternParameters,
        worker_count: int | None = None,
        *,
        show_close_citation: bool = True,
    ) -> None:
        super().__init__()
        self._show_close_citation = bool(show_close_citation)
        self.state = PatternState(parameters)
        self.compute = ComputeSession()
        self.plot = PatternPlot(self)
        self.controls = ControlDock(self)
        self.session = SessionController(self)
        self._pending_display_rotation = None
        self._translation_corners = []
        self._translation_preview = None
        self._translation_vector_direction = None
        self._translation_selection = None
        self._translation_undo = [[], []]
        available_cpus = max(1, os.cpu_count() or 1)
        automatic_workers = min(4, available_cpus)
        self.compute.worker_count = int(worker_count or automatic_workers)
        self.compute.worker_count = int(
            np.clip(self.compute.worker_count, 1, available_cpus)
        )
        self._configure_worker_pool(self.compute.worker_count)

        self.setWindowTitle("Cubic Tilt GB · Dichromatic Pattern")
        self.resize(1380, 860)
        self.setMinimumSize(1040, 680)

        self.plot._create_plot()
        self.controls._create_control_dock()
        self._create_timers()
        self._create_shortcuts()
        self._update_geometry_labels()
        self.plot._set_initial_view()
        self._start_parallel_regeneration(compute_coincidences=True)
        self.view_refresh_timer.stop()
        self._set_mode("idle")

    def _create_timers(self) -> None:
        self.display_rotation_timer = QtCore.QTimer(self)
        self.display_rotation_timer.setSingleShot(True)
        self.display_rotation_timer.setInterval(16)
        self.display_rotation_timer.timeout.connect(self._flush_display_rotation)
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
        self.view_counts_timer = QtCore.QTimer(self)
        self.view_counts_timer.setSingleShot(True)
        self.view_counts_timer.setInterval(16)
        self.view_counts_timer.timeout.connect(self._refresh_view_display)
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
            ("Escape", self._cancel_interaction),
        )
        self.shortcuts: list[QtGui.QShortcut] = []
        for key, callback in shortcuts:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def _choose_grain_color(self, grain):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self.state.grain_colors[grain]), self, f"Choose G{grain + 1} color"
        )
        if color.isValid() and color.name() != self.state.grain_colors[grain]:
            self.state.grain_colors[grain] = color.name()
            self._refresh_appearance()

    def _on_layer_symbol_changed(self, layer, index):
        combo = self.controls.layer_symbol_combos[layer]
        symbol = combo.itemData(index)
        if symbol == self.state.layer_symbols[layer]:
            return
        if symbol is None or symbol in self.state.layer_symbols:
            with QtCore.QSignalBlocker(combo):
                combo.setCurrentIndex(combo.findData(self.state.layer_symbols[layer]))
            return
        self.state.layer_symbols[layer] = symbol
        self._refresh_appearance()

    def _on_layer_size_changed(self, layer, percentage):
        scale = float(percentage) / 100.0
        if not MIN_LAYER_SIZE_SCALE <= scale <= MAX_LAYER_SIZE_SCALE:
            with QtCore.QSignalBlocker(self.controls.layer_size_spins[layer]):
                self.controls.layer_size_spins[layer].setValue(100 * self.state.layer_size_scales[layer])
            return
        if scale == self.state.layer_size_scales[layer]:
            return
        self.state.layer_size_scales[layer] = scale
        self.plot.refresh_marker_sizes()

    def _reset_appearance(self, *_args):
        self.state.grain_colors = list(DEFAULT_GRAIN_COLORS)
        self.state.layer_symbols = default_layer_symbols(self.state.layer_count)
        self.state.layer_size_scales = default_layer_size_scales(self.state.layer_count)
        self._refresh_appearance()

    def _refresh_appearance(self):
        self.controls._refresh_appearance_controls()
        self.plot.refresh_appearance()
        if self.state.manual_counts is not None:
            self._show_manual_counts()

    def _on_reference_axes_toggled(self, visible):
        self.state.show_reference_axes = bool(visible)
        self.plot._update_reference_axes()

    def _has_displacements(self):
        return any(self.state.grain_displacements)

    def _custom_position_base(self):
        return self.state.near_cell is None and (
            any(not np.array_equal(f, np.eye(2)) for f in self.state.deformations)
            or np.any(self.state.translations != 0)
        )

    def _displacement_signature(self):
        return tuple(tuple((tuple(op["delta"]),
                            None if op["bounds"] is None else tuple(op["bounds"]))
                           for op in operations)
                     for operations in self.state.grain_displacements)

    def _sync_translation_controls(self, message=None):
        if not hasattr(self.controls, "translation_status"):
            return
        controls = self.controls
        grain = controls.translation_grain_combo.currentData()
        histories = self.state.grain_displacements
        # Validation reports the operation budget. Even a full history can
        # accept an unchanged target or remove the active region's offset.
        controls.translation_whole_button.setEnabled(True)
        controls.translation_region_button.setEnabled(self._translation_selection is not None)
        controls.translation_clear_button.setEnabled(bool(self._translation_corners))
        controls.translation_vector_button.setEnabled(len(self.state.selected_atoms) == 2)
        controls.translation_undo_button.setEnabled(bool(histories[grain] or self._translation_undo[grain]))
        controls.translation_reset_button.setEnabled(any(histories))
        counts = f"G1: {len(histories[0])} moves · G2: {len(histories[1])} moves"
        if message is None and len(self._translation_corners) == 2:
            low, high = np.minimum(*self._translation_corners), np.maximum(*self._translation_corners)
            message = f"Rectangle x=[{low[0]:.4g}, {high[0]:.4g}], y=[{low[1]:.4g}, {high[1]:.4g}] a₀."
        controls.translation_status.setText(counts + ("\n" + message if message else ""))

    def _clear_translation_selection(self, *, sync_offset=True):
        self._translation_corners = []
        self._translation_preview = None
        self._translation_selection = None
        if sync_offset and hasattr(self.controls, "translation_dx_spin"):
            grain = self.controls.translation_grain_combo.currentData()
            self._set_translation_offset(whole_grain_displacement(self.state.grain_displacements[grain]), keep_direction=True)
        self.plot._draw_translation_region()
        self._sync_translation_controls()

    def _cancel_interaction(self):
        self._clear_translation_selection()
        self._set_mode("idle")

    def _on_translation_components_changed(self, *_args):
        self._translation_vector_direction = None
        self.controls.translation_vector_length_spin.setEnabled(False)

    def _set_translation_offset(self, offset, *, keep_direction=False):
        if keep_direction and self._translation_vector_direction is not None:
            direction = self._translation_vector_direction
            keep_direction = np.linalg.norm(np.asarray(offset) - direction * np.dot(offset, direction)) <= 1e-8
        if not keep_direction:
            self._on_translation_components_changed()
        for spin, value in zip((self.controls.translation_dx_spin, self.controls.translation_dy_spin), offset):
            with QtCore.QSignalBlocker(spin):
                spin.setValue(float(value))
        if keep_direction and self._translation_vector_direction is not None:
            with QtCore.QSignalBlocker(self.controls.translation_vector_length_spin):
                self.controls.translation_vector_length_spin.setValue(float(
                    np.dot(offset, self._translation_vector_direction)
                ))

    def _restore_translation_selection(self, selection):
        self._translation_selection = deepcopy(selection)
        if selection is None:
            self._translation_corners = []
            grain = self.controls.translation_grain_combo.currentData()
            target = whole_grain_displacement(self.state.grain_displacements[grain])
        else:
            xmin, xmax, ymin, ymax = selection["bounds"]
            offset = np.asarray(selection["target"])
            self._translation_corners = [np.array([xmin, ymin]) + offset, np.array([xmax, ymax]) + offset]
            target = selection["target"]
        self._set_translation_offset(target, keep_direction=True)
        self._translation_preview = None
        self.plot._draw_translation_region()

    def _use_translation_vector(self):
        if len(self.state.selected_atoms) != 2:
            self._sync_translation_controls("Measure P1 and P2 first using Measure vector (V).")
            return
        first, second = self.state.selected_atoms
        displacement = np.asarray(second.position - first.position, dtype=float)
        length = float(np.hypot(*displacement))
        if not np.isfinite(length) or length <= 1e-12:
            self._sync_translation_controls("The measured vector needs a nonzero planar length.")
            return
        if length > self.controls.translation_vector_length_spin.maximum():
            self._sync_translation_controls("The measured vector exceeds the supported shift length.")
            return
        self._translation_vector_direction = displacement / length
        with QtCore.QSignalBlocker(self.controls.translation_vector_length_spin):
            self.controls.translation_vector_length_spin.setValue(length)
        self.controls.translation_vector_length_spin.setEnabled(True)
        self._on_translation_vector_length(self.controls.translation_vector_length_spin.value())
        self._sync_translation_controls("Copied P1 → P2 in model axes. Adjust its planar length, then set the offset.")

    def _on_translation_vector_length(self, length):
        if self._translation_vector_direction is None:
            return
        offset = self._translation_vector_direction * length
        for spin, value in zip((self.controls.translation_dx_spin, self.controls.translation_dy_spin), offset):
            with QtCore.QSignalBlocker(spin):
                spin.setValue(float(value))

    def _on_translation_grain_changed(self, *_args):
        self._clear_translation_selection()
        grain = self.controls.translation_grain_combo.currentData()
        self._set_translation_offset(whole_grain_displacement(self.state.grain_displacements[grain]))
        if self.state.interaction_mode == "translate":
            self._set_mode("idle")

    def _start_translation_selection(self, *_args):
        self._flush_display_rotation()
        self._clear_translation_selection(sync_offset=False)
        self._set_mode("translate")
        self._sync_translation_controls("Drag a rectangle, or click two opposite corners. Esc cancels; all axial layers are included.")

    def _on_translation_hover(self, point):
        if self.state.interaction_mode != "translate" or len(self._translation_corners) != 1:
            return
        if not self.plot.view_box.sceneBoundingRect().contains(point):
            return
        position = self.plot.view_box.mapSceneToView(point)
        self._preview_translation_corner(self.plot._from_view(np.array([position.x(), position.y()])))

    def _preview_translation_corner(self, position):
        if len(self._translation_corners) == 1:
            self._translation_preview = np.asarray(position, dtype=float).copy()
            self.plot._draw_translation_region()

    def _pick_translation_corner(self, position):
        if len(self._translation_corners) == 1:
            if np.any(np.abs(position - self._translation_corners[0]) < 1e-8):
                self._sync_translation_controls("Choose an opposite corner with nonzero width and height.")
                return
        self._translation_corners.append(np.asarray(position, dtype=float).copy())
        self._translation_preview = None
        self.plot._draw_translation_region()
        if len(self._translation_corners) == 2:
            low, high = np.minimum(*self._translation_corners), np.maximum(*self._translation_corners)
            grain = self.controls.translation_grain_combo.currentData()
            self._translation_selection = {
                "grain": grain,
                "base_history": deepcopy(self.state.grain_displacements[grain]),
                "bounds": [float(low[0]), float(high[0]), float(low[1]), float(high[1])],
                "target": [0.0, 0.0],
            }
            self._set_translation_offset([0.0, 0.0], keep_direction=True)
            self._set_mode("idle")
        self._sync_translation_controls(
            "First corner selected; click the opposite corner." if len(self._translation_corners) == 1 else None
        )

    def _apply_translation(self, local=False):
        try:
            self.plot._await_current_scene()
            if local and self._translation_selection is None:
                raise ValueError("Select two opposite rectangle corners first.")
            target = [self.controls.translation_dx_spin.value(), self.controls.translation_dy_spin.value()]
            grain = self.controls.translation_grain_combo.currentData()
            histories = normalize_grain_displacements(self.state.grain_displacements)
            selection = deepcopy(self._translation_selection) if local else None
            if local:
                histories[grain] = set_region_displacement(selection["base_history"], target, selection["bounds"])
                selection["target"] = target
            else:
                histories[grain] = set_whole_grain_displacement(histories[grain], target)
            if histories == self.state.grain_displacements:
                if not local:
                    self._clear_translation_selection()
                self._sync_translation_controls("This absolute offset is already applied; no additional shift.")
                return
            self._translation_undo[grain].append({
                "operations": deepcopy(self.state.grain_displacements[grain]),
                "selection": deepcopy(self._translation_selection),
            })
            # Histories already have an operation budget; cap in-memory edit
            # snapshots as well for arbitrarily many region adjustments.
            self._translation_undo[grain] = self._translation_undo[grain][-MAX_DISPLACEMENT_OPERATIONS:]
            self._replace_displacements(histories, f"Set absolute offset of {'rectangle in' if local else 'whole'} G{grain + 1}; all axial layers affected.", selection)
        except (ValueError, RuntimeError, TimeoutError) as error:
            self._sync_translation_controls(str(error))

    def _undo_translation(self, *_args):
        histories = normalize_grain_displacements(self.state.grain_displacements)
        grain = self.controls.translation_grain_combo.currentData()
        selection = None
        if self._translation_undo[grain]:
            previous = self._translation_undo[grain].pop()
            histories[grain] = previous["operations"]
            selection = previous["selection"]
        elif histories[grain]:
            histories[grain].pop()
        else:
            return
        self._replace_displacements(histories, f"Undid last move in G{grain + 1}.", selection)
        self._set_translation_offset(selection["target"] if selection else whole_grain_displacement(histories[grain]))

    def _reset_translations(self, *_args):
        self._translation_undo = [[], []]
        if self._has_displacements():
            self._replace_displacements([[], []], "All manual moves reset; existing bulk deformation is retained.")
        self._clear_translation_selection()
        self._set_translation_offset([0, 0])

    def _replace_displacements(self, histories, message, selection=None):
        self.near_debounce_timer.stop()
        self.near_poll_timer.stop()
        if self.compute.near_search is not None:
            self.compute.near_search.cancel()
        self._cancel_parallel_work()
        self.state.grain_displacements = histories
        # F and uniform translations are the physical base of these moves;
        # remove stale fit claims without changing that base structure.
        self.state.near_cell = None
        self.state.near_solutions = []
        self.state.manual_strain_fit = None
        self.state.manual_unstrained_vertices = None
        self.state.manual_unstrained_cutoff = None
        self.state.near_method = "local"
        with QtCore.QSignalBlocker(self.controls.near_method_combo):
            self.controls.near_method_combo.setCurrentIndex(self.controls.near_method_combo.findData("local"))
        with QtCore.QSignalBlocker(self.controls.near_combo):
            self.controls.near_combo.clear()
        boundary = [point.copy() for point in self.state.selected_points]
        self._clear_lattice_selections(reset_translation_undo=False)
        self._restore_translation_selection(selection)
        # The GB is a fixed spatial clipping reference, not a material region
        # that follows a moved grain. Retain its position and side filters.
        self.state.selected_points = boundary
        for index, label in enumerate(self.plot.boundary_labels[:len(boundary)]):
            label.setText(f"B{index + 1} · fixed reference")
            label.show()
        self.plot._draw_boundary()
        self._sync_context_controls()
        self._sync_near_controls()
        self._sync_translation_controls(message)
        self.controls.near_info.setPlainText("Manual positions updated. Local matching uses the moved atoms; strain fitting requires resetting moves and any retained bulk deformation.")
        self.state.buffer_bounds = None
        self._start_parallel_regeneration(True)
        self._update_title()
        self._update_status(message)

    def _on_display_rotation(self, angle):
        self._pending_display_rotation = float(angle)
        with (
            QtCore.QSignalBlocker(self.controls.rotation_spin),
            QtCore.QSignalBlocker(self.controls.rotation_slider),
        ):
            self.controls.rotation_spin.setValue(angle)
            self.controls.rotation_slider.setValue(round(angle * 10))
        if not self.display_rotation_timer.isActive():
            self.display_rotation_timer.start()

    def _flush_display_rotation(self):
        self.display_rotation_timer.stop()
        angle, self._pending_display_rotation = self._pending_display_rotation, None
        if angle is not None:
            self._apply_display_rotation(angle)

    def _apply_display_rotation(self, angle):
        angle = float(angle)
        if abs(angle - self.state.display_rotation_deg) < 1e-10:
            return
        center, width, height = self.plot._view_geometry()
        model_center = self.plot._from_view(center)
        self.state.display_rotation_deg = angle
        with (
            QtCore.QSignalBlocker(self.controls.rotation_spin),
            QtCore.QSignalBlocker(self.controls.rotation_slider),
        ):
            self.controls.rotation_spin.setValue(angle)
            self.controls.rotation_slider.setValue(round(angle * 10))
        center = self.plot._to_view(model_center)
        self.plot.view_box.setRange(
            xRange=(center[0] - width / 2, center[0] + width / 2),
            yRange=(center[1] - height / 2, center[1] + height / 2),
            padding=0,
        )
        self._update_visible_points()
        self.plot._update_common_cell()
        self.plot._draw_boundary()
        self.plot._draw_vector()
        self.plot._draw_manual_cell()
        self.plot._draw_translation_region()
        self._update_title()
        if not self.plot._buffer_contains_view():
            self.view_refresh_timer.start()

    def _start_manual_cell(self, *_args):
        if self.state.manual_strain_fit is not None:
            self._manual_pick_warning(
                "Restore the local structure before editing cell vertices."
            )
            return
        if self.state.interaction_mode == "cell":
            self._set_mode("idle")
            return
        if len(self.state.manual_vertices) == 4:
            self._clear_manual_cell()
        self._set_mode("cell")

    def _clear_manual_cell(self, *_args):
        self.state.manual_vertices = []
        self.state.manual_counts = None
        self.state.manual_count_error = None
        self.state.manual_count_key = None
        self.compute.manual_count_pending = None
        self.state.manual_local_cutoff = None
        if (
            self.compute.manual_count_future is not None
            and self.compute.manual_count_future.cancel()
        ):
            self.compute.manual_count_future = None
        self.controls.manual_info.setPlainText(
            "No manual cell. Counts use the picked layer and each grain's own atom vertices; periodicity is not assumed."
        )
        self.plot._draw_manual_cell()
        if self.state.interaction_mode == "cell":
            self._set_mode("idle")

    def _undo_manual_vertex(self, *_args):
        if self.state.manual_strain_fit is not None:
            return
        if not self.state.manual_vertices:
            return
        self.state.manual_vertices.pop()
        self.state.manual_counts = None
        self.state.manual_count_error = None
        self.state.manual_count_key = None
        self.compute.manual_count_pending = None
        self.controls.manual_info.setPlainText(
            "Continue picking vertices in perimeter order."
        )
        self.plot._draw_manual_cell()
        self._set_mode("cell")

    def _invalidate_manual_local_source(self):
        # Picked vertices own their original atom endpoints. A new cutoff
        # refreshes available markers, not the existing selection or its mode.
        if not self.local_active and any(
            vertex.source != "CSL" for vertex in self.state.manual_vertices
        ):
            self._clear_manual_cell()

    def _sync_manual_strain_controls(self):
        active = self.state.manual_strain_fit is not None
        self.controls.manual_strain_details.setVisible(active)
        if not active:
            self.controls.manual_strain_details.clear()
        with QtCore.QSignalBlocker(self.controls.near_method_combo):
            self.controls.near_method_combo.setItemText(
                0,
                (
                    "Local cell · bulk strain applied"
                    if active
                    else "Local matching · no bulk strain"
                ),
            )
        custom = self._has_displacements() or self._custom_position_base()
        ready = (
            not custom
            and self.local_active
            and len(self.state.manual_vertices) == 4
            and any(vertex.source != "CSL" for vertex in self.state.manual_vertices)
        )
        completable = self._manual_completion_available()
        self.controls.manual_complete_button.setEnabled(completable)
        self.controls.manual_complete_button.setText(
            "Complete parallelogram…" if len(self.state.manual_vertices) == 3
            else "Complete by symmetry…"
        )
        self.controls.manual_strain_button.setEnabled(not custom and (active or ready))
        self.controls.manual_strain_limit.setEnabled((ready or completable) and not active)
        self.controls.manual_rotation_limit.setEnabled((ready or completable) and not active)
        self.controls.manual_strain_button.setText(
            "Restore original local structure"
            if active
            else "Apply bulk strain to selected cell"
        )
        for button in (
            self.controls.manual_pick_button,
            self.controls.manual_clear_button,
            self.controls.manual_undo_button,
        ):
            button.setEnabled(not active)
        if custom:
            self.controls.manual_strain_note.setText(
                "Strain fitting and symmetry completion are unavailable for manually moved atoms. "
                "Reset moves first; change angle/lattice to reset retained bulk deformation."
            )
        elif not active and len(self.state.manual_vertices) < 4:
            self.controls.manual_strain_note.setText(
                "Select a local near-CSL cell first. Both grains share the deformation; strain and small rotations have separate limits."
            )

    def _toggle_manual_strain(self, *_args):
        if self._has_displacements() or self._custom_position_base():
            return
        if self.state.manual_strain_fit is not None:
            vertices = self.state.manual_unstrained_vertices
            cutoff = self.state.manual_unstrained_cutoff
            self._cancel_parallel_work()
            self._clear_near_cell()
            self._sync_near_controls()
            self.state.manual_vertices = vertices
            self.state.manual_local_cutoff = cutoff
            self.plot._draw_manual_cell()
            self._start_parallel_regeneration(True)
            self._queue_manual_count()
            self.controls.manual_strain_note.setText(
                "Original local structure restored; no bulk strain."
            )
            return
        if not (
            self.local_active
            and len(self.state.manual_vertices) == 4
            and any(vertex.source != "CSL" for vertex in self.state.manual_vertices)
        ):
            self._manual_pick_warning(
                "First select four same-layer vertices using Local matching."
            )
            return
        try:
            fit = strain_selected_cell(
                self._manual_grain_polygons(),
                self.state.angle_deg,
                self.controls.manual_strain_limit.value(),
                self.state.crystal.lattice,
                self.state.crystal.axis,
                self.state.manual_vertices[0].layers,
                self.controls.manual_rotation_limit.value(),
                scale=self.state.crystal.scale,
            )
        except ValueError as error:
            self.controls.manual_strain_note.setText(str(error))
            self._update_status(str(error))
            return
        vertices = self.state.manual_vertices
        cutoff = self.state.manual_local_cutoff
        self.near_debounce_timer.stop()
        self.near_poll_timer.stop()
        if self.compute.near_search is not None:
            self.compute.near_search.cancel()
        self._cancel_parallel_work()
        self._clear_lattice_selections()
        self.state.manual_unstrained_vertices = vertices
        self.state.manual_unstrained_cutoff = cutoff
        self.state.manual_strain_fit = fit
        # Generate once per successful application, not while repainting the
        # view: panning and display rotation must preserve the reading position.
        self.controls.manual_strain_details.setPlainText(
            selected_cell_strain_readout(
                fit,
                self.state.angle_deg,
                strain_limit_percent=self.controls.manual_strain_limit.value(),
                rotation_limit_deg=self.controls.manual_rotation_limit.value(),
            )
        )
        self.state.near_cell = fit.cell
        self.state.deformations = (fit.cell.f1, fit.cell.f2)
        self.state.translations = fit.translations.copy()
        self.state.manual_vertices = [
            CellVertex(pair.mean(axis=0), vertex.layer, "CSL", pair.copy(), vertex.second_layer)
            for vertex, pair in zip(vertices, fit.vertices.transpose(1, 0, 2))
        ]
        self._clear_local_pairs()
        self._sync_near_controls()
        self.plot._draw_manual_cell()
        self.controls.manual_strain_note.setText(
            f"Applied: max principal strain {100*fit.cell.max_strain:.6f}%; "
            f"rotations G1 {fit.rotations_deg[0]:+.6f}°, G2 {fit.rotations_deg[1]:+.6f}°; "
            f"four-pair residual {fit.residual:.2e} a₀. "
            + ("Full-lattice CSL is recomputed, layer by layer." if self.state.crystal.homophase
               else "Coincident columns are recomputed across layers.")
        )
        self.controls.near_info.setPlainText(
            "SELECTED-CELL BULK STRAIN (not stress-free)\n"
            "All four original pairs aligned. Polar decomposition F = R U:\n"
            + "\n".join(
                f"G{g+1}: rotation {fit.rotations_deg[g]:+.8f}°; principal strains "
                + np.array2string(100 * (fit.stretches[g] - 1), precision=8)
                + " %"
                for g in (0, 1)
            )
            + f"\nReference θ = {self.state.angle_deg:.8f}°; polar-frame θ = {self.state.angle_deg+fit.rotations_deg[0]-fit.rotations_deg[1]:.8f}°.\n"
            "Uniform shifts align the two cell centroids at their original midpoint.\n"
            "t1/t2 / a₀ (analysis x,y):\n"
            + np.array2string(fit.translations, precision=8)
            + "\n"
            + tensor_readout(fit.cell, self.state.angle_deg, selected=True)
        )
        self._start_parallel_regeneration(True)
        self._queue_manual_count()
        self._update_title()

    def _common_site_candidates(self):
        """All displayed markers: reject wrong-layer clicks *after* hit-testing."""
        if self.state.grain_signature != self._geometry_signature():
            return []
        candidates = []
        states = self._region_states()
        if not self.state.csl_updating:
            for layer, (points, partners) in enumerate(zip(
                self.state.coincident_points, self.state.coincident_second_layers
            )):
                if layer not in self.state.visible_grain_layers[0]:
                    continue
                keep = np.isin(partners, tuple(self.state.visible_grain_layers[1]))
                if len(self.state.selected_points) == 2:
                    for grain in (0, 1):
                        keep &= selected_region_mask(
                            points,
                            *self.state.selected_points,
                            states[2 * grain],
                            states[2 * grain + 1],
                        )
                candidates.extend(
                    CellVertex(point.copy(), layer, "CSL", None, self._second_layer(partner))
                    for point, partner in zip(points[keep], partners[keep])
                )
        if self.local_active and not self.state.local_updating:
            keep = self._local_pair_mask()
            pairs = self.state.local_pairs
            candidates.extend(
                CellVertex(point.copy(), int(layer), "local", np.stack((first, second)),
                           self._second_layer(second_layer))
                for point, layer, second_layer, first, second in zip(
                    pairs.midpoints[keep], pairs.layers[keep], pairs.second_layers[keep],
                    pairs.first[keep], pairs.second[keep],
                )
            )
        return candidates

    def _second_layer(self, layer):
        """A vertex's G2 layer, stored only when grain layers are unrelated."""
        return None if self.state.crystal.homophase else int(layer)

    def _manual_completion_available(self):
        # Completion operates on original lattice coordinates. It does not
        # depend on the Near-CSL toggle or the selected matching method.
        return (
            self.state.manual_strain_fit is None
            and self.state.near_cell is None
            and not self._has_displacements()
            and not self._custom_position_base()
            and len(self.state.manual_vertices) in (2, 3)
        )

    def _manual_completion_key(self):
        return (
            self._geometry_signature(),
            tuple((vertex.layers, vertex.source) for vertex in self.state.manual_vertices),
            tuple(self._manual_grain_polygons().ravel()),
            self.controls.manual_strain_limit.value(),
            self.controls.manual_rotation_limit.value(),
        )

    def _complete_manual_cell(self, *_args):
        if not self._manual_completion_available():
            return
        key = self._manual_completion_key()
        try:
            candidates = cell_completion_candidates(
                self._manual_grain_polygons(), self.state.angle_deg,
                lattice=self.state.crystal.lattice, axis=self.state.crystal.axis,
                layer=self.state.manual_vertices[0].layers,
                percent=self.controls.manual_strain_limit.value(),
                max_rotation_deg=self.controls.manual_rotation_limit.value(),
                scale=self.state.crystal.scale,
            )
        except ValueError as error:
            self._manual_pick_warning(str(error))
            return
        if not candidates:
            self._manual_pick_warning(
                "No non-degenerate symmetry cell found. Pick a third vertex for parallelogram completion."
                if len(self.state.manual_vertices) == 2 else
                "These three paired vertices cannot form compatible non-degenerate parallelograms."
            )
            return
        dialog = CellCompletionDialog(
            candidates, len(self.state.manual_vertices), self.state.grain_colors,
            self.state.display_rotation_deg, parent=self,
        )
        dialog.accepted.connect(lambda: self._accept_manual_completion(dialog, key))
        dialog.open()

    def _accept_manual_completion(self, dialog, key):
        if not self._manual_completion_available() or key != self._manual_completion_key():
            self._manual_pick_warning("Selection changed; generate a new completion preview.")
            return
        candidate = dialog.selected_candidate
        if candidate.fit is None:
            return
        count = len(self.state.manual_vertices)
        first = self.state.manual_vertices[0]
        generated = []
        for pair in candidate.vertices[:, count:].transpose(1, 0, 2):
            source = ("CSL" if np.linalg.norm(pair[0] - pair[1]) <= COINCIDENCE_TOLERANCE_FACTOR
                      else candidate.source)
            generated.append(CellVertex(pair.mean(axis=0), first.layer, source, pair.copy(),
                                        first.second_layer))
        self.state.manual_vertices = self.state.manual_vertices + generated
        self.plot._draw_manual_cell()
        self._set_mode("idle")
        self._queue_manual_count()
        if all(vertex.source == "CSL" for vertex in self.state.manual_vertices):
            self.controls.manual_strain_note.setText(
                "Exact CSL cell completed; all four pairs already coincide. No strain is needed."
            )
        else:
            self.controls.manual_strain_note.setText(
                f"Cell completed: max principal strain {100 * candidate.fit.cell.max_strain:.6f}%; "
                f"rotations G1 {candidate.fit.rotations_deg[0]:+.6f}°, G2 {candidate.fit.rotations_deg[1]:+.6f}°. "
                "Original atom positions retained. Use Apply bulk strain to align the cell."
            )

    def _cell_layer_label(self, layer):
        symbol = self.state.layer_symbols[layer]
        glyph = {
            "o": "○",
            "d": "◇",
            "t": "△",
            "s": "□",
            "p": "⬠",
            "h": "⬡",
            "star": "☆",
            "+": "+",
            "x": "×",
            "t1": "▽",
            "t2": "▷",
            "t3": "◁",
        }.get(symbol, symbol.removeprefix("number:"))
        return f"{glyph} layer"

    def _vertex_layers_label(self, layers, short=False):
        """One shared layer, or each grain's own layer for different grains."""
        if self.state.crystal.homophase:
            label = self._cell_layer_label(layers[0])
            return label.split()[0] if short else label
        glyphs = [self._cell_layer_label(layer).split()[0] for layer in layers]
        return f"G1 {glyphs[0]}/G2 {glyphs[1]}" + ("" if short else " layers")

    def _resolve_cell_vertex(self, candidate):
        if candidate.grain_positions is not None:
            return candidate
        # Exact CSL stores marker centers; resolve its two real atoms once at
        # selection time, not later from a possibly panned-away drawing buffer.
        endpoints = []
        for grain, layer in zip(self.state.grains, candidate.layers):
            points = grain.positions[grain.layers == layer]
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
        if not self.state.manual_vertices:
            return np.empty((2, 0, 2))
        return np.stack(
            [vertex.grain_positions for vertex in self.state.manual_vertices], axis=1
        )

    def _manual_pick_warning(self, message):
        # Keep rejection feedback beside the picking controls, even if the
        # general status label is outside the currently scrolled panel.
        self.controls.manual_info.setPlainText(message)
        self._update_status(message)

    def _pick_manual_vertex(self, position):
        candidates = self._common_site_candidates()
        if not candidates:
            self._update_status(
                "No visible common sites; enable Near-CSL or choose a CSL angle."
            )
            return
        positions = np.array([candidate.position for candidate in candidates])
        pixel_size = np.asarray(self.plot.view_box.viewPixelSize())
        delta = self.plot._to_view(positions - position) / np.maximum(pixel_size, 1e-12)
        distance = np.linalg.norm(delta, axis=1)
        index = int(np.argmin(distance))
        if distance[index] > max(14.0, self.plot._base_marker_diameter()):
            self._update_status(
                "Click a gold CSL or purple near-pair marker in the first vertex's layer."
            )
            return
        candidate = candidates[index]
        hit_layers = {
            candidates[i].layers
            for i in np.flatnonzero(np.abs(distance - distance[index]) < 1e-6)
        }
        if len(hit_layers) > 1:
            self._manual_pick_warning(
                "Overlapping markers from different layers; isolate one layer in the Layers tab, then pick again. Point not selected."
            )
            return
        if (
            self.state.manual_vertices
            and candidate.layers != self.state.manual_vertices[0].layers
        ):
            self._manual_pick_warning(
                f"Wrong layer/symbol: expected {self._vertex_layers_label(self.state.manual_vertices[0].layers)}, "
                f"clicked {self._vertex_layers_label(candidate.layers)}. Point not selected."
            )
            return
        if any(
            np.linalg.norm(candidate.position - v.position) < 1e-8
            for v in self.state.manual_vertices
        ):
            self._update_status("Choose a different common-site vertex.")
            return
        try:
            candidate = self._resolve_cell_vertex(candidate)
        except ValueError as error:
            self._update_status(str(error))
            return
        trial = self.state.manual_vertices + [candidate]
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
        self.state.manual_vertices = trial
        if candidate.source == "local":
            self.state.manual_local_cutoff = self.controls.local_distance_spin.value()
        self.plot._draw_manual_cell()
        if len(trial) == 4:
            self._set_mode("idle")
            self._queue_manual_count()
        else:
            self.controls.manual_info.setPlainText(
                f"Selected {len(trial)}/4 vertices in {self._vertex_layers_label(candidate.layers)}. "
                "Continue around the perimeter in the same layer."
            )
            self._update_status()

    def _queue_manual_count(self, *_args):
        if len(self.state.manual_vertices) != 4:
            return
        points = self._manual_grain_polygons()
        filtered = self.controls.manual_visible_check.isChecked()
        boundary = (
            np.array(self.state.selected_points)
            if filtered and len(self.state.selected_points) == 2
            else None
        )
        states = self._region_states() if filtered else (True, True, True, True)
        layer = self.state.manual_vertices[0].layers
        key = (
            tuple(points.ravel()),
            self._geometry_signature(),
            filtered,
            layer,
            states,
            tuple(boundary.ravel()) if boundary is not None else None,
        )
        if key == self.state.manual_count_key:
            return
        self.state.manual_count_key = key
        self.state.manual_counts = None
        self.state.manual_count_error = None
        self.compute.manual_count_pending = (
            points,
            self.state.angle_deg,
            tuple(f.copy() for f in self.state.deformations),
            self.state.crystal.lattice,
            self.state.crystal.axis,
            boundary,
            states,
            layer,
            self.state.translations.copy(),
            normalize_grain_displacements(self.state.grain_displacements),
            self.state.crystal.scale,
        )
        if (
            self.compute.manual_count_future is not None
            and self.compute.manual_count_future.cancel()
        ):
            self.compute.manual_count_future = None
        self.controls.manual_info.setPlainText(
            "Counting the entire cell in the background (independent of viewport)…"
        )
        self.plot._draw_manual_cell()
        self.manual_count_timer.start()
        self._poll_manual_count()

    def _poll_manual_count(self):
        if self.compute.manual_count_future is not None:
            if not self.compute.manual_count_future.done():
                return
            try:
                pid, counts = self.compute.manual_count_future.result()
                self.compute.worker_process_ids.add(pid)
                if self.compute.manual_count_running_key == self.state.manual_count_key:
                    self.state.manual_counts = counts
                    self._show_manual_counts()
            except Exception as error:
                if self.compute.manual_count_running_key == self.state.manual_count_key:
                    self.state.manual_count_error = str(error)
                    self.controls.manual_info.setPlainText(
                        f"Cell count unavailable: {error}\nNo partial count is reported."
                    )
                    self.plot.manual_annotation.setText(
                        "Manual cell · count unavailable"
                    )
            self.compute.manual_count_future = None
        if self.compute.manual_count_pending is not None:
            args, self.compute.manual_count_pending = (
                self.compute.manual_count_pending,
                None,
            )
            executor = self.compute.background_executor()
            self.compute.manual_count_running_key = self.state.manual_count_key
            try:
                self.compute.manual_count_future = executor.submit(
                    cell_count_worker, *args
                )
            except Exception as error:
                self.state.manual_count_error = str(error)
                self.controls.manual_info.setPlainText(
                    f"Cell count unavailable: {error}"
                )
                self.plot._draw_manual_cell()
        if self.compute.manual_count_future is None:
            self.manual_count_timer.stop()

    def _show_manual_counts(self):
        counts = self.state.manual_counts
        layers = self.state.manual_vertices[0].layers
        custom = self._has_displacements() or self._custom_position_base()
        lines = [
            (
                "Selected-cell bulk strain; common translations verified."
                if self.state.manual_strain_fit is not None
                else "Manual region; periodicity not verified."
            ),
            f"Only the picked {self._vertex_layers_label(layers)} is counted; not an all-layer total.",
            "Each grain uses its own four current atom vertices.",
            (
                "GB side visibility applied."
                if self.controls.manual_visible_check.isChecked()
                else "Both grains; independent of view and display-layer filter."
            ),
        ]
        for grain, layer in enumerate(layers):
            interior = counts.interior[grain, layer]
            boundary = counts.boundary[grain, layer]
            lines.append(
                f"G{grain+1}: area {counts.areas[grain]:.6g} a₀²"
            )
            if counts.half_open_available[grain]:
                total = counts.half_open[grain, layer]
                edge = counts.half_open_edges[grain, layer]
                corner = counts.half_open_corners[grain, layer]
                lines.append(
                    ("  Half-open geometric count · " if custom else "  ")
                    + f"{self._cell_layer_label(layer)}: {total} atoms = "
                    f"{interior} inside + {edge} edge + {corner} corner"
                )
                lines.append(
                    f"  Closed polygon: {interior+boundary} atoms ({boundary} on the boundary); "
                    "no periodic equivalence is assumed."
                    if custom else
                    f"  Closed outline would show {interior+boundary} point instances "
                    f"({boundary} on the boundary); opposite-edge copies are duplicates."
                )
            else:
                lines.append(
                    f"  {self._cell_layer_label(layer)}: {interior+boundary} atoms "
                    f"in the closed polygon = {interior} inside + {boundary} boundary"
                )
                lines.append(
                    "  No half-open parallelogram count is defined for this quadrilateral."
                    if custom else "  The selected quadrilateral is not a parallelogram, so no "
                    "periodic unique-cell count is defined."
                )
        lines.append("G1/G2 counted separately; overlapping atoms are not merged.")
        self.controls.manual_info.setPlainText("\n".join(lines))
        self.plot._draw_manual_cell()

    def _fit_manual_cell(self, *_args):
        if len(self.state.manual_vertices) == 4:
            self.plot._fit_model_corners(self._manual_grain_polygons().reshape(-1, 2))

    def _empty_coincidences(self):
        return self.state.empty_coincidences()[0]

    def _clear_coincidences(self):
        self.state.coincident_points, self.state.coincident_second_layers = (
            self.state.empty_coincidences()
        )

    def _grain_layer_sets(self, layers):
        """Restrict shared layer IDs to the layers each grain actually has."""
        return [set(layers) & set(range(count)) for count in self.state.crystal.layer_counts]

    def _coincidence_dirty_layers(self, grain, layers):
        """Coincidence groups affected by a grain filter; G2 can affect any group."""
        return set(layers) if grain == 0 or self.state.crystal.homophase else None

    def _set_all_layers_visible(self, visible: bool) -> None:
        layers = set(range(self.state.layer_count)) if visible else set()
        grain_layers = self._grain_layer_sets(layers)
        dirty_grains = {
            (grain, layer)
            for grain, (previous, current) in enumerate(
                zip(self.state.visible_grain_layers, grain_layers))
            for layer in previous ^ current
        }
        dirty_coincidences = (self.state.visible_layers ^ layers
                              if self.state.crystal.homophase else None)
        self.state.visible_grain_layers = grain_layers
        self._sync_shared_visible_layers()
        for checks in self.controls.grain_layer_checks:
            for check in checks:
                with QtCore.QSignalBlocker(check):
                    check.setChecked(visible)
        self._sync_compatibility_layer_checks()
        self.state.selected_layer = -1
        with QtCore.QSignalBlocker(self.controls.layer_combo):
            self.controls.layer_combo.setCurrentIndex(0)
        self._apply_layer_visibility(dirty_grains, dirty_coincidences)

    def _sync_shared_visible_layers(self) -> None:
        self.state.visible_layers = set.intersection(*self.state.visible_grain_layers)

    def _sync_compatibility_layer_checks(self) -> None:
        for layer, check in enumerate(self.controls.layer_checks):
            with QtCore.QSignalBlocker(check):
                check.setChecked(layer in self.state.visible_layers)

    def _on_grain_layer_toggled(self, grain: int, layer: int, visible: bool) -> None:
        if visible:
            self.state.visible_grain_layers[grain].add(layer)
        else:
            self.state.visible_grain_layers[grain].discard(layer)
        self._sync_shared_visible_layers()
        self._sync_compatibility_layer_checks()
        equal_selections = (
            self.state.visible_grain_layers[0] == self.state.visible_grain_layers[1]
        )
        self.state.selected_layer = (
            next(iter(self.state.visible_layers))
            if equal_selections and len(self.state.visible_layers) == 1
            else -1
        )
        self._apply_layer_visibility(
            {(grain, layer)}, self._coincidence_dirty_layers(grain, {layer}))

    def _on_axial_layer_toggled(self, layer: int, visible: bool) -> None:
        grains = [grain for grain, count in enumerate(self.state.crystal.layer_counts)
                  if layer < count]
        for grain in grains:
            if visible:
                self.state.visible_grain_layers[grain].add(layer)
            else:
                self.state.visible_grain_layers[grain].discard(layer)
            with QtCore.QSignalBlocker(self.controls.grain_layer_checks[grain][layer]):
                self.controls.grain_layer_checks[grain][layer].setChecked(visible)
        self._sync_shared_visible_layers()
        self.state.selected_layer = (
            next(iter(self.state.visible_layers))
            if len(self.state.visible_layers) == 1
            else -1
        )
        self._apply_layer_visibility(
            {(grain, layer) for grain in grains},
            self._coincidence_dirty_layers(max(grains, default=0), {layer}),
        )

    def _apply_layer_visibility(self, dirty_grains=None, dirty_coincidences=None) -> None:
        self.plot._rebuild_legend()
        self._update_visible_points(
            dirty_grain_layers=dirty_grains,
            dirty_coincidence_layers=dirty_coincidences,
        )

    def _on_layer_changed(self, *_args):
        self.state.selected_layer = self.controls.layer_combo.currentData()
        layers = (
            set(range(self.state.layer_count))
            if self.state.selected_layer < 0
            else {self.state.selected_layer}
        )
        self.state.visible_grain_layers = self._grain_layer_sets(layers)
        self._sync_shared_visible_layers()
        for grain, checks in enumerate(self.controls.grain_layer_checks):
            for layer, check in enumerate(checks):
                with QtCore.QSignalBlocker(check):
                    check.setChecked(layer in self.state.visible_grain_layers[grain])
        self._sync_compatibility_layer_checks()
        self._apply_layer_visibility()

    def _geometry_signature(self):
        """Identify physical positions independently of the view rectangle."""
        return (
            self.state.crystal.lattice,
            self.state.crystal.axis,
            self.state.crystal.scale,
            self.state.angle_deg,
            tuple(self.state.deformations[0].ravel()),
            tuple(self.state.deformations[1].ravel()),
            tuple(self.state.translations.ravel()),
            self._displacement_signature(),
        )

    def _update_geometry_labels(self):
        self.plot._update_reference_axes()
        crystal = self.state.crystal
        model = self.state.geometry
        if crystal.homophase:
            name = f"{model.lattice} ⟨{model.axis}⟩ Tilt GB"
            units = "Coordinates in a₀ · 1 = one lattice constant"
        else:
            name = f"{crystal.label} Interface"
            units = (f"Coordinates in a₀ of G1 · a₀(G2)/a₀(G1) = {crystal.scale:.6g}"
                     if crystal.scale != 1.0 else "Coordinates in a₀ · 1 = one lattice constant")
        self.setWindowTitle(name + " · Dichromatic Pattern")
        self.controls.heading.setText(
            f"<div style='font-size:17px;font-weight:700;color:#17212b'>{html.escape(name)}</div>"
            f"<div style='color:#64748b;margin-top:3px'>{units}</div>"
        )
        self.plot.plot_item.setLabel("bottom", f"{model.x_label} / a₀", units="")
        self.plot.plot_item.setLabel("left", f"{model.y_label} / a₀", units="")
        legend_note = (
            "\nLegend shows the first 6 enabled layers; use the grain/layer checkboxes to inspect others."
            if self.state.layer_count > 6 else ""
        )
        if crystal.homophase:
            self.controls.layer_info.setText(
                f"{model.layer_count} axial layers per grain · spacing = {model.layer_spacing:.6g} a₀\n"
                f"Axial repeat = {model.axial_period:.6g} a₀"
                + legend_note
                + "\nCSL and near-pair markers require the layer to be visible in both grains."
            )
        else:
            self.controls.layer_info.setText(
                "\n".join(
                    f"G{grain + 1} {geometry.lattice} {geometry.axis_label}: "
                    f"{geometry.layer_count} layers · spacing "
                    f"{scale * geometry.layer_spacing:.6g} a₀ · repeat "
                    f"{scale * geometry.axial_period:.6g} a₀"
                    for grain, (geometry, scale) in enumerate(
                        zip(self.state.geometries, crystal.scales))
                )
                + legend_note
                + "\nGrain layers are unrelated: coincident columns and near pairs match "
                "projected columns of any G1/G2 layers, both visible."
            )
        self.controls.axial_spin.setToolTip(
            "Vector readout only: P2 shifts by "
            + " or ".join(dict.fromkeys(
                self._format_lattice_vector(geometry.axial_repeat_half_indices)
                + f" along {geometry.axis_label}"
                + ("" if crystal.homophase else f" in G{grain + 1}")
                for grain, geometry in enumerate(self.state.geometries)
            ))
            + " per step. This selects a periodic atom "
            "represented by the same projected column; it does not add thickness, "
            "duplicate the structure, or change plotted atoms."
        )

    def _on_heterophase_toggled(self, checked):
        controls = self.controls
        if checked:
            # G2 starts as a copy of G1: the structure is unchanged until edited.
            first = self.state.geometries[0]
            widgets = (controls.structure2_combo, controls.axis2_combo,
                       controls.lattice_ratio_spin)
            with ExitStack() as blockers:
                for widget in widgets:
                    blockers.enter_context(QtCore.QSignalBlocker(widget))
                controls.structure2_combo.setCurrentIndex(
                    controls.structure2_combo.findData(first.lattice))
                index = controls.axis2_combo.findData(first.axis)
                controls.axis2_combo.setCurrentIndex(
                    index if index >= 0 else controls.axis2_combo.findData(None))
                controls.custom_axis2_edit.setText(" ".join(map(str, first.axis_indices)))
                controls.lattice_ratio_spin.setValue(1.0)
            controls._sync_heterophase_rows()
            return
        controls._sync_heterophase_rows()
        self._on_geometry_changed(apply_custom=True)

    def _requested_crystal(self, apply_custom):
        """The structure selected by the controls, or None while a custom axis is pending."""
        controls = self.controls
        heterophase = controls.heterophase_check.isChecked()
        controls.custom_axis_row.setVisible(controls.axis_combo.currentData() is None)
        controls._sync_heterophase_rows()
        grains = [(controls.structure_combo, controls.axis_combo, controls.custom_axis_edit)]
        if heterophase:
            grains.append((controls.structure2_combo, controls.axis2_combo,
                           controls.custom_axis2_edit))
        lattices, axes = [], []
        for structure, axis_combo, edit in grains:
            axis = axis_combo.currentData()
            if axis is None:
                if not apply_custom:
                    return None
                axis = edit.text()
            lattices.append(structure.currentData())
            axes.append(axis)
        if not heterophase:
            return bicrystal(lattices[0], axes[0])
        ratio = controls.lattice_ratio_spin.value()
        if ratio == round(self.state.crystal.scale, controls.lattice_ratio_spin.decimals()):
            ratio = self.state.crystal.scale  # Keep a saved ratio the spin box rounds.
        return bicrystal(lattices, axes, ratio)

    def _on_geometry_changed(self, *_args, apply_custom=False):
        controls = self.controls
        try:
            crystal = self._requested_crystal(apply_custom)
        except ValueError as error:
            controls.axis_error.setText(str(error))
            controls.axis_error.show()
            # Keep the edited axis text for correction; restore the rest.
            state = self.state
            widgets = (controls.structure_combo, controls.structure2_combo,
                       controls.lattice_ratio_spin)
            with ExitStack() as blockers:
                for widget in widgets:
                    blockers.enter_context(QtCore.QSignalBlocker(widget))
                for combo, geometry in zip(widgets[:2], state.geometries):
                    combo.setCurrentIndex(combo.findData(geometry.lattice))
                controls.lattice_ratio_spin.setValue(state.crystal.scale)
            return
        if crystal is None:
            return
        controls.axis_error.clear()
        controls.axis_error.hide()
        parameters = replace(
            self.state.parameters, lattice=crystal.lattices[0], axis=crystal.axes[0],
            lattice2=None if crystal.homophase else crystal.lattices[1],
            axis2=None if crystal.homophase else crystal.axes[1],
            lattice_ratio=crystal.scale,
        )
        previous = self.state.crystal
        if crystal == previous:
            self.state.parameters = parameters
            return
        for timer in (
            self.angle_preview_timer,
            self.coincidence_timer,
            self.view_refresh_timer,
            self.near_debounce_timer,
            self.near_poll_timer,
        ):
            timer.stop()
        self._cancel_parallel_work()
        if self.compute.near_search is not None:
            self.compute.near_search.cancel()
        self._clear_lattice_selections()
        self.state.grain_displacements = [[], []]
        self._set_translation_offset([0, 0])
        self._sync_translation_controls("Manual moves cleared by the lattice/axis change.")
        self.state.apply_crystal(crystal)
        self.state.layer_symbols = resize_layer_symbols(self.state.layer_symbols, self.state.layer_count)
        self.state.layer_size_scales = resize_layer_size_scales(
            self.state.layer_size_scales, self.state.layer_count)
        self.state.render_error = None
        self.state.parameters = parameters
        # Keep an angle only while its meaning (both axes, grain exchange) holds.
        if (crystal.axes != previous.axes or crystal.homophase != previous.homophase
                or self.state.angle_deg > self.state.angle_range.maximum_deg):
            self.state.angle_deg = default_crystal_angle_deg(crystal)
        self.state.pending_angle = self.state.angle_deg
        self.state.angle_update_active = False
        self._clear_near_cell(reset_custom=True)
        self._populate_preset_combo()
        self.state.axial_repeat = 0
        with QtCore.QSignalBlocker(self.controls.axial_spin):
            self.controls.axial_spin.setValue(0)
            self.controls.axial_spin.setPrefix("+")
        self._sync_angle_controls()
        self._update_geometry_labels()
        # Hide the previous geometry while its replacement is generated.
        self.state.grains = []
        self.state.grain_signature = None
        self.state.visible_atom_masks = []
        self.state.visible_atom_counts = [0, 0]
        self.state.visible_coincidence_counts = [0] * self.state.layer_count
        self._clear_coincidences()
        self.state.selected_layer = -1
        self.state.visible_grain_layers = [set(range(count)) for count in crystal.layer_counts]
        self._sync_shared_visible_layers()
        self.controls._populate_layer_combo()
        self.controls._rebuild_layer_checks()
        self.plot._create_layer_items()
        self.plot._update_marker_sizes()
        self.state.buffer_bounds = None
        self._update_title()
        if self.state.near_enabled:
            self._queue_near_search()
        else:
            self._start_parallel_regeneration(True)
            self.controls.near_info.setPlainText(
                "Off. Original unstrained lattices displayed."
            )
        self._update_status(f"Changed to {crystal.label}; previous selections cleared.")
        self._sync_translation_controls("Manual moves cleared by the lattice/axis change.")

    def _populate_preset_combo(self):
        combo = self.controls.preset_combo
        with QtCore.QSignalBlocker(combo):
            combo.clear()
            combo.addItem("Custom angle", None)
            for index, preset in enumerate(self.state.presets):
                combo.addItem(preset.label, index)
        self.controls.preset_label.setText(
            "CSL preset" if self.state.crystal.homophase else "Orientation relationship")

    @property
    def common_cell(self):
        if self._has_displacements() or self._custom_position_base():
            return None
        return (
            self.state.near_cell
            if self.state.near_cell is not None
            else exact_csl_cell(
                self.state.angle_deg,
                lattice=self.state.crystal.lattice,
                axis=self.state.crystal.axis,
                scale=self.state.crystal.scale,
            )
        )

    def _clear_near_cell(self, *, reset_custom=False):
        preserve_base = not reset_custom and (self._has_displacements() or self._custom_position_base())
        if self.state.near_cell is not None:
            self._clear_lattice_selections()
        self.state.near_cell = None
        self.state.near_solutions = []
        if not preserve_base:
            self.state.deformations = (np.eye(2), np.eye(2))
            self.state.translations = np.zeros((2, 2))
        self.state.manual_strain_fit = None
        self.state.manual_unstrained_vertices = None
        self.state.manual_unstrained_cutoff = None
        self.controls.manual_strain_note.setText(
            "Select a local near-CSL cell first. Both grains share the strain; no individual atoms are snapped."
        )
        # A geometry switch can be between updating its model and rebuilding
        # per-layer plot items here; do not rebuild the legend in that interval.
        self.controls.local_distance_spin.setEnabled(self.local_active)
        self._sync_manual_strain_controls()
        self.plot._update_common_cell()
        with QtCore.QSignalBlocker(self.controls.near_combo):
            self.controls.near_combo.clear()

    @property
    def local_active(self):
        return (
            self.state.near_enabled
            and self.state.near_method == "local"
            and self.state.manual_strain_fit is None
        )

    def _clear_local_pairs(self):
        self.state.local_pairs = LocalPairs.empty()
        self.state.local_error = None
        self.state.local_updating = self.local_active
        self.plot._set_scatter(self.plot.local_match_item, np.empty((0, 2)))
        self.plot.local_link_item.setData([], [])

    def _sync_near_controls(self):
        custom = self._has_displacements() or self._custom_position_base()
        index = self.controls.near_method_combo.findData("strain")
        self.controls.near_method_combo.model().item(index).setEnabled(not custom)
        strain_method = self.state.near_method == "strain"
        strained = self.state.near_enabled and strain_method and not custom
        for control in (
            self.controls.strain_spin,
            self.controls.search_index_spin,
            self.controls.near_combo,
        ):
            control.setEnabled(strained)
            control.setVisible(strain_method)
            if control is not self.controls.near_combo:
                self.controls.near_form.labelForField(control).setVisible(strain_method)
        self.controls.local_distance_spin.setEnabled(self.local_active)
        self.controls.local_distance_spin.setVisible(not strain_method)
        self.controls.near_form.labelForField(
            self.controls.local_distance_spin
        ).setVisible(not strain_method)
        self.plot._rebuild_legend()
        self._sync_manual_strain_controls()

    def _on_near_method_changed(self, *_args):
        if self.controls.near_method_combo.currentData() == "strain" and (self._has_displacements() or self._custom_position_base()):
            with QtCore.QSignalBlocker(self.controls.near_method_combo):
                self.controls.near_method_combo.setCurrentIndex(self.controls.near_method_combo.findData("local"))
            return
        self.state.near_method = self.controls.near_method_combo.currentData()
        self._sync_near_controls()
        if self.state.near_enabled:
            self._queue_near_search()
        else:
            self.controls.near_info.setPlainText(
                "Off. Enable Near-CSL to run the selected method."
            )
        self._update_title()

    def _toggle_near(self, enabled):
        self.state.near_enabled = bool(enabled)
        self._invalidate_manual_local_source()
        self._sync_near_controls()
        self.controls.near_button.setText(
            "Disable Near-CSL" if enabled else "Enable Near-CSL"
        )
        if enabled:
            self._queue_near_search()
        else:
            self.near_debounce_timer.stop()
            self.near_poll_timer.stop()
            if self.compute.near_search is not None:
                self.compute.near_search.cancel()
            self._clear_local_pairs()
            self._clear_near_cell()
            self._cancel_parallel_work()
            self._start_parallel_regeneration(True)
            self.controls.near_info.setPlainText(
                "Off. Manual positions and their bulk deformation are retained."
                if self._has_displacements() or self._custom_position_base()
                else "Off. Original unstrained lattices restored."
            )
        self._update_title()

    def _queue_near_search(self, *_args):
        if not self.state.near_enabled:
            return
        if (
            self.state.manual_strain_fit is not None
            and self.state.near_method == "local"
        ):
            return  # Keep selected-cell strain on worker-count changes / fallback.
        self._invalidate_manual_local_source()
        self.near_debounce_timer.stop()
        self.near_poll_timer.stop()
        if self.compute.near_search is not None:
            self.compute.near_search.cancel()
        self._clear_local_pairs()
        self._clear_near_cell()
        if self.local_active:
            # A threshold change only changes annotations, preserving picks.
            # Rebuild only if switching away from strain or needing a halo.
            if self.compute.parallel_stage == "local":
                self._cancel_parallel_work()
            if (
                self.state.grain_signature != self._geometry_signature()
                or not self.plot._buffer_contains_view()
            ):
                self._start_parallel_regeneration(True)
            else:
                self.near_debounce_timer.start()
            self.plot._update_local_overlay()
            self._update_title()
            return
        self._cancel_parallel_work()
        self._start_parallel_regeneration(True)
        self.controls.near_info.setPlainText("Waiting for parameters to settle…")
        self.near_debounce_timer.start()

    def _start_near_search(self):
        if not self.state.near_enabled or self.state.manual_strain_fit is not None:
            return
        if self.local_active:
            self._start_local_matching()
            return
        if self._has_displacements() or self._custom_position_base():
            return
        if self.compute.near_search is None:
            self.compute.near_search = NearSearch(
                self.compute.worker_count, executor=self.compute.thread_executor()
            )
        self.compute.near_search.request(
            self.state.angle_deg,
            self.controls.strain_spin.value(),
            self.controls.search_index_spin.value(),
            lattice=self.state.crystal.lattice,
            axis=self.state.crystal.axis,
            scale=self.state.crystal.scale,
        )
        self.near_poll_timer.start()
        self._poll_near_search()  # Cached complete searches need no timer delay.

    def _poll_near_search(self):
        if (
            self.compute.near_search is None
            or not self.state.near_enabled
            or self.state.near_method != "strain"
            or self._has_displacements()
            or self._custom_position_base()
        ):
            self.near_poll_timer.stop()
            return
        result = self.compute.near_search.poll()
        if self.compute.near_search.error:
            self.controls.near_info.setPlainText(
                "Near-CSL search failed: " + self.compute.near_search.error
            )
            self.near_poll_timer.stop()
        elif result is not None:
            self.near_poll_timer.stop()
            self.state.near_solutions = result
            with QtCore.QSignalBlocker(self.controls.near_combo):
                self.controls.near_combo.clear()
                self.controls.near_combo.addItems([cell.label() for cell in result])
            if result:
                self._select_near_cell(0)
            else:
                self.controls.near_info.setPlainText(
                    "No compatible cell found within this bounded search.\n"
                    "Increase the index bound or principal strain limit.\nUnstrained lattices remain displayed."
                )
        elif self.compute.near_search.busy:
            self.controls.near_info.setPlainText(
                f"Searching strained periodic cells…\n"
                f"{self.compute.near_search.completed}/{self.compute.near_search.total} chunks complete\n"
                f"{self.compute.worker_count} workers; navigation remains available."
            )

    def _select_near_cell(self, index):
        if (
            not self.state.near_enabled
            or self._has_displacements()
            or self._custom_position_base()
            or self.local_active
            or not 0 <= index < len(self.state.near_solutions)
        ):
            return
        self._cancel_parallel_work()
        self._clear_lattice_selections()
        self.state.near_cell = self.state.near_solutions[index]
        self.state.deformations = (self.state.near_cell.f1, self.state.near_cell.f2)
        self.controls.near_info.setPlainText(
            tensor_readout(self.state.near_cell, self.state.angle_deg)
        )
        self.plot._update_common_cell()
        self._start_parallel_regeneration(True)

    def _start_local_matching(self):
        if not self.local_active or len(self.state.grains) != 2:
            return
        if self.compute.parallel_stage in ("grains", "waiting", "coincidences"):
            # Exact detection will start matching after the new atoms arrive.
            if self.compute.parallel_stage == "grains":
                self.compute.parallel_payload["compute_coincidences"] = True
            elif self.compute.parallel_stage == "waiting":
                self.compute.regeneration_pending[1]["compute_coincidences"] = True
            return
        if self.state.grain_signature != self._geometry_signature():
            return
        self.near_debounce_timer.stop()
        self._cancel_parallel_work()
        self._clear_local_pairs()
        distance = self.controls.local_distance_spin.value()
        self.compute.parallel_stage = "local"
        self.compute.parallel_payload = {
            "geometry_signature": self.state.grain_signature,
            "distance": distance,
        }
        try:
            if not self.state.crystal.homophase:
                # Mutual-nearest columns compete across every layer of both
                # grains, so this match is not split into layer groups.
                self.compute.parallel_futures = [self.compute.thread_executor().submit(
                    projected_local_match_worker, *self.state.grains, distance,
                )]
            else:
                executor, matching_workers = self.compute.matching_executor(
                    sum(len(grain.positions) for grain in self.state.grains),
                    self.state.layer_count,
                )
                groups = np.array_split(
                    np.arange(self.state.layer_count),
                    min(matching_workers, self.state.layer_count),
                )
                selections = [grain.layer_selections() for grain in self.state.grains]
                self.compute.parallel_futures = [
                    executor.submit(
                        local_match_layers_worker,
                        grain_layer_subset(self.state.grains[0], layers, selections[0]),
                        grain_layer_subset(self.state.grains[1], layers, selections[1]),
                        distance,
                        layers,
                    )
                    for layers in groups
                ]
        except Exception as error:
            self._parallel_failure(error)
            return
        self.parallel_poll_timer.start()
        self.plot._update_local_overlay()
        self._update_status()

    def _local_pair_mask(self):
        """Visibility in physical coordinates; also used for manual picking."""
        pairs = self.state.local_pairs
        mask = np.full(len(pairs.layers), self.local_active, dtype=bool)
        if self.state.grain_signature != self._geometry_signature():
            mask[:] = False
        # Both endpoints' own layers must be shown (one shared layer if homophase).
        mask &= np.isin(pairs.layers, tuple(self.state.visible_grain_layers[0]))
        mask &= np.isin(pairs.second_layers, tuple(self.state.visible_grain_layers[1]))
        if len(self.state.selected_points) == 2:
            states = self._region_states()
            for grain, points in enumerate((pairs.first, pairs.second)):
                mask &= selected_region_mask(
                    points,
                    *self.state.selected_points,
                    states[2 * grain],
                    states[2 * grain + 1],
                )
        midpoints = pairs.midpoints
        # Reject buffer-edge assignments lacking the full mutual-neighbor halo.
        # All displayed view points have a 2*d halo after buffered regeneration.
        if self.local_active and self.state.buffer_bounds is not None:
            left, right, bottom, top = self.state.buffer_bounds
            halo = 2 * self.controls.local_distance_spin.value()
            mask &= (
                (midpoints[:, 0] >= left + halo)
                & (midpoints[:, 0] <= right - halo)
                & (midpoints[:, 1] >= bottom + halo)
                & (midpoints[:, 1] <= top - halo)
            )
        return mask

    def _fit_near_cell(self):
        """Fit whichever periodic cell is displayed (exact CSL or Near-CSL)."""
        cell = self.common_cell
        if cell is None:
            return
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ cell.cell.T
        if self.state.manual_strain_fit is not None:
            corners += self.state.manual_strain_fit.origin
        self.plot._fit_model_corners(corners)

    def _configure_worker_pool(self, worker_count: int) -> None:
        self.compute.configure(worker_count)

    def _on_worker_count_changed(self, worker_count: int) -> None:
        recount = (
            len(self.state.manual_vertices) == 4 and self.state.manual_counts is None
        )
        self._cancel_parallel_work()
        self.near_poll_timer.stop()
        if self.compute.near_search is not None:
            self.compute.near_search.close()
            self.compute.near_search = None
        self._configure_worker_pool(worker_count)
        self._start_parallel_regeneration(True)
        if self.state.near_enabled:
            self._queue_near_search()
        self._update_status(
            f"Compute pool changed to {self.compute.worker_count} workers."
        )
        if recount:
            self.state.manual_count_key = None
            self._queue_manual_count()

    def _cancel_parallel_work(self) -> None:
        self.compute.cancel_parallel()
        if hasattr(self, "parallel_poll_timer"):
            self.parallel_poll_timer.stop()

    def _regenerate_buffer(self, compute_coincidences: bool) -> None:
        if self.compute.parallel_stage == "local":
            self._cancel_parallel_work()
        self._clear_local_pairs()
        try:
            self._regenerate_buffer_checked(compute_coincidences)
        except GeometryLimitError as error:
            self.state.csl_updating = False
            self.state.angle_update_active = False
            self.state.render_error = str(error)
            if self.local_active:
                self.state.local_updating = False
                self.state.local_error = str(error)
                self.plot._update_local_overlay()
            self._update_status(str(error))

    def _regenerate_buffer_checked(self, compute_coincidences: bool) -> None:
        center, width, height, bounds = self.plot._buffer_geometry()
        half_angle = 0.5 * self.state.angle_deg
        crystal = self.state.crystal
        self.state.grains = [
            projected_columns(
                width,
                height,
                sign * half_angle,
                center=tuple(center),
                deformation=self.state.deformations[grain],
                lattice=crystal.lattices[grain],
                axis=crystal.axes[grain],
                translation=self.state.translations[grain],
                displacements=self.state.grain_displacements[grain],
                scale=crystal.scales[grain],
            )
            for grain, sign in enumerate((1.0, -1.0))
        ]
        self.state.buffer_bounds = bounds
        self.state.render_error = None
        self.state.grain_signature = self._geometry_signature()
        if compute_coincidences:
            self._compute_coincidences()
        else:
            self._clear_coincidences()
            self.state.csl_updating = True
        self._update_visible_points()
        self._update_title()

    def _start_parallel_regeneration(self, compute_coincidences: bool) -> None:
        self._cancel_parallel_work()
        self._clear_local_pairs()
        center, width, height, bounds = self.plot._buffer_geometry()
        half_angle = 0.5 * self.state.angle_deg
        crystal = self.state.crystal
        args = [
            (
                width, height, sign * half_angle, tuple(center),
                self.state.deformations[index].copy(),
                crystal.lattices[index], crystal.axes[index],
                self.state.translations[index].copy(),
                normalize_grain_displacements(self.state.grain_displacements)[index],
                crystal.scales[index],
            )
            for index, sign in enumerate((1.0, -1.0))
        ]
        payload = {
            "bounds": bounds,
            "compute_coincidences": compute_coincidences,
            "geometry_signature": self._geometry_signature(),
        }
        # Keep only the newest request. Running jobs cannot be cancelled, so
        # drain them before adding replacements instead of growing the queue.
        self.compute.regeneration_pending = (args, payload)
        self._clear_coincidences()
        self.state.csl_updating = True
        for item in self.plot.coincidence_items:
            self.plot._set_scatter(item, np.empty((0, 2)))
        self.compute.parallel_stage = "waiting"
        self._submit_pending_regeneration()
        if self.compute.parallel_stage is None:
            return  # A submission failure has already reported its outcome.
        self._update_status()
        self.parallel_poll_timer.start()

    def _submit_pending_regeneration(self) -> None:
        self.compute.parallel_retired = [
            future for future in self.compute.parallel_retired if not future.done()
        ]
        if self.compute.parallel_retired or self.compute.regeneration_pending is None:
            return
        args, payload = self.compute.regeneration_pending
        self.compute.regeneration_pending = None
        executor = self.compute.thread_executor()
        try:
            self.compute.parallel_futures = []
            for grain_args in args:
                self.compute.parallel_futures.append(
                    executor.submit(generate_grain_worker, *grain_args)
                )
        except Exception as error:  # pragma: no cover - platform-specific failure
            self._parallel_failure(error)
            return
        self.compute.parallel_payload = payload
        self.compute.parallel_stage = "grains"

    def _start_parallel_coincidences(self) -> None:
        if self.compute.parallel_stage == "waiting":
            self.compute.regeneration_pending[1]["compute_coincidences"] = True
            return
        if self.compute.parallel_stage == "grains":
            self.compute.parallel_payload["compute_coincidences"] = True
            return
        if len(self.state.grains) != 2:
            return
        self._cancel_parallel_work()
        tolerance = COINCIDENCE_TOLERANCE_FACTOR
        projected = not self.state.crystal.homophase
        try:
            if projected:
                self.compute.parallel_futures = [self.compute.thread_executor().submit(
                    projected_coincidence_worker, *self.state.grains, tolerance,
                )]
            else:
                executor, matching_workers = self.compute.matching_executor(
                    sum(len(grain.positions) for grain in self.state.grains),
                    self.state.layer_count,
                )
                layer_groups = np.array_split(
                    np.arange(self.state.layer_count),
                    min(matching_workers, self.state.layer_count),
                )
                selections = [grain.layer_selections() for grain in self.state.grains]
                self.compute.parallel_futures = [
                    executor.submit(
                        coincidence_layers_worker,
                        grain_layer_subset(self.state.grains[0], layers, selections[0]),
                        grain_layer_subset(self.state.grains[1], layers, selections[1]),
                        tolerance,
                        layers,
                    )
                    for layers in layer_groups
                ]
        except Exception as error:  # pragma: no cover - platform-specific failure
            self._parallel_failure(error)
            return
        self.compute.parallel_payload = {"projected": projected}
        self.compute.parallel_stage = "coincidences"
        self.state.csl_updating = True
        self.parallel_poll_timer.start()
        self._update_status()

    def _poll_parallel_work(self) -> None:
        if self.compute.parallel_stage == "waiting":
            self._submit_pending_regeneration()
        if not self.compute.parallel_futures or not all(
            future.done() for future in self.compute.parallel_futures
        ):
            return
        stage = self.compute.parallel_stage
        try:
            results = [future.result() for future in self.compute.parallel_futures]
        except Exception as error:  # pragma: no cover - platform-specific failure
            self._parallel_failure(error)
            return

        self.compute.worker_process_ids.update(
            process_id for process_id, _result in results
        )
        self.compute.parallel_futures = []
        self.compute.parallel_stage = None
        if stage == "grains":
            self.state.grains = [grain for _process_id, grain in results]
            self.state.render_error = None
            self.state.grain_signature = self.compute.parallel_payload[
                "geometry_signature"
            ]
            self.state.buffer_bounds = self.compute.parallel_payload["bounds"]  # type: ignore[assignment]
            compute_coincidences = bool(
                self.compute.parallel_payload.get("compute_coincidences", True)
            )
            self.compute.parallel_payload = {}
            self._update_visible_points()
            self._update_title()
            if compute_coincidences:
                self._start_parallel_coincidences()
            else:
                self.parallel_poll_timer.stop()
            if not self.plot._buffer_contains_view():
                self.view_refresh_timer.start()
            return

        if stage == "coincidences":
            projected = self.compute.parallel_payload.get("projected", False)
            self.compute.parallel_payload = {}
            if projected:
                ((_pid, (points, partners)),) = results
                self.state.coincident_points = points
                self.state.coincident_second_layers = partners
            else:
                by_layer = {
                    layer: points for _pid, batch in results for layer, points in batch
                }
                self._set_same_layer_coincidences(
                    [by_layer[layer] for layer in range(self.state.layer_count)]
                )
            self.state.csl_updating = False
            self.state.angle_update_active = False
            self.parallel_poll_timer.stop()
            self._update_visible_points(grains_changed=False)
            self._update_title()
            if self.local_active:
                self._start_local_matching()
            else:
                self.compute.warm_matching_pool(
                    sum(len(grain.positions) for grain in self.state.grains),
                    self.state.layer_count,
                )
            return

        if stage == "local":
            self.parallel_poll_timer.stop()
            current = (
                self.local_active
                and self.compute.parallel_payload.get("geometry_signature")
                == self._geometry_signature()
                and self.compute.parallel_payload.get("distance")
                == self.controls.local_distance_spin.value()
            )
            self.compute.parallel_payload = {}
            if current:
                self.state.local_pairs = LocalPairs.concatenate(
                    batch for _pid, batch in results
                )
                self.state.local_updating = False
                self.plot._update_local_overlay()
                self._update_status()
                self.compute.warm_matching_pool(
                    sum(len(grain.positions) for grain in self.state.grains),
                    self.state.layer_count,
                )

    def _parallel_failure(self, error: Exception) -> None:
        local_failure = self.compute.parallel_stage == "local"
        self._cancel_parallel_work()
        if isinstance(error, BrokenProcessPool) and self.compute.retire_matching_pool():
            if local_failure:
                self._start_local_matching()
            elif len(self.state.grains) == 2 and self.state.grain_signature == self._geometry_signature():
                self._start_parallel_coincidences()
            else:
                self._start_parallel_regeneration(True)
            self._update_status("Matching process exited; retrying on a background thread.")
            return
        if local_failure:
            self.state.local_updating = False
            self.state.local_error = str(error)
            self.plot._update_local_overlay()
            self._update_status(
                "Local matching failed; original lattices are unchanged."
            )
            return
        if isinstance(error, GeometryLimitError):
            self.state.csl_updating = False
            self.state.angle_update_active = False
            self.state.render_error = str(error)
            if self.local_active:
                self.state.local_updating = False
                self.state.local_error = str(error)
                self.plot._update_local_overlay()
            self._update_status(str(error))
            return
        self.near_poll_timer.stop()
        if self.compute.near_search is not None:
            self.compute.near_search.close()
            self.compute.near_search = None
        failed_executor = self.compute.executor
        self.compute.executor = None
        if failed_executor is not None:
            failed_executor.shutdown(wait=False, cancel_futures=True)
        self.compute.worker_count = 1
        if hasattr(self.controls, "worker_spin"):
            with QtCore.QSignalBlocker(self.controls.worker_spin):
                self.controls.worker_spin.setValue(1)
        self.state.angle_update_active = False
        self._regenerate_buffer(compute_coincidences=True)
        if self.state.near_enabled:
            self._queue_near_search()
        self._update_status(
            f"Parallel calculation failed; using one process ({error})."
        )

    def _compute_coincidences(self) -> None:
        if len(self.state.grains) != 2:
            return
        tolerance = COINCIDENCE_TOLERANCE_FACTOR
        if self.state.crystal.homophase:
            self._set_same_layer_coincidences(same_layer_coincidence_sites(
                self.state.grains[0], self.state.grains[1], tolerance
            ))
        else:
            self.state.coincident_points, self.state.coincident_second_layers = (
                projected_coincidence_sites(*self.state.grains, tolerance)
            )
        self.state.csl_updating = False
        if self.local_active:
            self._start_local_matching()

    def _set_same_layer_coincidences(self, sites):
        self.state.coincident_points = tuple(sites)
        self.state.coincident_second_layers = tuple(
            np.full(len(points), layer, dtype=np.int16) for layer, points in enumerate(sites)
        )

    def _refresh_view_buffer(self) -> None:
        self._start_parallel_regeneration(
            compute_coincidences=not self.state.angle_update_active
        )
        if self.state.angle_update_active:
            self.coincidence_timer.start()

    def _on_view_range_changed(self, *_args) -> None:
        self._sync_view_scale_control()
        self.plot._draw_boundary()
        self.plot._draw_vector()
        self.plot._draw_manual_cell()
        if not self.view_counts_timer.isActive():
            self.view_counts_timer.start()
        if (
            not self.plot._buffer_contains_view()
            and self.compute.parallel_stage not in ("grains", "waiting")
        ):
            self.view_refresh_timer.start()

    def _on_field_size_changed(self, raw_value: int) -> None:
        scale = raw_value / 100.0
        center, _width, _height = self.plot._view_geometry()
        width = self.state.parameters.width * scale
        height = self.state.parameters.height * scale
        self.plot.view_box.setRange(
            xRange=(center[0] - 0.5 * width, center[0] + 0.5 * width),
            yRange=(center[1] - 0.5 * height, center[1] + 0.5 * height),
            padding=0.0,
        )

    def _sync_view_scale_control(self) -> None:
        _center, width, _height = self.plot._view_geometry()
        scale = float(
            np.clip(width / self.state.parameters.width, VIEW_SCALE_MIN, VIEW_SCALE_MAX)
        )
        blocker = QtCore.QSignalBlocker(self.controls.view_slider)
        self.controls.view_slider.setValue(round(scale * 100.0))
        del blocker
        self.controls.view_scale_label.setText(f"{scale:.2f}×")

    def _region_states(self) -> tuple[bool, bool, bool, bool]:
        return tuple(check.isChecked() for check in self.controls.region_checks)  # type: ignore[return-value]

    def _update_visible_points(
        self, *_args, grains_changed=True,
        dirty_grain_layers=None, dirty_coincidence_layers=None,
    ) -> None:
        self.plot._update_reference_axes()
        if len(self.state.grains) != 2:
            return
        states = self._region_states()
        boundary_ready = len(self.state.selected_points) == 2
        incremental = dirty_grain_layers is not None and len(self.state.visible_atom_masks) == 2
        if grains_changed and not incremental:
            self.state.visible_atom_masks = []
        grain_items = enumerate(self.state.grains) if grains_changed else ()
        for grain_index, grain in grain_items:
            changed_layers = (
                {layer for index, layer in dirty_grain_layers if index == grain_index}
                if incremental else None
            )
            if changed_layers == set():
                continue
            selections = grain.layer_selections()
            if boundary_ready and not incremental:
                mask = selected_region_mask(
                    grain.positions,
                    self.state.selected_points[0],
                    self.state.selected_points[1],
                    states[2 * grain_index],
                    states[2 * grain_index + 1],
                )
            elif not incremental:
                mask = np.ones(len(grain.positions), dtype=bool)
            visible_mask = (
                self.state.visible_atom_masks[grain_index]
                if incremental else np.zeros(len(grain.positions), dtype=bool)
            )
            for layer, item in enumerate(self.plot.grain_layer_items[grain_index]):
                if changed_layers is not None and layer not in changed_layers:
                    continue
                selection = selections.get(layer, slice(0, 0))
                if layer in self.state.visible_grain_layers[grain_index]:
                    if incremental:
                        layer_mask = (
                            selected_region_mask(
                                grain.positions[selection], *self.state.selected_points,
                                states[2 * grain_index], states[2 * grain_index + 1],
                            )
                            if boundary_ready else np.ones(len(grain.positions[selection]), dtype=bool)
                        )
                    else:
                        layer_mask = mask[selection]
                    visible_mask[selection] = layer_mask
                    points = (
                        grain.positions[selection][layer_mask]
                        if boundary_ready else grain.positions[selection]
                    )
                else:
                    visible_mask[selection] = False
                    points = np.empty((0, 2))
                self.plot._set_scatter(item, points)
            if not incremental:
                self.state.visible_atom_masks.append(visible_mask)

        for layer, (points, partners, item) in enumerate(zip(
            self.state.coincident_points, self.state.coincident_second_layers,
            self.plot.coincidence_items, strict=True,
        )):
            if dirty_coincidence_layers is not None and layer not in dirty_coincidence_layers:
                continue
            # Both grains' layers must be shown: G1's group and each G2 partner.
            points = points[np.isin(partners, tuple(self.state.visible_grain_layers[1]))]
            if boundary_ready:
                first_mask = selected_region_mask(
                    points,
                    self.state.selected_points[0],
                    self.state.selected_points[1],
                    states[0],
                    states[1],
                )
                second_mask = selected_region_mask(
                    points,
                    self.state.selected_points[0],
                    self.state.selected_points[1],
                    states[2],
                    states[3],
                )
                visible_points = points[first_mask & second_mask]
            else:
                visible_points = points
            if layer not in self.state.visible_grain_layers[0]:
                visible_points = np.empty((0, 2))
            self.plot._set_scatter(item, visible_points)
        self._refresh_visible_counts()
        self._queue_manual_count()

    def _refresh_view_display(self) -> None:
        self.plot._update_marker_sizes()
        self._refresh_visible_counts()

    def _refresh_visible_counts(self) -> None:
        if self.view_counts_timer.isActive():
            self.view_counts_timer.stop()
            self.plot._update_marker_sizes()
        self.plot._update_local_overlay()
        if len(self.state.grains) != 2 or len(self.state.visible_atom_masks) != 2:
            return
        x_min, x_max, y_min, y_max = self.plot._view_range()

        def count_in_view(item) -> int:
            # Scatter data already includes rotation and layer/region filters.
            # Reuse those coordinates instead of rotating and filtering all
            # buffered model points again on every mouse movement.
            x, y = item.getData()
            return int(
                np.count_nonzero(
                    (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
                )
            )

        for index, items in enumerate(self.plot.grain_layer_items):
            self.state.visible_atom_counts[index] = sum(
                count_in_view(item) for item in items
            )
        for layer, item in enumerate(self.plot.coincidence_items):
            self.state.visible_coincidence_counts[layer] = count_in_view(item)
        self._update_status()

    def _update_title(self) -> None:
        self.plot._update_common_cell()
        preset = matching_angle_preset(self.state.angle_deg, lattice=self.state.crystal)
        suffix = f" · {preset.name}" if preset is not None else ""
        if self.state.manual_strain_fit is not None:
            fit = self.state.manual_strain_fit
            suffix = f"<br/>SELECTED-CELL BULK STRAIN · {100*self.state.near_cell.max_strain:.6f}% · ΔR₁/ΔR₂ = {fit.rotations_deg[0]:+.4f}°/{fit.rotations_deg[1]:+.4f}°"
        elif self.state.near_cell is not None:
            suffix = (
                f" · STRAINED near-CSL · {100*self.state.near_cell.max_strain:.3f}%"
            )
        elif self.local_active:
            suffix += f" · LOCAL near-CSL · d ≤ {self.controls.local_distance_spin.value():.4f} a₀"
            if not self._has_displacements() and not self._custom_position_base():
                suffix += " · unstrained"
        if self._has_displacements():
            suffix += " · MANUAL ATOM MOVES"
        if self.state.display_rotation_deg:
            suffix += f" · display rotation {self.state.display_rotation_deg:.1f}°"
        angle_label = "reference θ" if self.state.manual_strain_fit is not None else "θ"
        kind = ("tilt dichromatic pattern" if self.state.crystal.homophase
                else "interface dichromatic pattern")
        self.plot._set_title(
            f"<span style='font-size:15px;color:#17212b'>"
            f"{html.escape(self.state.crystal.label)} {kind} · {angle_label} = {self.state.angle_deg:.2f}°"
            f"{suffix}</span>"
        )
        prefix = "Reference" if self.state.manual_strain_fit is not None else "Exact"
        exact = f"{prefix} θ = {self.state.angle_deg:.8f}°"
        if isinstance(preset, CSLPreset):
            tail = ", ".join(
                str(int(preset.n * index)) for index in self.state.geometry.axis_indices
            )
            exact += f"  ·  quaternion ({preset.m}, {tail})"
        self.controls.angle_exact_label.setText(exact)

    def _queue_angle_update(self, angle_deg: float) -> None:
        angle_deg = float(np.clip(angle_deg, 0.0, self.state.angle_range.maximum_deg))
        preset = matching_angle_preset(float(angle_deg), lattice=self.state.crystal)
        self.state.pending_angle = (
            preset.angle_deg if preset is not None else float(angle_deg)
        )
        self._sync_angle_controls(self.state.pending_angle)
        if abs(self.state.pending_angle - self.state.angle_deg) > 1.0e-12:
            cleared_moves = self._has_displacements() or self._custom_position_base()
            if cleared_moves:
                self.state.grain_displacements = [[], []]
                self._clear_translation_selection()
                self._set_translation_offset([0, 0])
                self._clear_near_cell(reset_custom=True)
                self._sync_translation_controls("Manual moves cleared by the angle change.")
            if self.state.near_enabled:
                if self.compute.near_search is not None:
                    self.compute.near_search.cancel()
                self.near_debounce_timer.stop()
                self._clear_near_cell()
            self._cancel_parallel_work()
            if not self.state.angle_update_active:
                self._clear_lattice_selections()
            if cleared_moves:
                self._sync_translation_controls("Manual moves cleared by the angle change.")
            self.state.angle_update_active = True
        # Throttle previews rather than debouncing them: while the handle is
        # moving, the newest orientation is rendered at most once per frame.
        if not self.angle_preview_timer.isActive():
            self.angle_preview_timer.start()
        self.coincidence_timer.start()

    def _apply_pending_angle(self) -> None:
        if abs(self.state.pending_angle - self.state.angle_deg) <= 1.0e-12:
            return
        self.state.angle_deg = self.state.pending_angle
        self._start_parallel_regeneration(compute_coincidences=False)

    def _finish_angle_update(self) -> None:
        self.coincidence_timer.stop()
        self.angle_preview_timer.stop()
        self._apply_pending_angle()
        self._start_parallel_coincidences()
        self._update_title()
        if self.state.near_enabled:
            self._queue_near_search()

    def _on_preset_changed(self, combo_index: int) -> None:
        preset_index = self.controls.preset_combo.itemData(combo_index)
        if preset_index is None:
            return
        self._cancel_parallel_work()
        preset = self.state.presets[int(preset_index)]
        if self.state.near_enabled:
            if self.compute.near_search is not None:
                self.compute.near_search.cancel()
            self._clear_near_cell()
        self.coincidence_timer.stop()
        self.state.pending_angle = preset.angle_deg
        self.state.angle_update_active = True
        self._clear_lattice_selections()
        self._sync_angle_controls(preset.angle_deg)
        self._finish_angle_update()

    def _sync_angle_controls(self, angle: float | None = None) -> None:
        angle = self.state.angle_deg if angle is None else angle
        slider_blocker = QtCore.QSignalBlocker(self.controls.angle_slider)
        spin_blocker = QtCore.QSignalBlocker(self.controls.angle_spin)
        combo_blocker = QtCore.QSignalBlocker(self.controls.preset_combo)
        bounds = self.state.angle_range
        self.controls.angle_slider.setRange(0, round(bounds.maximum_deg * 100))
        self.controls.angle_spin.setRange(0.0, bounds.maximum_deg)
        self.controls.angle_range_label.setText(f"Allowed range: 0–{bounds.maximum_deg:g}°")
        if bounds.symmetry_order is None:
            explanation = "Unknown crystal symmetry; using the general 0–180° range."
        elif self.state.crystal.homophase:
            explanation = (
                f"{self.state.geometry.axis_label}: {bounds.symmetry_order}-fold axial symmetry; "
                f"rotation period {bounds.period_deg:g}°. Exchanging the grains gives "
                f"the fixed-axis range 0–{bounds.maximum_deg:g}°."
            )
        else:
            explanation = (
                "Both grains' axial symmetries give a rotation period of "
                f"{bounds.period_deg:g}°. "
                + ("A two-fold axis shared by both grains identifies ±θ"
                   if bounds.sign_equivalent else
                   "No shared two-fold axis identifies ±θ, and the grains cannot be exchanged")
                + f"; range 0–{bounds.maximum_deg:g}°."
            )
        for widget in (self.controls.angle_spin, self.controls.angle_slider,
                       self.controls.angle_range_label):
            widget.setToolTip(explanation)
        self.controls.angle_slider.setValue(round(angle * 100.0))
        self.controls.angle_spin.setValue(angle)
        preset = matching_angle_preset(angle, lattice=self.state.crystal)
        combo_index = 0
        if preset is not None:
            combo_index = self.state.presets.index(preset) + 1
        self.controls.preset_combo.setCurrentIndex(combo_index)
        del slider_blocker, spin_blocker, combo_blocker

    def _sync_context_controls(self) -> None:
        if hasattr(self.controls, "gb_region_widget"):
            self.controls.gb_region_widget.setVisible(
                len(self.state.selected_points) == 2
            )
        if hasattr(self.controls, "vector_options_widget"):
            self.controls.vector_options_widget.setVisible(
                self.state.interaction_mode == "vector"
                or bool(self.state.selected_atoms)
            )

    def _set_mode(self, mode: str) -> None:
        if mode != "translate" and len(self._translation_corners) == 1:
            self._clear_translation_selection()
        self.state.interaction_mode = mode
        gb_blocker = QtCore.QSignalBlocker(self.controls.pick_gb_button)
        vector_blocker = QtCore.QSignalBlocker(self.controls.vector_button)
        self.controls.pick_gb_button.setChecked(mode == "boundary")
        self.controls.vector_button.setChecked(mode == "vector")
        with QtCore.QSignalBlocker(self.controls.translation_select_button):
            self.controls.translation_select_button.setChecked(mode == "translate")
        with QtCore.QSignalBlocker(self.controls.manual_pick_button):
            self.controls.manual_pick_button.setChecked(mode == "cell")
        del gb_blocker, vector_blocker
        cursor = (
            QtCore.Qt.CursorShape.CrossCursor
            if mode != "idle"
            else QtCore.Qt.CursorShape.OpenHandCursor
        )
        self.plot.plot_widget.viewport().setCursor(cursor)
        self._sync_context_controls()
        self._sync_translation_controls()
        self._update_status()

    def _start_new_boundary(self, *_args) -> None:
        self.state.selected_points.clear()
        self.plot._set_scatter(self.plot.boundary_endpoint_item, np.empty((0, 2)))
        self.plot.boundary_item.setData([], [])
        for item in (*self.plot.boundary_labels, *self.plot.side_labels):
            item.hide()
        self._set_mode("boundary")
        self._update_visible_points()

    def _start_vector_measurement(self, *_args) -> None:
        self.state.selected_atoms.clear()
        self.plot._set_scatter(self.plot.vector_endpoint_item, np.empty((0, 2)))
        self.plot.vector_item.setData([], [])
        self.plot.vector_arrow.hide()
        self.plot.vector_annotation.hide()
        for item in self.plot.vector_labels:
            item.hide()
        self._set_mode("vector")

    def _clear_lattice_selections(self, *, reset_translation_undo=True) -> None:
        if reset_translation_undo:
            self._translation_undo = [[], []]
            self._on_translation_components_changed()
        self._clear_translation_selection(sync_offset=reset_translation_undo)
        self._clear_manual_cell()
        self.state.selected_points.clear()
        self.state.selected_atoms.clear()
        self.plot.boundary_item.setData([], [])
        self.plot.vector_item.setData([], [])
        self.plot._set_scatter(self.plot.boundary_endpoint_item, np.empty((0, 2)))
        self.plot._set_scatter(self.plot.vector_endpoint_item, np.empty((0, 2)))
        self.plot.vector_arrow.hide()
        for item in (
            *self.plot.boundary_labels,
            *self.plot.vector_labels,
            *self.plot.side_labels,
            self.plot.vector_annotation,
        ):
            item.hide()
        self._set_mode("idle")

    def _handle_view_click(self, position: np.ndarray) -> None:
        if self.state.interaction_mode == "idle":
            return
        if self.state.grain_signature != self._geometry_signature():
            self._update_status(
                "Lattice positions are updating; pick again when the new atoms appear."
            )
            return
        position = self.plot._from_view(position)
        if self.state.interaction_mode == "translate":
            self._pick_translation_corner(position)
            return
        if self.state.interaction_mode == "cell":
            self._pick_manual_vertex(position)
            return
        atom, pixel_distance = self._nearest_atom(position)
        if atom is None or pixel_distance > 13.0:
            self._update_status("No atom nearby; the current selection is preserved.")
            return
        if self.state.interaction_mode == "boundary":
            self._pick_boundary_atom(atom)
        else:
            self._pick_vector_atom(atom)

    def _on_scene_mouse_click(self, event) -> None:
        """Route a true click to picking; drag gestures never reach this slot."""

        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        point = self.plot.view_box.mapSceneToView(event.scenePos())
        self._handle_view_click(np.array([point.x(), point.y()], dtype=float))

    def _nearest_atom(self, position: np.ndarray) -> tuple[SelectedAtom | None, float]:
        if self.state.grain_signature != self._geometry_signature():
            return None, float("inf")
        x_min, x_max, y_min, y_max = self.plot._view_range()
        scene_rect = self.plot.view_box.sceneBoundingRect()
        pixels_per_x = scene_rect.width() / max(x_max - x_min, 1.0e-12)
        pixels_per_y = scene_rect.height() / max(y_max - y_min, 1.0e-12)
        best_atom = None
        best_distance_squared = np.inf
        for grain_index, (grain, visible_mask) in enumerate(
            zip(self.state.grains, self.state.visible_atom_masks, strict=True)
        ):
            candidate_indices = np.flatnonzero(visible_mask)
            if not len(candidate_indices):
                continue
            candidate_points = grain.positions[candidate_indices]
            deltas = self.plot._to_view(candidate_points - position)
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
            self.state.selected_points
            and np.linalg.norm(atom.position - self.state.selected_points[0]) < 1.0e-8
        ):
            self._update_status("Choose a different second atom.")
            return
        self.state.selected_points.append(atom.position.copy())
        self.plot._set_scatter(
            self.plot.boundary_endpoint_item, np.asarray(self.state.selected_points)
        )
        index = len(self.state.selected_points) - 1
        label = self.plot.boundary_labels[index]
        label.setText(f"B{index + 1}  {self._atom_name(atom)}")
        label.setPos(*self.plot._to_view(atom.position))
        label.show()
        if len(self.state.selected_points) == 2:
            self._set_mode("idle")
            self.plot._draw_boundary()
            self._update_visible_points()
        else:
            self._update_status()

    def _pick_vector_atom(self, atom: SelectedAtom) -> None:
        if (
            self.state.selected_atoms
            and np.linalg.norm(atom.position - self.state.selected_atoms[0].position)
            < 1.0e-8
        ):
            self._update_status("Choose a different second atom.")
            return
        self.state.selected_atoms.append(atom)
        points = np.asarray(
            [selected.position for selected in self.state.selected_atoms]
        )
        self.plot._set_scatter(self.plot.vector_endpoint_item, points)
        index = len(self.state.selected_atoms) - 1
        label = self.plot.vector_labels[index]
        label.setText(f"P{index + 1}  {self._atom_name(atom)}")
        label.setPos(*self.plot._to_view(atom.position))
        label.show()
        if len(self.state.selected_atoms) == 2:
            self._set_mode("idle")
            self.plot._draw_vector()
        self._update_status()

    def _request_vector_annotation_repaint(self) -> None:
        """Schedule a full viewport paint after revealing the vector readout.

        TextItem is a parent graphics item with its text stored in a child item.
        When its text, position, and visibility all change inside the scene's
        mouse-release callback, Qt can occasionally retain only the old dirty
        region.  A later zoom repaints the whole viewport and makes the already
        visible annotation appear.  Updating the viewport here provides that
        paint immediately after the click event returns.
        """

        self.plot.vector_annotation.update()
        self.plot.plot_widget.scene().update()
        self.plot.plot_widget.viewport().update()
        # Run once after the mouse-release callback. This prevents the newly
        # shown child text item from being lost in Qt's stale dirty region.
        QtCore.QTimer.singleShot(0, self._repaint_vector_annotation)

    def _repaint_vector_annotation(self) -> None:
        if self.plot.vector_annotation.isVisible():
            self.plot.vector_annotation.update()
            self.plot.plot_widget.viewport().repaint()

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
        if len(self.state.selected_atoms) != 2:
            return ""
        first, second = self.state.selected_atoms
        cross_grain = first.grain_index != second.grain_index
        displacement = self._selected_vector_displacement()
        vectors = self._selected_crystal_vectors()
        lines = [
            f"{'Cross-grain' if cross_grain else 'Crystal'} P1→P2 · "
            f"{self._atom_name(first)} → {self._atom_name(second)} · axial {self.state.axial_repeat:+d}"
        ]
        for grain, vector in vectors:
            # Each grain's components use its own lattice constant.
            unit = "a₀" if grain == 0 or self.state.crystal.scale == 1.0 else "a₀(G2)"
            lines.append(
                f"G{grain+1} current (polar): {format_direction_components(vector.current, unit)}"
            )
        # Material coordinates can stay integer while current spatial
        # components change under strain. Keep these notions explicitly apart.
        if any(vector.strained for _, vector in vectors):
            for grain, vector in vectors:
                lines.append(
                    f"G{grain+1} lattice [uvw]: {format_direction_components(vector.lattice, unit='')}"
                )
        lines.append(f"Current |Δr|/a₀ = {np.linalg.norm(displacement):.4f}")
        if cross_grain:
            projected = self.plot._to_view(displacement[:2])
            lines.append(f"View Δxy/a₀ = ({projected[0]:.4f}, {projected[1]:.4f})")
        return "\n".join(lines)

    def _selected_vector_displacement(self):
        """Actual 3D displacement, including phase height and P2's periodic image."""
        first, second = self.state.selected_atoms
        # Both grains share the tilt axis, even after their in-plane F and
        # translations. Only z may be obtained from reference indices (each
        # grain's own layer heights); x/y must come from actual atom positions.
        dz = axial_displacement(
            self.state.crystal, first.grain_index, first.half_indices,
            second.grain_index, second.half_indices, self.state.axial_repeat,
        )
        return np.r_[second.position - first.position, dz]

    def _selected_crystal_vectors(self):
        if len(self.state.selected_atoms) != 2:
            return []
        first, second = self.state.selected_atoms
        grains = (
            (0, 1) if first.grain_index != second.grain_index else (first.grain_index,)
        )
        displacement = self._selected_vector_displacement()
        return [
            (
                grain,
                crystal_vector_coordinates(
                    displacement,
                    (1 if grain == 0 else -1) * self.state.angle_deg / 2,
                    self.state.deformations[grain],
                    self.state.crystal.lattices[grain],
                    self.state.crystal.axes[grain],
                    scale=self.state.crystal.scales[grain],
                ),
            )
            for grain in grains
        ]

    def _set_region_states(self, states: tuple[bool, bool, bool, bool]) -> None:
        blockers = [
            QtCore.QSignalBlocker(check) for check in self.controls.region_checks
        ]
        for check, state in zip(self.controls.region_checks, states, strict=True):
            check.setChecked(state)
        del blockers
        self._update_visible_points()

    def _reset_view(self, *_args) -> None:
        _center, width, height = self.plot._view_geometry()
        self.plot.view_box.setRange(
            xRange=(-0.5 * width, 0.5 * width),
            yRange=(-0.5 * height, 0.5 * height),
            padding=0.0,
        )
        self.view_refresh_timer.start(0)

    def _on_axial_repeat_changed(self, value: int) -> None:
        self.state.axial_repeat = int(value)
        self.controls.axial_spin.setPrefix("+" if value >= 0 else "")
        self.plot._draw_vector()
        self._update_status()

    def _update_status(self, message: str | None = None) -> None:
        if self.state.render_error is not None:
            message = self.state.render_error
        if message is None:
            if self.state.interaction_mode == "boundary":
                message = (
                    "GB mode · click B1"
                    if not self.state.selected_points
                    else "B1 selected · click B2"
                )
            elif self.state.interaction_mode == "cell":
                layer = (
                    f" · {self._cell_layer_label(self.state.manual_vertices[0].layer)}"
                    if self.state.manual_vertices
                    else ""
                )
                message = f"Manual cell · pick C{len(self.state.manual_vertices)+1}/4 around the perimeter{layer}"
            elif self.state.interaction_mode == "vector":
                message = (
                    "Vector mode · click P1"
                    if not self.state.selected_atoms
                    else "P1 selected · click P2"
                )
            elif self.state.interaction_mode == "translate":
                message = f"Translate region · click corner {len(self._translation_corners) + 1}/2"
            else:
                message = "Ready · choose an interaction tool"
        homophase = self.state.crystal.homophase
        visible_status_layers = sorted(
            self.state.visible_layers if homophase else self.state.visible_grain_layers[0])
        csl_state = (
            "updating…"
            if self.state.csl_updating
            else f"{sum(self.state.visible_coincidence_counts)} total · "
            + ", ".join(
                f"{layer_name(layer)}: {count}"
                for layer, count in (
                    (layer, self.state.visible_coincidence_counts[layer])
                    for layer in visible_status_layers[:6]
                )
            )
            + (", …" if len(visible_status_layers) > 6 else "")
        )
        compute_state = (
            f"{self.compute.worker_count} workers · {self.compute.parallel_stage} running"
            if self.compute.parallel_stage is not None
            else f"{self.compute.worker_count} worker{'s' if self.compute.worker_count != 1 else ''}"
        )
        self.controls.status_label.setText(
            "<div style='font-weight:700;color:#17212b'>"
            f"{html.escape(message)}</div>"
            "<div style='margin-top:7px;color:#526174'>Drag to pan · wheel to zoom<br>"
            "Selections stay active while navigating</div>"
            "<div style='margin-top:7px;color:#526174'>"
            f"Visible G1 / G2: {self.state.visible_atom_counts[0]} / {self.state.visible_atom_counts[1]}<br>"
            f"{'Same-layer CSL' if homophase else 'Coincident columns (by G1 layer)'}: {csl_state}<br>"
            f"Compute: {compute_state}</div>"
        )

    def save_session(self, path) -> None:
        self.session.save(path)

    def load_session(self, path) -> None:
        self.session.load(path)

    def _choose_session_save_path(self) -> None:
        self.session.choose_save_path()

    def _choose_session_import_path(self) -> None:
        self.session.choose_import_path()

    def _choose_export_path(self) -> None:
        clean = self.controls.clean_export_check.isChecked()
        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export atoms only" if clean else "Export dichromatic pattern",
            "dichromatic_atoms.png" if clean else "dichromatic_pattern.png",
            "PNG image (*.png)",
        )
        if filename:
            try:
                self.plot.save(
                    Path(filename), clean=clean,
                    transparent=not self.controls.export_background_check.isChecked(),
                )
            except Exception as error:
                QtWidgets.QMessageBox.warning(self, "PNG export failed", str(error))

    def _choose_svg_export_path(self) -> None:
        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export atoms as vectors", "dichromatic_atoms.svg", "SVG image (*.svg)",
        )
        if filename:
            try:
                self.plot.save_svg(
                    Path(filename),
                    transparent=not self.controls.export_background_check.isChecked(),
                )
            except Exception as error:
                QtWidgets.QMessageBox.warning(self, "SVG export failed", str(error))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._show_close_citation:
            # Disable first so a nested or repeated close request cannot show
            # a second modal dialog while this one is active.
            self._show_close_citation = False
            message = QtWidgets.QMessageBox(self)
            message.setWindowTitle("Thank you for using DichromaticMap")
            message.setIcon(QtWidgets.QMessageBox.Icon.Information)
            message.setTextFormat(QtCore.Qt.TextFormat.RichText)
            message.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextBrowserInteraction
            )
            message.setText(
                "<p>Thank you for using DichromaticMap.</p>"
                "<p>If you find it helpful, please cite:</p>"
                f"<p>{html.escape(CITATION_TEXT)}<br>"
                f'<a href="{CITATION_DOI}">{CITATION_DOI}</a></p>'
            )
            message.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            for label in message.findChildren(QtWidgets.QLabel):
                label.setOpenExternalLinks(True)
            message.exec()
        self.manual_count_timer.stop()
        self.display_rotation_timer.stop()
        self.angle_preview_timer.stop()
        self.coincidence_timer.stop()
        self.view_refresh_timer.stop()
        self.view_counts_timer.stop()
        self.near_debounce_timer.stop()
        self.near_poll_timer.stop()
        self.parallel_poll_timer.stop()
        self.session.close()
        self.compute.close()
        super().closeEvent(event)
