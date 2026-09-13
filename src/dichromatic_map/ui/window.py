"""Window composition and interactive workflows.

State, workers, controls and drawing each have a dedicated owner. The legacy
window attributes below are properties onto those owners for API compatibility.
"""

from __future__ import annotations
import html
import os
from pathlib import Path
from dataclasses import replace
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from . import LAYER_SYMBOLS
from ..crystal import (
    get_geometry,
    projected_columns,
    layer_name,
    GeometryLimitError,
    csl_presets,
    matching_csl_preset,
    crystal_vector_coordinates,
    format_direction_components,
)
from ..cells import validate_cell_vertices, selected_region_mask
from ..matching import exact_csl_cell, LocalPairs, COINCIDENCE_TOLERANCE_FACTOR
from ..state import (
    PatternParameters,
    SelectedAtom,
    CellVertex,
    default_angle_deg,
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
)
from ..matching import same_layer_coincidence_sites
from .controls import ControlDock
from .plot import PatternPlot


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
        self, parameters: PatternParameters, worker_count: int | None = None
    ) -> None:
        super().__init__()
        self.state = PatternState(parameters)
        self.compute = ComputeSession()
        self.plot = PatternPlot(self)
        self.controls = ControlDock(self)
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
        self._regenerate_buffer(compute_coincidences=True)
        self.view_refresh_timer.stop()
        self._set_mode("idle")

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

    def _on_display_rotation(self, angle):
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
        if any(vertex.source == "local" for vertex in self.state.manual_vertices) and (
            not self.local_active
            or self.state.manual_local_cutoff
            != self.controls.local_distance_spin.value()
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
        ready = (
            self.local_active
            and len(self.state.manual_vertices) == 4
            and any(vertex.source == "local" for vertex in self.state.manual_vertices)
        )
        self.controls.manual_strain_button.setEnabled(active or ready)
        self.controls.manual_strain_limit.setEnabled(ready and not active)
        self.controls.manual_rotation_limit.setEnabled(ready and not active)
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
        if not active and len(self.state.manual_vertices) < 4:
            self.controls.manual_strain_note.setText(
                "Select a local near-CSL cell first. Both grains share the deformation; strain and small rotations have separate limits."
            )

    def _toggle_manual_strain(self, *_args):
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
            and any(vertex.source == "local" for vertex in self.state.manual_vertices)
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
                self.state.geometry.lattice,
                self.state.geometry.axis,
                self.state.manual_vertices[0].layer,
                self.controls.manual_rotation_limit.value(),
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
            CellVertex(pair.mean(axis=0), vertex.layer, "CSL", pair.copy())
            for vertex, pair in zip(vertices, fit.vertices.transpose(1, 0, 2))
        ]
        self._clear_local_pairs()
        self._sync_near_controls()
        self.plot._draw_manual_cell()
        self.controls.manual_strain_note.setText(
            f"Applied: max principal strain {100*fit.cell.max_strain:.6f}%; "
            f"rotations G1 {fit.rotations_deg[0]:+.6f}°, G2 {fit.rotations_deg[1]:+.6f}°; "
            f"four-pair residual {fit.residual:.2e} a₀. Full-lattice CSL is recomputed, layer by layer."
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
            for layer, points in enumerate(self.state.coincident_points):
                if layer not in self.state.visible_layers:
                    continue
                keep = np.ones(len(points), dtype=bool)
                if len(self.state.selected_points) == 2:
                    for grain in (0, 1):
                        keep &= selected_region_mask(
                            points,
                            *self.state.selected_points,
                            states[2 * grain],
                            states[2 * grain + 1],
                        )
                candidates.extend(
                    CellVertex(point.copy(), layer, "CSL") for point in points[keep]
                )
        if self.local_active and not self.state.local_updating:
            keep = self._local_pair_mask()
            candidates.extend(
                CellVertex(point.copy(), int(layer), "local", np.stack((first, second)))
                for point, layer, first, second in zip(
                    self.state.local_pairs.midpoints[keep],
                    self.state.local_pairs.layers[keep],
                    self.state.local_pairs.first[keep],
                    self.state.local_pairs.second[keep],
                )
            )
        return candidates

    @staticmethod
    def _cell_layer_label(layer):
        symbol = LAYER_SYMBOLS[layer % len(LAYER_SYMBOLS)]
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
        }.get(symbol, symbol)
        return f"{glyph} layer"

    def _resolve_cell_vertex(self, candidate):
        if candidate.grain_positions is not None:
            return candidate
        # Exact CSL stores marker centers; resolve its two real atoms once at
        # selection time, not later from a possibly panned-away drawing buffer.
        endpoints = []
        for grain in self.state.grains:
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
            candidates[i].layer
            for i in np.flatnonzero(np.abs(distance - distance[index]) < 1e-6)
        }
        if len(hit_layers) > 1:
            self._manual_pick_warning(
                "Overlapping markers from different layers; isolate one layer in the Layers tab, then pick again. Point not selected."
            )
            return
        if (
            self.state.manual_vertices
            and candidate.layer != self.state.manual_vertices[0].layer
        ):
            self._manual_pick_warning(
                f"Wrong layer/symbol: expected {self._cell_layer_label(self.state.manual_vertices[0].layer)}, "
                f"clicked {self._cell_layer_label(candidate.layer)}. Point not selected."
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
                f"Selected {len(trial)}/4 vertices in {self._cell_layer_label(candidate.layer)}. "
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
        layer = self.state.manual_vertices[0].layer
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
            self.state.geometry.lattice,
            self.state.geometry.axis,
            boundary,
            states,
            layer,
            self.state.translations.copy(),
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
        layer = self.state.manual_vertices[0].layer
        lines = [
            (
                "Selected-cell bulk strain; common translations verified."
                if self.state.manual_strain_fit is not None
                else "Manual region; periodicity not verified."
            ),
            f"Only the picked {self._cell_layer_label(layer)} is counted; not an all-layer total.",
            "Each grain uses its own four current atom vertices.",
            (
                "GB side visibility applied."
                if self.controls.manual_visible_check.isChecked()
                else "Both grains; independent of view and display-layer filter."
            ),
        ]
        for grain in (0, 1):
            interior = counts.interior[grain, layer]
            boundary = counts.boundary[grain, layer]
            lines.append(
                f"G{grain+1} ({'blue' if grain == 0 else 'red'}): area {counts.areas[grain]:.6g} a₀²"
            )
            if counts.half_open_available[grain]:
                total = counts.half_open[grain, layer]
                edge = counts.half_open_edges[grain, layer]
                corner = counts.half_open_corners[grain, layer]
                lines.append(
                    f"  {self._cell_layer_label(layer)}: {total} atoms = "
                    f"{interior} inside + {edge} edge + {corner} corner"
                )
                lines.append(
                    f"  Closed outline would show {interior+boundary} point instances "
                    f"({boundary} on the boundary); opposite-edge copies are duplicates."
                )
            else:
                lines.append(
                    f"  {self._cell_layer_label(layer)}: {interior+boundary} atoms "
                    f"in the closed polygon = {interior} inside + {boundary} boundary"
                )
                lines.append(
                    "  The selected quadrilateral is not a parallelogram, so no "
                    "periodic unique-cell count is defined."
                )
        lines.append("G1/G2 counted separately; overlapping atoms are not merged.")
        self.controls.manual_info.setPlainText("\n".join(lines))
        self.plot._draw_manual_cell()

    def _fit_manual_cell(self, *_args):
        if len(self.state.manual_vertices) == 4:
            self.plot._fit_model_corners(self._manual_grain_polygons().reshape(-1, 2))

    def _empty_coincidences(self):
        return tuple(np.empty((0, 2)) for _ in range(self.state.geometry.layer_count))

    def _set_all_layers_visible(self, visible: bool) -> None:
        layers = set(range(self.state.geometry.layer_count)) if visible else set()
        self.state.visible_grain_layers = [set(layers), set(layers)]
        self._sync_shared_visible_layers()
        for checks in self.controls.grain_layer_checks:
            for check in checks:
                with QtCore.QSignalBlocker(check):
                    check.setChecked(visible)
        self._sync_compatibility_layer_checks()
        self.state.selected_layer = -1
        with QtCore.QSignalBlocker(self.controls.layer_combo):
            self.controls.layer_combo.setCurrentIndex(0)
        self._apply_layer_visibility()

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
        self._apply_layer_visibility()

    def _on_axial_layer_toggled(self, layer: int, visible: bool) -> None:
        for grain in (0, 1):
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
        self._apply_layer_visibility()

    def _apply_layer_visibility(self) -> None:
        self.plot._rebuild_legend()
        self._update_visible_points()

    def _on_layer_changed(self, *_args):
        self.state.selected_layer = self.controls.layer_combo.currentData()
        layers = (
            set(range(self.state.geometry.layer_count))
            if self.state.selected_layer < 0
            else {self.state.selected_layer}
        )
        self.state.visible_grain_layers = [set(layers), set(layers)]
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
            self.state.geometry.lattice,
            self.state.geometry.axis,
            self.state.angle_deg,
            tuple(self.state.deformations[0].ravel()),
            tuple(self.state.deformations[1].ravel()),
            tuple(self.state.translations.ravel()),
        )

    def _update_geometry_labels(self):
        model = self.state.geometry
        name = f"{model.lattice} ⟨{model.axis}⟩ Tilt GB"
        self.setWindowTitle(name + " · Dichromatic Pattern")
        self.controls.heading.setText(
            f"<div style='font-size:17px;font-weight:700;color:#17212b'>{name}</div>"
            "<div style='color:#64748b;margin-top:3px'>Coordinates in a₀ · 1 = one lattice constant</div>"
        )
        self.plot.plot_item.setLabel("bottom", f"{model.x_label} / a₀", units="")
        self.plot.plot_item.setLabel("left", f"{model.y_label} / a₀", units="")
        self.controls.layer_info.setText(
            f"{model.layer_count} axial layers per grain · spacing = {model.layer_spacing:.6g} a₀\n"
            f"Axial repeat = {model.axial_period:.6g} a₀"
            + (
                "\nLegend shows the first 6 enabled layers; use the grain/layer checkboxes to inspect others."
                if model.layer_count > 6
                else ""
            )
            + "\nCSL and near-pair markers require the layer to be visible in both grains."
        )
        self.controls.axial_spin.setToolTip(
            "Vector readout only: P2 shifts by "
            + self._format_lattice_vector(model.axial_repeat_half_indices)
            + f" per step along {model.axis_label}. This selects a periodic atom "
            "represented by the same projected column; it does not add thickness, "
            "duplicate the structure, or change plotted atoms."
        )

    def _on_geometry_changed(self, *_args, apply_custom=False):
        lattice = self.controls.structure_combo.currentData()
        axis = self.controls.axis_combo.currentData()
        self.controls.custom_axis_row.setVisible(axis is None)
        if axis is None:
            if not apply_custom:
                return
            axis = self.controls.custom_axis_edit.text()
        try:
            geometry = get_geometry(lattice, axis)
        except ValueError as error:
            self.controls.axis_error.setText(str(error))
            self.controls.axis_error.show()
            with QtCore.QSignalBlocker(self.controls.structure_combo):
                self.controls.structure_combo.setCurrentIndex(
                    self.controls.structure_combo.findData(self.state.geometry.lattice)
                )
            return
        self.controls.axis_error.clear()
        self.controls.axis_error.hide()
        axis = geometry.axis
        if (lattice, axis) == (self.state.geometry.lattice, self.state.geometry.axis):
            return
        axis_changed = axis != self.state.geometry.axis
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
        self.state.geometry = geometry
        self.state.render_error = None
        self.state.parameters = replace(
            self.state.parameters, lattice=lattice, axis=axis
        )
        self.state.presets = csl_presets(axis)
        if axis_changed:
            self.state.angle_deg = default_angle_deg(axis)
        self.state.pending_angle = self.state.angle_deg
        self.state.angle_update_active = False
        self._clear_near_cell()
        with QtCore.QSignalBlocker(self.controls.preset_combo):
            self.controls.preset_combo.clear()
            self.controls.preset_combo.addItem("Custom angle", None)
            for index, preset in enumerate(self.state.presets):
                self.controls.preset_combo.addItem(preset.label, index)
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
        self.state.visible_coincidence_counts = [0] * geometry.layer_count
        self.state.coincident_points = self._empty_coincidences()
        self.state.selected_layer = -1
        all_layers = set(range(geometry.layer_count))
        self.state.visible_grain_layers = [set(all_layers), set(all_layers)]
        self.state.visible_layers = set(all_layers)
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
        self._update_status(
            f"Changed to {lattice} {self.state.geometry.axis_label}; previous selections cleared."
        )

    @property
    def common_cell(self):
        return (
            self.state.near_cell
            if self.state.near_cell is not None
            else exact_csl_cell(
                self.state.angle_deg,
                lattice=self.state.geometry.lattice,
                axis=self.state.geometry.axis,
            )
        )

    def _clear_near_cell(self):
        if self.state.near_cell is not None:
            self._clear_lattice_selections()
        self.state.near_cell = None
        self.state.near_solutions = []
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
        strain_method = self.state.near_method == "strain"
        strained = self.state.near_enabled and strain_method
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
                "Off. Original unstrained lattices restored."
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
        if self.compute.near_search is None:
            self.compute.near_search = NearSearch(
                self.compute.worker_count, executor=self.compute.executor
            )
        self.compute.near_search.request(
            self.state.angle_deg,
            self.controls.strain_spin.value(),
            self.controls.search_index_spin.value(),
            lattice=self.state.geometry.lattice,
            axis=self.state.geometry.axis,
        )
        self.near_poll_timer.start()

    def _poll_near_search(self):
        if (
            self.compute.near_search is None
            or not self.state.near_enabled
            or self.state.near_method != "strain"
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
        if self.compute.parallel_stage in ("grains", "coincidences"):
            # Exact detection will start matching after the new atoms arrive.
            if self.compute.parallel_stage == "grains":
                self.compute.parallel_payload["compute_coincidences"] = True
            return
        if self.state.grain_signature != self._geometry_signature():
            return
        self.near_debounce_timer.stop()
        self._cancel_parallel_work()
        self._clear_local_pairs()
        executor = self.compute.background_executor()
        groups = np.array_split(
            np.arange(self.state.geometry.layer_count),
            min(self.compute.worker_count, self.state.geometry.layer_count),
        )
        distance = self.controls.local_distance_spin.value()
        self.compute.parallel_stage = "local"
        self.compute.parallel_payload = {
            "geometry_signature": self.state.grain_signature,
            "distance": distance,
        }
        try:
            self.compute.parallel_futures = [
                executor.submit(
                    local_match_layers_worker,
                    self.state.grains[0],
                    self.state.grains[1],
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
        mask &= np.isin(pairs.layers, tuple(self.state.visible_layers))
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
        self.state.grains = [
            projected_columns(
                width,
                height,
                +half_angle,
                center=tuple(center),
                deformation=self.state.deformations[0],
                lattice=self.state.geometry.lattice,
                axis=self.state.geometry.axis,
                translation=self.state.translations[0],
            ),
            projected_columns(
                width,
                height,
                -half_angle,
                center=tuple(center),
                deformation=self.state.deformations[1],
                lattice=self.state.geometry.lattice,
                axis=self.state.geometry.axis,
                translation=self.state.translations[1],
            ),
        ]
        self.state.buffer_bounds = bounds
        self.state.render_error = None
        self.state.grain_signature = self._geometry_signature()
        if compute_coincidences:
            self._compute_coincidences()
        else:
            self.state.coincident_points = self._empty_coincidences()
            self.state.csl_updating = True
        self._update_visible_points()
        self._update_title()

    def _start_parallel_regeneration(self, compute_coincidences: bool) -> None:
        if self.compute.executor is None:
            self._regenerate_buffer(compute_coincidences)
            return
        self._cancel_parallel_work()
        self._clear_local_pairs()
        center, width, height, bounds = self.plot._buffer_geometry()
        half_angle = 0.5 * self.state.angle_deg
        try:
            self.compute.parallel_futures = [
                self.compute.executor.submit(
                    generate_grain_worker,
                    width,
                    height,
                    sign * half_angle,
                    tuple(center),
                    self.state.deformations[index],
                    self.state.geometry.lattice,
                    self.state.geometry.axis,
                    self.state.translations[index],
                )
                for index, sign in enumerate((1.0, -1.0))
            ]
        except Exception as error:  # pragma: no cover - platform-specific failure
            self._parallel_failure(error)
            return
        self.compute.parallel_payload = {
            "bounds": bounds,
            "compute_coincidences": compute_coincidences,
            "geometry_signature": self._geometry_signature(),
        }
        self.state.coincident_points = self._empty_coincidences()
        self.state.csl_updating = True
        for item in self.plot.coincidence_items:
            self.plot._set_scatter(item, np.empty((0, 2)))
        self.compute.parallel_stage = "grains"
        self._update_status()
        self.parallel_poll_timer.start()

    def _start_parallel_coincidences(self) -> None:
        if self.compute.parallel_stage == "grains":
            self.compute.parallel_payload["compute_coincidences"] = True
            return
        if self.compute.executor is None:
            self._compute_coincidences()
            self.state.angle_update_active = False
            self._update_visible_points()
            return
        if len(self.state.grains) != 2:
            return
        self._cancel_parallel_work()
        tolerance = COINCIDENCE_TOLERANCE_FACTOR
        layer_groups = np.array_split(
            np.arange(self.state.geometry.layer_count),
            min(self.compute.worker_count, self.state.geometry.layer_count),
        )
        try:
            self.compute.parallel_futures = [
                self.compute.executor.submit(
                    coincidence_layers_worker,
                    self.state.grains[0],
                    self.state.grains[1],
                    tolerance,
                    layers,
                )
                for layers in layer_groups
            ]
        except Exception as error:  # pragma: no cover - platform-specific failure
            self._parallel_failure(error)
            return
        self.compute.parallel_stage = "coincidences"
        self.state.csl_updating = True
        self.parallel_poll_timer.start()
        self._update_status()

    def _poll_parallel_work(self) -> None:
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
            by_layer = {
                layer: points for _pid, batch in results for layer, points in batch
            }
            self.state.coincident_points = tuple(
                by_layer[layer] for layer in range(self.state.geometry.layer_count)
            )
            self.state.csl_updating = False
            self.state.angle_update_active = False
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

    def _parallel_failure(self, error: Exception) -> None:
        local_failure = self.compute.parallel_stage == "local"
        self._cancel_parallel_work()
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
        self.state.coincident_points = same_layer_coincidence_sites(
            self.state.grains[0], self.state.grains[1], tolerance
        )
        self.state.csl_updating = False
        if self.local_active:
            self._start_local_matching()

    def _refresh_view_buffer(self) -> None:
        self._start_parallel_regeneration(
            compute_coincidences=not self.state.angle_update_active
        )
        if self.state.angle_update_active:
            self.coincidence_timer.start()

    def _on_view_range_changed(self, *_args) -> None:
        self._sync_view_scale_control()
        self.plot._update_marker_sizes()
        self.plot._draw_boundary()
        self.plot._draw_vector()
        self.plot._draw_manual_cell()
        self._refresh_visible_counts()
        if (
            not self.plot._buffer_contains_view()
            and self.compute.parallel_stage != "grains"
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

    def _update_visible_points(self, *_args) -> None:
        if len(self.state.grains) != 2:
            return
        states = self._region_states()
        boundary_ready = len(self.state.selected_points) == 2
        self.state.visible_atom_masks = []
        for grain_index, grain in enumerate(self.state.grains):
            if boundary_ready:
                mask = selected_region_mask(
                    grain.positions,
                    self.state.selected_points[0],
                    self.state.selected_points[1],
                    states[2 * grain_index],
                    states[2 * grain_index + 1],
                )
            else:
                mask = np.ones(len(grain.positions), dtype=bool)
            mask &= np.isin(
                grain.layers, tuple(self.state.visible_grain_layers[grain_index])
            )
            self.state.visible_atom_masks.append(mask)
            for layer, item in enumerate(self.plot.grain_layer_items[grain_index]):
                layer_mask = grain.layers == layer
                self.plot._set_scatter(item, grain.positions[mask & layer_mask])

        for layer, (points, item) in enumerate(
            zip(self.state.coincident_points, self.plot.coincidence_items, strict=True)
        ):
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
            if layer not in self.state.visible_layers:
                visible_points = np.empty((0, 2))
            self.plot._set_scatter(item, visible_points)
        self._refresh_visible_counts()
        self._queue_manual_count()

    def _refresh_visible_counts(self) -> None:
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
        preset = matching_csl_preset(
            self.state.angle_deg, axis=self.state.geometry.axis
        )
        suffix = f" · {preset.name}" if preset is not None else ""
        if self.state.manual_strain_fit is not None:
            fit = self.state.manual_strain_fit
            suffix = f"<br/>SELECTED-CELL BULK STRAIN · {100*self.state.near_cell.max_strain:.6f}% · ΔR₁/ΔR₂ = {fit.rotations_deg[0]:+.4f}°/{fit.rotations_deg[1]:+.4f}°"
        elif self.state.near_cell is not None:
            suffix = (
                f" · STRAINED near-CSL · {100*self.state.near_cell.max_strain:.3f}%"
            )
        elif self.local_active:
            suffix += f" · LOCAL near-CSL · d ≤ {self.controls.local_distance_spin.value():.4f} a₀ · unstrained"
        if self.state.display_rotation_deg:
            suffix += f" · display rotation {self.state.display_rotation_deg:.1f}°"
        angle_label = "reference θ" if self.state.manual_strain_fit is not None else "θ"
        self.plot._set_title(
            f"<span style='font-size:15px;color:#17212b'>"
            f"{self.state.geometry.lattice} ⟨{self.state.geometry.axis}⟩ tilt dichromatic pattern · {angle_label} = {self.state.angle_deg:.2f}°"
            f"{suffix}</span>"
        )
        prefix = "Reference" if self.state.manual_strain_fit is not None else "Exact"
        exact = f"{prefix} θ = {self.state.angle_deg:.8f}°"
        if preset is not None:
            tail = ", ".join(
                str(int(preset.n * index)) for index in self.state.geometry.axis_indices
            )
            exact += f"  ·  quaternion ({preset.m}, {tail})"
        self.controls.angle_exact_label.setText(exact)

    def _queue_angle_update(self, angle_deg: float) -> None:
        preset = matching_csl_preset(float(angle_deg), axis=self.state.geometry.axis)
        self.state.pending_angle = (
            preset.angle_deg if preset is not None else float(angle_deg)
        )
        self._sync_angle_controls(self.state.pending_angle)
        if abs(self.state.pending_angle - self.state.angle_deg) > 1.0e-12:
            if self.state.near_enabled:
                if self.compute.near_search is not None:
                    self.compute.near_search.cancel()
                self.near_debounce_timer.stop()
                self._clear_near_cell()
            self._cancel_parallel_work()
            if not self.state.angle_update_active:
                self._clear_lattice_selections()
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
        self._regenerate_buffer(compute_coincidences=False)

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
        self.controls.angle_slider.setValue(round(angle * 100.0))
        self.controls.angle_spin.setValue(angle)
        preset = matching_csl_preset(angle, axis=self.state.geometry.axis)
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
        self.state.interaction_mode = mode
        gb_blocker = QtCore.QSignalBlocker(self.controls.pick_gb_button)
        vector_blocker = QtCore.QSignalBlocker(self.controls.vector_button)
        self.controls.pick_gb_button.setChecked(mode == "boundary")
        self.controls.vector_button.setChecked(mode == "vector")
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

    def _clear_lattice_selections(self) -> None:
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
            lines.append(
                f"G{grain+1} current (polar): {format_direction_components(vector.current)}"
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
        axial_half_indices = (
            second.half_indices
            - first.half_indices
            + self.state.axial_repeat * self.state.geometry.axial_repeat_half_indices
        )
        dz = 0.5 * axial_half_indices @ self.state.geometry.frame[:, 2]
        # Both grains share the tilt axis, even after their in-plane F and
        # translations. Only z may be obtained by subtracting cross-grain
        # reference indices; x/y must come from their actual atom positions.
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
                    self.state.geometry.lattice,
                    self.state.geometry.axis,
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
            else:
                message = "Ready · choose an interaction tool"
        visible_status_layers = sorted(self.state.visible_layers)
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
            self.plot.save(Path(filename))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.manual_count_timer.stop()
        self.near_debounce_timer.stop()
        self.near_poll_timer.stop()
        self.parallel_poll_timer.stop()
        self.compute.close()
        super().closeEvent(event)
