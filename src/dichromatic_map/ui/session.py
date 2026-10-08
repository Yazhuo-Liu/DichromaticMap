"""Session file dialogs and restoration of the viewer's physical state."""

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from ._qt import QtCore, QtWidgets
from .. import session as session_files
from ..compute import run_numerical_task
from ..crystal import projected_columns, rotation_matrix_2d
from ..state import BUFFER_FACTOR, VIEW_SCALE_MIN, VIEW_SCALE_MAX
from ..strain import selected_cell_strain_readout, tensor_readout


class SessionController:
    def __init__(self, owner):
        self.owner = owner
        self.pending_future = None
        self._executor = None
        self._closed = False

    def _background(self, function, *args):
        """Keep painting/completion active while preserving synchronous API calls.

        A separate, single session worker lets import replace an analysis whose
        numerical worker is still busy. Workers never access widgets or the
        live mutable state. Only this main-thread continuation commits a load.
        """
        if self._closed:
            raise RuntimeError("The viewer is closed")
        if self.pending_future is not None:
            raise RuntimeError("A session operation is already running")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        future = self._executor.submit(run_numerical_task, function, *args)
        self.pending_future = future
        loop = QtCore.QEventLoop()
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: loop.quit() if self._closed or future.done() else None)
        try:
            if not future.done():
                timer.start(10)
                loop.exec(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            if self._closed:
                future.cancel()
                raise RuntimeError("The viewer closed during the session operation")
            return future.result()
        finally:
            timer.stop()
            self.pending_future = None

    def close(self):
        self._closed = True
        if self.pending_future is not None:
            self.pending_future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def settings(self):
        controls = self.owner.controls
        return {
            "region_states": list(self.owner._region_states()),
            "manual_visible": controls.manual_visible_check.isChecked(),
            "show_common_cell": controls.cell_check.isChecked(),
            "local_distance": controls.local_distance_spin.value(),
            "strain_percent": controls.strain_spin.value(),
            "search_index": controls.search_index_spin.value(),
            "manual_strain_percent": controls.manual_strain_limit.value(),
            "manual_rotation_deg": controls.manual_rotation_limit.value(),
        }

    def save(self, path):
        owner = self.owner
        owner._flush_display_rotation()
        # Commit a slider preview before serializing; cached counts and worker
        # results are not the source of the exported numerical tables.
        if owner.state.angle_update_active or owner.angle_preview_timer.isActive():
            owner._finish_angle_update()
        # The small physical payload excludes large display buffers and copies
        # every saved array before Qt completion events can change live state.
        payload = dict(
            format=session_files.FORMAT, schema_version=session_files.SCHEMA_VERSION,
            state=session_files._state_payload(owner.state),
            settings=self.settings(), view_range=list(owner.plot._view_range()),
        )
        self._background(self._save_payload, path, payload)

    @staticmethod
    def _save_payload(path, payload):
        snapshot = session_files._snapshot_from_payload(payload)
        session_files.save_session(path, snapshot.state, snapshot.settings, snapshot.view_range)

    @classmethod
    def _prepare_load(cls, path):
        snapshot = session_files.load_session(path)
        return snapshot, cls._check_view(snapshot)

    def load(self, path):
        self.owner._flush_display_rotation()
        inputs = self._input_key()
        snapshot, prepared = self._background(self._prepare_load, path)
        if inputs != self._input_key():
            raise RuntimeError("Session import cancelled because the current controls changed")
        owner = self.owner
        previous = session_files.SessionSnapshot(
            owner.state, self.settings(), owner.plot._view_range(),
        )
        previous_prepared = (
            (owner.state.grains, owner.state.buffer_bounds)
            if len(owner.state.grains) == 2 and owner.state.buffer_bounds is not None
            and owner.state.grain_signature == owner._geometry_signature() else None
        )
        try:
            self._apply(snapshot, prepared)
        except Exception:
            # Validation normally rejects bad files before this point. Keep
            # the current session recoverable if applying a valid file fails.
            self._apply(previous, previous_prepared)
            raise

    def _input_key(self):
        """Detect programmatic/reentrant edits while ordinary input is paused."""
        state, controls = self.owner.state, self.owner.controls
        return (
            id(state), state.parameters, state.geometry.lattice, state.geometry.axis,
            state.pending_angle, controls.angle_spin.value(), controls.rotation_spin.value(),
            tuple(frozenset(layers) for layers in state.visible_grain_layers),
            tuple(state.grain_colors), tuple(state.layer_symbols), tuple(state.layer_size_scales),
            state.axial_repeat, state.show_reference_axes, state.near_enabled, state.near_method,
            self.owner._region_states(),
            tuple(sorted(self.settings().items())),
        )

    @staticmethod
    def _check_view(snapshot):
        state = snapshot.state
        x0, x1, y0, y1 = snapshot.view_range
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        corners = corners @ rotation_matrix_2d(state.display_rotation_deg)
        low, high = corners.min(axis=0), corners.max(axis=0)
        width, height = BUFFER_FACTOR * (high - low)
        if state.near_enabled and state.near_method == "local" and state.manual_strain_fit is None:
            width = max(width, high[0] - low[0] + 4.4 * snapshot.settings["local_distance"])
            height = max(height, high[1] - low[1] + 4.4 * snapshot.settings["local_distance"])
        # Exercise the same allocation limits before replacing the current
        # state. No workers, widgets or selections are changed during this check.
        grains = [
            projected_columns(
                width, height, sign * state.angle_deg / 2,
                center=(low + high) / 2, deformation=state.deformations[grain],
                lattice=state.geometry.lattice, axis=state.geometry.axis,
                translation=state.translations[grain],
            )
            for grain, sign in enumerate((1, -1))
        ]
        center = (low + high) / 2
        bounds = (center[0] - width / 2, center[0] + width / 2,
                  center[1] - height / 2, center[1] + height / 2)
        return grains, bounds

    def _apply(self, snapshot, prepared=None):
        if prepared is None:
            prepared = self._background(self._check_view, snapshot)
        owner = self.owner
        controls, plot, compute = owner.controls, owner.plot, owner.compute
        for timer in (owner.angle_preview_timer, owner.coincidence_timer,
                      owner.view_refresh_timer, owner.near_debounce_timer,
                      owner.near_poll_timer, owner.manual_count_timer):
            timer.stop()
        owner.display_rotation_timer.stop()
        owner._pending_display_rotation = None
        owner._cancel_parallel_work()
        if compute.near_search is not None:
            compute.near_search.cancel()
        if compute.manual_count_future is not None:
            compute.manual_count_future.cancel()
        # A running count may not be cancellable. Drop its publication handle,
        # so a result from the previous session cannot replace imported counts.
        compute.manual_count_future = None
        compute.manual_count_pending = None
        compute.manual_count_running_key = None
        state, settings = snapshot.state, snapshot.settings
        state.manual_count_key = None
        state.manual_counts = None
        owner.state = state
        widgets = [
            controls.structure_combo, controls.axis_combo, controls.layer_combo,
            controls.rotation_spin, controls.rotation_slider, controls.reference_axes_check,
            controls.axial_spin, controls.cell_check, controls.manual_visible_check,
            controls.local_distance_spin, controls.strain_spin, controls.search_index_spin,
            controls.manual_strain_limit, controls.manual_rotation_limit,
            controls.near_button, controls.near_method_combo, controls.near_combo,
            *controls.region_checks, plot.view_box,
        ]
        with ExitStack() as blockers:
            for widget in widgets:
                blockers.enter_context(QtCore.QSignalBlocker(widget))
            controls.structure_combo.setCurrentIndex(controls.structure_combo.findData(state.geometry.lattice))
            index = controls.axis_combo.findData(state.geometry.axis)
            controls.axis_combo.setCurrentIndex(index if index >= 0 else controls.axis_combo.findData(None))
            controls.custom_axis_edit.setText(" ".join(map(str, state.geometry.axis_indices)))
            controls.custom_axis_row.setVisible(index < 0)
            controls.axis_error.clear()
            controls.axis_error.hide()
            with QtCore.QSignalBlocker(controls.preset_combo):
                controls.preset_combo.clear()
                controls.preset_combo.addItem("Custom angle", None)
                for index, preset in enumerate(state.presets):
                    controls.preset_combo.addItem(preset.label, index)
            controls._populate_layer_combo()
            controls._rebuild_layer_checks()
            controls.rotation_spin.setValue(state.display_rotation_deg)
            controls.rotation_slider.setValue(round(state.display_rotation_deg * 10))
            controls.reference_axes_check.setChecked(state.show_reference_axes)
            controls.axial_spin.setValue(state.axial_repeat)
            controls.axial_spin.setPrefix("+" if state.axial_repeat >= 0 else "")
            controls.cell_check.setChecked(settings["show_common_cell"])
            controls.manual_visible_check.setChecked(settings["manual_visible"])
            for check, checked in zip(controls.region_checks, settings["region_states"]):
                check.setChecked(checked)
            for widget, key in (
                (controls.local_distance_spin, "local_distance"),
                (controls.strain_spin, "strain_percent"),
                (controls.search_index_spin, "search_index"),
                (controls.manual_strain_limit, "manual_strain_percent"),
                (controls.manual_rotation_limit, "manual_rotation_deg"),
            ):
                widget.setValue(settings[key])
            controls.near_button.setChecked(state.near_enabled)
            controls.near_button.setText("Disable Near-CSL" if state.near_enabled else "Enable Near-CSL")
            controls.near_method_combo.setCurrentIndex(controls.near_method_combo.findData(state.near_method))
            controls.near_combo.clear()
            controls.near_combo.addItems([cell.label() for cell in state.near_solutions])
            if state.near_solutions:
                controls.near_combo.setCurrentIndex(0)
            plot._create_layer_items()
            owner._sync_angle_controls()
            owner._update_geometry_labels()
            owner._sync_near_controls()
            plot.refresh_appearance()
            self._restore_result_text()
            owner._update_title()
            plot.view_box.setLimits(
                minXRange=state.parameters.width * VIEW_SCALE_MIN,
                maxXRange=state.parameters.width * VIEW_SCALE_MAX,
            )
            x0, x1, y0, y1 = snapshot.view_range
            plot.view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        owner._sync_view_scale_control()
        plot._update_marker_sizes()
        plot._update_common_cell()
        plot._draw_boundary()
        plot._draw_vector()
        plot._draw_manual_cell()
        owner._set_mode(state.interaction_mode)
        state.grains, state.buffer_bounds = prepared
        state.grain_signature = owner._geometry_signature()
        state.render_error = None
        owner._update_visible_points()
        owner._start_parallel_coincidences()
        owner._queue_manual_count()
        owner._update_status("Session restored.")

    def _restore_result_text(self):
        owner = self.owner
        state, controls = owner.state, owner.controls
        if state.manual_strain_fit is not None:
            details = selected_cell_strain_readout(
                state.manual_strain_fit, state.angle_deg,
                strain_limit_percent=controls.manual_strain_limit.value(),
                rotation_limit_deg=controls.manual_rotation_limit.value(),
            )
            controls.manual_strain_details.setPlainText(details)
            controls.manual_strain_note.setText("Saved bulk strain restored. Use Restore original local structure to undo it.")
            controls.near_info.setPlainText(details)
        elif state.near_cell is not None:
            controls.near_info.setPlainText(tensor_readout(state.near_cell, state.angle_deg))
        elif owner.local_active:
            controls.near_info.setPlainText("Local matching enabled; rebuilding near pairs…")
        else:
            controls.near_info.setPlainText(
                "No strain cell applied. Change search settings to run a new search."
                if state.near_enabled else "Off. Original unstrained lattices displayed."
            )
        count = len(state.manual_vertices)
        controls.manual_info.setPlainText(
            "No manual cell." if not count else
            f"Restored {count}/4 vertices; " + ("recomputing counts…" if count == 4 else "continue picking around the perimeter.")
        )

    def choose_save_path(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.owner, "Save session and numerical tables", "dichromatic_session.dmap",
            "DichromaticMap session (*.dmap)",
        )
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".dmap")
        try:
            self.save(path)
        except (OSError, ValueError, RuntimeError) as error:
            if self._closed:
                return
            QtWidgets.QMessageBox.warning(self.owner, "Cannot save session", str(error))
            return
        self.owner._update_status(f"Saved session and numerical tables: {path.name}")

    def choose_import_path(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.owner, "Import DichromaticMap session", "", "DichromaticMap session (*.dmap)",
        )
        if not filename:
            return
        try:
            self.load(Path(filename))
        except (OSError, ValueError, RuntimeError) as error:
            if self._closed:
                return
            QtWidgets.QMessageBox.warning(self.owner, "Cannot import session", str(error))
