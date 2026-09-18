"""Session file dialogs and restoration of the viewer's physical state."""

from contextlib import ExitStack
from pathlib import Path
import numpy as np

from ._qt import QtCore, QtWidgets
from .. import session as session_files
from ..crystal import projected_columns, rotation_matrix_2d
from ..state import BUFFER_FACTOR, VIEW_SCALE_MIN, VIEW_SCALE_MAX
from ..strain import selected_cell_strain_readout, tensor_readout


class SessionController:
    def __init__(self, owner):
        self.owner = owner

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
        # Commit a slider preview before serializing; cached counts and worker
        # results are not the source of the exported numerical tables.
        if owner.state.angle_update_active or owner.angle_preview_timer.isActive():
            owner._finish_angle_update()
        session_files.save_session(
            path, owner.state, self.settings(), owner.plot._view_range(),
        )

    def load(self, path):
        snapshot = session_files.load_session(path)
        self._check_view(snapshot)
        owner = self.owner
        previous = session_files.SessionSnapshot(
            owner.state, self.settings(), owner.plot._view_range(),
        )
        try:
            self._apply(snapshot)
        except Exception:
            # Validation normally rejects bad files before this point. Keep
            # the current session recoverable if applying a valid file fails.
            self._apply(previous)
            raise

    @staticmethod
    def _check_view(snapshot):
        state = snapshot.state
        x0, x1, y0, y1 = snapshot.view_range
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        corners = corners @ rotation_matrix_2d(state.display_rotation_deg)
        low, high = corners.min(axis=0), corners.max(axis=0)
        width, height = BUFFER_FACTOR * (high - low)
        if state.near_enabled and state.near_method == "local" and state.manual_strain_fit is None:
            width = max(width, 4.4 * snapshot.settings["local_distance"])
            height = max(height, 4.4 * snapshot.settings["local_distance"])
        # Exercise the same allocation limits before replacing the current
        # state. No workers, widgets or selections are changed during this check.
        for grain, sign in enumerate((1, -1)):
            projected_columns(
                width, height, sign * state.angle_deg / 2,
                center=(low + high) / 2, deformation=state.deformations[grain],
                lattice=state.geometry.lattice, axis=state.geometry.axis,
                translation=state.translations[grain],
            )

    def _apply(self, snapshot):
        owner = self.owner
        controls, plot, compute = owner.controls, owner.plot, owner.compute
        for timer in (owner.angle_preview_timer, owner.coincidence_timer,
                      owner.view_refresh_timer, owner.near_debounce_timer,
                      owner.near_poll_timer, owner.manual_count_timer):
            timer.stop()
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
        owner._start_parallel_regeneration(True)
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
        except (OSError, ValueError) as error:
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
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self.owner, "Cannot import session", str(error))
