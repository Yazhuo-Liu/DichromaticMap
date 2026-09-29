"""Candidate preview for symmetry and parallelogram cell completion."""

import numpy as np

from ._qt import QtCore, QtWidgets, pg
from ..crystal import rotation_matrix_2d


class CellCompletionDialog(QtWidgets.QDialog):
    def __init__(self, candidates, picked_count, colors, display_rotation, parent=None):
        super().__init__(parent)
        self.candidates = candidates
        self.picked_count = picked_count
        self.colors = colors
        self.rotation = rotation_matrix_2d(display_rotation)
        self.setWindowTitle("Complete common cell")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(720, 740)
        layout = QtWidgets.QVBoxLayout(self)
        help_text = QtWidgets.QLabel(
            "Preview original atom vertices: '(auto)' marks automatically completed vertices. "
            "Use candidate completes the selection. Exact CSL cells need no strain; "
            "near cells can be aligned separately afterward."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.candidate_combo = QtWidgets.QComboBox()
        self.candidate_combo.setMinimumContentsLength(35)
        self.candidate_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        for index, candidate in enumerate(candidates):
            result = (f"strain {100 * candidate.fit.cell.max_strain:.4f}%"
                      if candidate.fit is not None else "fit unavailable")
            self.candidate_combo.addItem(
                f"{index + 1}: {candidate.atoms[0]}/{candidate.atoms[1]} atoms/layer · {result}"
            )
        layout.addWidget(self.candidate_combo)
        self.preview = pg.PlotWidget(background="w")
        self.preview.setMinimumHeight(260)
        self.preview.setAspectLocked(True)
        self.preview.setLabel("bottom", "Display x / a₀")
        self.preview.setLabel("left", "Display y / a₀")
        layout.addWidget(self.preview, 1)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(160)
        layout.addWidget(self.details)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.use_button = buttons.addButton("Use candidate", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.candidate_combo.currentIndexChanged.connect(self._show_candidate)
        self._show_candidate(0)

    @property
    def selected_candidate(self):
        return self.candidates[self.candidate_combo.currentIndex()]

    def _show_candidate(self, index):
        candidate = self.candidates[index]
        self.preview.clear()
        for grain, polygon in enumerate(candidate.vertices):
            points = polygon @ self.rotation.T
            outline = np.vstack((points, points[0]))
            style = QtCore.Qt.PenStyle.SolidLine if grain == 0 else QtCore.Qt.PenStyle.DashLine
            pen = pg.mkPen(self.colors[grain], width=2, style=style)
            self.preview.plot(outline[:, 0], outline[:, 1], pen=pen)
            self.preview.plot(
                points[:, 0], points[:, 1], pen=None, symbol="o",
                symbolSize=10, symbolPen=pg.mkPen(self.colors[grain]),
                symbolBrush=pg.mkBrush(self.colors[grain] if grain == 0 else "w"),
            )
        for number, midpoint in enumerate(candidate.vertices.mean(axis=0) @ self.rotation.T, start=1):
            suffix = " (auto)" if number > self.picked_count else ""
            label = pg.TextItem(f"C{number}{suffix}", color="#334155", anchor=(0, 1))
            label.setPos(*midpoint)
            self.preview.addItem(label)
        self.preview.autoRange(padding=0.12)
        lines = [
            candidate.description + " (analysis frame)",
            f"Atoms per selected layer G1/G2: {candidate.atoms[0]} / {candidate.atoms[1]}",
            f"Original areas G1/G2: {candidate.areas[0]:.6f} / {candidate.areas[1]:.6f} a₀²",
            "Original pair distances C1–C4 / a₀: " + np.array2string(
                np.linalg.norm(candidate.vertices[0] - candidate.vertices[1], axis=1), precision=6,
            ),
        ]
        if candidate.fit is not None:
            fit = candidate.fit
            lines += [
                f"Max |principal strain|: {100 * fit.cell.max_strain:.6f}%",
                f"Polar rotation G1/G2: {fit.rotations_deg[0]:+.6f}° / {fit.rotations_deg[1]:+.6f}°",
                f"Four-pair alignment residual: {fit.residual:.2e} a₀",
                "Within the selected strain and rotation limits. Primitive cell not assumed.",
            ]
        else:
            lines += [candidate.error, "Close this preview to adjust the limits or pick another edge."]
        self.details.setPlainText("\n".join(lines))
        self.use_button.setEnabled(candidate.fit is not None)
