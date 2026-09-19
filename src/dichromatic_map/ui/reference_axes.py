"""Screen-space crystallographic orientation reference for the two grains."""

import numpy as np
from ._qt import QtCore, QtGui, pg
from . import GRAIN_1_COLOR, GRAIN_2_COLOR


class GrainReferenceAxes(pg.GraphicsObject):
    """Four labeled arrows sharing a fixed origin, independent of data bounds."""

    def __init__(self, parent):
        super().__init__()
        self.setParentItem(parent)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.setZValue(20)
        self.directions = np.tile(np.eye(2), (2, 1, 1))
        self.labels = ("", "")
        self.colors = (GRAIN_1_COLOR, GRAIN_2_COLOR)
        self.origin = QtCore.QPointF()
        self._rect = QtCore.QRectF()
        self._groups = []
        self._key = None
        self._font = QtGui.QFont()
        self._font.setPointSize(9)
        self.setToolTip(
            "Orthogonal in-plane crystal reference directions in each grain's color. "
            "Both grains share a fixed origin. Arrows follow the polar rigid "
            "rotation under strain, not shear or stretch."
        )

    def set_colors(self, colors):
        colors = tuple(colors)
        if colors != self.colors:
            self.colors = colors
            self.update()

    def set_reference(self, directions, labels):
        directions = np.asarray(directions, dtype=float)
        labels = tuple(labels)
        key = (labels, tuple(directions.ravel()))
        if key == self._key:
            return
        self._key = key
        self.directions = directions.copy()
        self.labels = labels
        self._layout()
        self.update()

    def _layout(self):
        metrics = QtGui.QFontMetricsF(self._font)
        sizes = [(metrics.horizontalAdvance(label) + 4, metrics.height() + 2)
                 for label in self.labels]
        length, gap, padding = 36.0, 10.0, 10.0
        # Reserve the full rotation envelope. Re-centering on the current
        # arrow/text bounds makes the origin jump as the orientation changes.
        half_width = length + gap + max(width for width, _ in sizes) + padding
        half_height = length + gap + max(height for _, height in sizes) + padding
        rect = QtCore.QRectF(0, 0, 2 * half_width, 2 * half_height)
        if rect != self._rect:
            self.prepareGeometryChange()
            self._rect = rect
        self.origin = self._rect.center()
        self._groups = []
        for directions in self.directions:
            arrows = []
            for direction, label, (width, height) in zip(directions, self.labels, sizes):
                # The analysis frame is y-up; graphics-widget coordinates are y-down.
                unit = np.array([direction[0], -direction[1]])
                tip = length * unit
                normal = np.array([-unit[1], unit[0]])
                head = np.array([tip, tip - 8 * unit + 3.5 * normal,
                                 tip - 8 * unit - 3.5 * normal])
                polygon = QtGui.QPolygonF([QtCore.QPointF(*point) for point in head])
                # Continuous alignment avoids switching the label abruptly
                # from its left edge to its center or right edge near vertical.
                center = (length + gap + np.array([width, height]) / 2) * unit
                text_rect = QtCore.QRectF(
                    center[0] - width / 2, center[1] - height / 2, width, height,
                )
                arrows.append((QtCore.QPointF(*tip), polygon, text_rect, label))
            self._groups.append(arrows)

    def boundingRect(self):  # noqa: N802 - Qt virtual
        return QtCore.QRectF(self._rect)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(pg.mkPen("#cbd5e1", width=1))
        painter.setBrush(pg.mkBrush(255, 255, 255, 235))
        painter.drawRoundedRect(self._rect.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        painter.save()
        painter.translate(self.origin)
        painter.setFont(self._font)
        for color, arrows in zip(self.colors, self._groups):
            painter.setBrush(pg.mkBrush(color))
            for tip, polygon, text_rect, label in arrows:
                painter.setPen(pg.mkPen(color, width=2))
                painter.drawLine(QtCore.QPointF(0, 0), tip)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawPolygon(polygon)
                painter.setPen(pg.mkPen(color))
                painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()
