"""Shared layer-marker rendering for the plot and appearance controls."""

from functools import lru_cache

from ._qt import QtCore, QtGui, pg
from . import GRAIN_1_COLOR, GRAIN_1_EDGE, LAYER_SYMBOLS
from ..crystal import MAX_LAYERS
from pyqtgraph.graphicsItems.ScatterPlotItem import drawSymbol


@lru_cache(maxsize=len(LAYER_SYMBOLS) + MAX_LAYERS)
def marker_symbol(key):
    """Resolve a stored marker key without changing PyQtGraph's symbol registry.

    Numbered circles provide distinct layer markers beyond the twelve original
    shapes. Cache paths so scatter-item symbol atlases can reuse them. This
    function is called on the GUI thread, with a QApplication already running.
    """
    if key in LAYER_SYMBOLS:
        return key
    if not isinstance(key, str) or not key.startswith("number:"):
        raise ValueError(f"Unknown layer marker: {key!r}")
    number = key.removeprefix("number:")
    if not number.isdecimal() or not 1 <= int(number) <= MAX_LAYERS or str(int(number)) != number:
        raise ValueError(f"Unknown layer marker: {key!r}")
    path = QtGui.QPainterPath()
    path.addEllipse(QtCore.QRectF(-0.5, -0.5, 1.0, 1.0))
    font = QtGui.QFont()
    font.setPixelSize(100)
    font.setBold(True)
    text = QtGui.QPainterPath()
    text.addText(QtCore.QPointF(), font, number)
    bounds = text.boundingRect()
    scale = min(0.72 / bounds.width(), 0.52 / bounds.height())
    transform = QtGui.QTransform()
    transform.scale(scale, scale)
    transform.translate(-bounds.center().x(), -bounds.center().y())
    path.addPath(transform.map(text))
    path.setFillRule(QtCore.Qt.FillRule.OddEvenFill)
    return path


def grain_edge_color(color, grain):
    """Keep the original G1 outline and darken custom filled-grain colors."""
    chosen = QtGui.QColor(color)
    if grain == 1:
        return chosen.name()
    if chosen.name() == GRAIN_1_COLOR:
        return GRAIN_1_EDGE
    return chosen.darker(135).name()


def marker_icon(symbol, color, filled=True, size=18):
    """Render the same shape and fill convention used by scatter plot items."""
    symbol = marker_symbol(symbol) if isinstance(symbol, str) else symbol
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.translate(size / 2, size / 2)
    brush_color = QtGui.QColor(color)
    brush_color.setAlpha(185 if filled else 0)
    drawSymbol(
        painter, symbol, max(1, size - 4),
        pg.mkPen(grain_edge_color(color, 0 if filled else 1), width=0.8 if filled else 1.5),
        pg.mkBrush(brush_color),
    )
    painter.end()
    return QtGui.QIcon(pixmap)
