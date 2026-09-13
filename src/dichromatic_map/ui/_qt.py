"""Use one Qt binding for the viewer and all PyQtGraph widgets."""

import os

# The viewer uses PySide6 throughout. PyQtGraph's automatic selection can
# otherwise prefer PyQt6 in environments containing multiple Qt bindings.
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

import pyqtgraph as pg

if pg.Qt.QT_LIB != "PySide6":
    raise RuntimeError(
        f"DichromaticMap requires PySide6, but PyQtGraph is already using {pg.Qt.QT_LIB}. "
        "Restart the Python process and launch the viewer with "
        "python -m dichromatic_map before importing PyQtGraph."
    )

from PySide6 import QtCore, QtGui, QtWidgets
