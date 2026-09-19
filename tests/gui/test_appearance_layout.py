"""Appearance controls fit when Qt uses wider fallback font metrics."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.gui
ROOT = Path(__file__).resolve().parents[2]


def test_appearance_layers_fit_without_horizontal_clipping_on_minimal_qt(tmp_path):
    # The minimal platform's fallback font made a label and its combo wider
    # than the available row. Hiding the horizontal scrollbar still clipped
    # the controls. A fresh process is required to select this Qt platform.
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "minimal"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", """
import runpy
import sys
from dichromatic_map.ui.controls import create_application
from PySide6 import QtCore

# conftest selects offscreen for ordinary tests; load its harness only after
# QApplication has fixed this process's platform to minimal.
app = create_application()
app.setQuitOnLastWindowClosed(False)
assert app.platformName() == 'minimal'
GuiHarness = runpy.run_path(sys.argv[1])['GuiHarness']
gui = GuiHarness(app, [])
try:
    window = gui.window(lattice='SC', axis='2 3 5', angle_deg=15, width=6, height=5)
    controls = window.controls
    controls.view_performance_section.toggle.setChecked(True)
    tabs = controls.view_performance_tabs
    tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == 'APPEARANCE'))
    window.resize(1040, 680)
    gui.settle(window)
    assert window.width() <= 1040
    assert len(controls.layer_symbol_combos) == 38
    scroll = controls.appearance_symbols_scroll
    rows = scroll.widget()

    def diagnostics():
        return (
            f'viewport={scroll.viewport().size()}, rows={rows.size()}, '
            f'minimumSizeHint={rows.minimumSizeHint()}, '
            f'horizontalMaximum={scroll.horizontalScrollBar().maximum()}'
        )

    gui.wait_for(
        window,
        lambda: scroll.horizontalScrollBar().maximum() == 0
            and rows.width() <= scroll.viewport().width(),
        timeout=5, diagnostics=diagnostics,
    )
    assert scroll.verticalScrollBar().maximum() > 0, diagnostics()
    for combo in controls.layer_symbol_combos:
        left = combo.mapTo(rows, QtCore.QPoint()).x()
        assert 0 <= left and left + combo.width() <= scroll.viewport().width(), diagnostics()
        assert combo.width() >= combo.minimumSizeHint().width(), diagnostics()

    last = controls.layer_symbol_combos[-1]
    scroll.ensureWidgetVisible(last, 0, 0)

    def last_layer_is_visible():
        top_left = last.mapTo(scroll.viewport(), QtCore.QPoint())
        return scroll.viewport().rect().contains(QtCore.QRect(top_left, last.size()))

    gui.wait_for(window, last_layer_is_visible, timeout=5, diagnostics=diagnostics)
    assert scroll.verticalScrollBar().value() > 0, diagnostics()
finally:
    gui.close()
""", str(Path(__file__).with_name("conftest.py"))],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
