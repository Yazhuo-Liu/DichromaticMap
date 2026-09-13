"""Start fresh processes so prior test imports cannot choose the Qt binding."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.gui


@pytest.fixture
def startup_environment():
    for dependency in ("PySide6", "pyqtgraph"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"{dependency} is required for GUI startup tests")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), environment.get("PYTHONPATH", ""))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYQTGRAPH_QT_LIB", None)
    return environment


def run_fresh(arguments, environment, directory):
    result = subprocess.run(
        [sys.executable, "-B", *arguments], cwd=directory, env=environment,
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


@pytest.mark.parametrize("requested_binding", [None, "PyQt5", "PyQt6"])
def test_command_line_export_uses_pyside6(startup_environment, tmp_path, requested_binding):
    # A conflicting environment variable must not select a different widget
    # family, even when the requested binding is installed alongside PySide6.
    if requested_binding is not None:
        startup_environment["PYQTGRAPH_QT_LIB"] = requested_binding
    image = tmp_path / "startup.png"
    result = run_fresh(
        ["-m", "dichromatic_map", "--workers", "4", "--save", str(image)],
        startup_environment, tmp_path,
    )
    assert "Saved dichromatic pattern" in result.stdout
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image.stat().st_size > 1000


@pytest.mark.parametrize("first_module", ["controls", "plot", "window"])
def test_gui_import_order_keeps_widgets_compatible(startup_environment, tmp_path, first_module):
    startup_environment["PYQTGRAPH_QT_LIB"] = "PyQt6"
    run_fresh(
        ["-c", """
import importlib
import sys
importlib.import_module('dichromatic_map.ui.' + sys.argv[1])
from dichromatic_map.ui.controls import create_application
from PySide6 import QtWidgets
import pyqtgraph as pg
assert pg.Qt.QT_LIB == 'PySide6'
app = create_application()
window = QtWidgets.QMainWindow()
plot = pg.PlotWidget()
assert isinstance(plot, QtWidgets.QWidget)
window.setCentralWidget(plot)
assert window.centralWidget() is plot
window.close()
app.processEvents()
""", first_module], startup_environment, tmp_path,
    )


def test_preloaded_incompatible_pyqtgraph_reports_how_to_restart(startup_environment, tmp_path):
    if importlib.util.find_spec("PyQt6") is None:
        pytest.skip("Requires a competing Qt binding; installed in Windows CI")
    startup_environment["PYQTGRAPH_QT_LIB"] = "PyQt6"
    run_fresh(
        ["-c", """
import sys
import pyqtgraph as pg
assert pg.Qt.QT_LIB == 'PyQt6'
try:
    from dichromatic_map.ui.controls import create_application
except RuntimeError as error:
    message = str(error)
    assert 'already using PyQt6' in message
    assert 'Restart the Python process' in message
    assert 'python -m dichromatic_map' in message
else:
    raise AssertionError('An already loaded incompatible binding must be rejected')
assert 'PySide6' not in sys.modules
"""], startup_environment, tmp_path,
    )
