"""Numbered layer markers stay distinguishable without system fonts."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from dichromatic_map.crystal import MAX_LAYERS
from dichromatic_map.ui._qt import QtGui
from dichromatic_map.ui.markers import marker_icon, marker_symbol

pytestmark = pytest.mark.gui
ROOT = Path(__file__).resolve().parents[2]


def path_fingerprint(number):
    path = marker_symbol(f"number:{number}")
    assert not path.isEmpty()
    return tuple(
        (element.x, element.y, element.type.value)
        for element in (path.elementAt(index) for index in range(path.elementCount()))
    )


def icon_fingerprint(number, filled, size=32):
    image = marker_icon(
        f"number:{number}", "#1677d2", filled=filled, size=size,
    ).pixmap(size, size).toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
    assert not image.isNull()
    pixels = bytes(image.constBits())
    assert any(pixels[3::4]), f"Marker {number} is fully transparent"
    return pixels


def test_all_numbered_markers_have_distinct_paths(qt_app):
    fingerprints = {path_fingerprint(number) for number in range(1, MAX_LAYERS + 1)}
    assert len(fingerprints) == MAX_LAYERS


@pytest.mark.parametrize("filled", [True, False], ids=["filled", "outline"])
@pytest.mark.parametrize("size", [18, 32], ids=["default-size", "large-size"])
def test_all_numbered_markers_render_distinct_icons(qt_app, filled, size):
    fingerprints = {
        icon_fingerprint(number, filled, size) for number in range(1, MAX_LAYERS + 1)
    }
    assert len(fingerprints) == MAX_LAYERS


def test_numbered_markers_remain_distinct_with_minimal_qt_platform(tmp_path):
    # QApplication fixes its platform at startup, so test the font-limited
    # minimal plugin in its own process. Previously addText rendered every
    # two-digit number as the same pair of missing-glyph rectangles here.
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
from PySide6 import QtWidgets

app = QtWidgets.QApplication([])
helpers = runpy.run_path(sys.argv[1])
numbers = range(13, 39)
paths = {helpers['path_fingerprint'](number) for number in numbers}
assert len(paths) == len(numbers), f'Only {len(paths)} distinct numbered paths'
for filled in (True, False):
    icons = {helpers['icon_fingerprint'](number, filled) for number in numbers}
    assert len(icons) == len(numbers), (
        f'Only {len(icons)} distinct numbered icons; filled={filled}'
    )
""", str(Path(__file__).resolve())],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
