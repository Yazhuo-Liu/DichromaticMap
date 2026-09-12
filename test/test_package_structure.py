"""Import boundaries, component ownership and packaged worker regressions."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import re
import io
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from dichromatic_map import cells, compute, crystal, matching
from dichromatic_map.state import PatternParameters, PatternState


def worker_imports():
    """Run inside a spawned process, not the test runner's Qt session."""
    return sorted(
        name
        for name in sys.modules
        if name.split(".")[0] in {"PySide6", "pyqtgraph", "matplotlib"}
    )


class PackageTests(unittest.TestCase):
    def test_numerical_modules_and_help_do_not_import_gui(self):
        # Fail on any attempted GUI import, even if the dependency happens to
        # be installed. Also reject reverse imports of the legacy adapters.
        program = """
import importlib.abc
import sys

class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in {"PySide6", "pyqtgraph", "matplotlib"} or top.startswith("tilt_gb_"):
            raise AssertionError("Unexpected dependency: " + fullname)

sys.meta_path.insert(0, NoGui())
from dichromatic_map import crystal, cells, matching, strain, compute, state
from dichromatic_map.__main__ import main
session = state.PatternState(state.PatternParameters())
assert session.geometry.layer_count == 2
assert matching.exact_csl_cell(session.angle_deg) is not None
sys.argv = ["dichromatic-map", "--help"]
try:
    main()
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("--help did not exit")
"""
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--workers", result.stdout)

    def test_public_exports_refer_to_domain_implementations(self):
        import dichromatic_map

        for module, names in (
            (crystal, ("get_geometry", "projected_columns")),
            (cells, ("count_cell_atoms",)),
            (
                matching,
                ("exact_csl_cell", "local_near_pairs", "same_layer_coincidence_sites"),
            ),
        ):
            for name in names:
                with self.subTest(name=name):
                    self.assertIs(getattr(dichromatic_map, name), getattr(module, name))

    def test_root_launcher_help_from_another_directory_without_gui(self):
        program = """
import importlib.abc
import runpy
import sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"PySide6", "pyqtgraph", "matplotlib"}:
            raise AssertionError("Unexpected GUI import: " + fullname)
sys.meta_path.insert(0, NoGui())
sys.argv = [sys.argv[1], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", program, str(ROOT / "main.py")],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--workers", result.stdout)

    def test_root_launcher_exports_without_project_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pattern.png"
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(ROOT / "main.py"),
                    "--lattice",
                    "BCC",
                    "--axis",
                    "100",
                    "--workers",
                    "2",
                    "--save",
                    str(output),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
                env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            with output.open("rb") as stream:
                self.assertEqual(stream.read(8), b"\x89PNG\r\n\x1a\n")

    def test_state_defaults_validation_and_session_isolation(self):
        first = PatternState(PatternParameters())
        second = PatternState(PatternParameters())
        self.assertEqual(first.parameters, PatternParameters())
        self.assertEqual(first.visible_grain_layers, [{0, 1}, {0, 1}])
        first.visible_grain_layers[0].remove(0)
        first.deformations[0][0, 0] = 1.1
        first.manual_vertices.append("sentinel")
        self.assertEqual(first.visible_grain_layers[1], {0, 1})
        self.assertEqual(second.visible_grain_layers, [{0, 1}, {0, 1}])
        np.testing.assert_array_equal(second.deformations[0], np.eye(2))
        self.assertFalse(second.manual_vertices)
        with self.assertRaises(ValueError):
            PatternState(PatternParameters(width=-1))

    def test_documentation_links_and_numerical_examples(self):
        paths = [ROOT / "README.md"]
        paths += [
            ROOT / "docs" / language / filename
            for language in ("en", "zh")
            for filename in ("README.md", "development.md")
        ]
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                content = path.read_text(encoding="utf-8")
                for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
                    if "://" in target or target.startswith("#"):
                        continue
                    self.assertTrue(
                        (path.parent / target.split("#", 1)[0]).is_file(),
                        f"Broken documentation link: {path}: {target}",
                    )
                for example in re.findall(r"```python\n(.*?)```", content, re.S):
                    with redirect_stdout(io.StringIO()):
                        exec(compile(example, str(path), "exec"), {})

    def test_spawn_workers_use_new_modules_without_gui(self):
        arguments = (4.0, 3.0, 17.0, (0.3, -0.2))
        vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        count_arguments = (vertices, 0.0, (np.eye(2), np.eye(2)))
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=compute.worker_initializer,
        ) as executor:
            pid, grain = executor.submit(
                compute.generate_grain_worker, *arguments
            ).result(30)
            self.assertNotEqual(pid, os.getpid())
            self.assertIs(type(grain), crystal.ProjectedGrain)
            expected = crystal.projected_columns(*arguments)
            np.testing.assert_array_equal(grain.half_indices, expected.half_indices)
            np.testing.assert_allclose(grain.positions, expected.positions)
            _, counts = executor.submit(
                compute.cell_count_worker, *count_arguments
            ).result(30)
            self.assertIs(type(counts), cells.CellAtomCounts)
            np.testing.assert_array_equal(
                counts.half_open, cells.count_cell_atoms(*count_arguments).half_open
            )
            self.assertEqual(executor.submit(worker_imports).result(30), [])

    def test_compute_session_owns_and_closes_its_thread_executor(self):
        session = compute.ComputeSession()
        executor = session.background_executor()
        try:
            self.assertIs(session.background_executor(), executor)
            self.assertEqual(executor.submit(abs, -2).result(10), 2)
            session.close()
            self.assertIsNone(session.local_thread_executor)
            self.assertIsNone(session.executor)
            with self.assertRaises(RuntimeError):
                executor.submit(abs, -2)
            session.close()  # Cleanup can safely be requested twice.
        finally:
            session.close()

    def test_window_composition_compatibility_and_resources(self):
        from PySide6 import QtGui
        from dichromatic_map.ui import controls
        from dichromatic_map.ui.window import DichromaticPatternWindow

        application = controls.create_application()
        window = DichromaticPatternWindow(PatternParameters(), worker_count=1)
        try:
            self.assertIs(window.parameters, window.state.parameters)
            self.assertIs(window.view_box, window.plot.view_box)
            self.assertIs(window.grain_layer_checks, window.controls.grain_layer_checks)
            window.display_rotation_deg = 12.5
            self.assertEqual(window.state.display_rotation_deg, 12.5)
            window.controls.grain_layer_checks[0][0].setChecked(False)
            self.assertNotIn(0, window.state.visible_grain_layers[0])
            self.assertIn(0, window.state.visible_grain_layers[1])
            resource_dir = Path(controls.__file__).parent / "resources"
            for name in ("chevron-down.svg", "chevron-up.svg"):
                path = resource_dir / "icons" / name
                self.assertTrue(path.is_file())
                self.assertIn(path.as_posix(), application.styleSheet())
                self.assertFalse(QtGui.QIcon(str(path)).pixmap(12, 12).isNull())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
