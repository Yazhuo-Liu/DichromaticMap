"""Validate the artifacts users install, independently of the source checkout."""

from contextlib import redirect_stdout
from email.parser import BytesParser
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile

import build
import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.packaging


def run(command, *, cwd, env=None, timeout=120):
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


@pytest.fixture(scope="module")
def distributions():
    with tempfile.TemporaryDirectory(prefix="dichromatic-distributions-") as directory:
        workspace = Path(directory)
        project = workspace / "project"
        project.mkdir()
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "CONTRIBUTING.md", "main.py"):
            shutil.copy2(ROOT / name, project / name)
        for name in ("src", "tests", "scripts", "docs"):
            shutil.copytree(
                ROOT / name, project / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"),
            )
        output = workspace / "dist"
        output.mkdir()
        archive = Path(build.ProjectBuilder(str(project)).build("sdist", str(output)))
        unpacked = workspace / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive) as source:
            members = source.getmembers()
            names = [member.name for member in members]
            # Extract only regular files/directories and keep paths inside the temporary tree.
            for member in members:
                parts = PurePosixPath(member.name).parts
                assert not member.name.startswith("/") and ".." not in parts
                destination = unpacked.joinpath(*parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    assert member.isfile(), f"Unexpected archive entry: {member.name}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as stream:
                        destination.write_bytes(stream.read())
        source_root = unpacked / PurePosixPath(names[0]).parts[0]
        # Building from the sdist detects omitted source files and UI resources.
        wheel = Path(build.ProjectBuilder(str(source_root)).build("wheel", str(output)))
        yield workspace, source_root, wheel


def test_source_distribution_includes_validation_and_documentation(distributions):
    _, source, _ = distributions
    for name in (
        "LICENSE", "README.md", "CONTRIBUTING.md", "main.py", "pyproject.toml",
        "scripts/run_tests.py", "tests/conftest.py", "docs/en/README.md",
        "docs/zh/README.md", "docs/images/gui-overview.png",
    ):
        assert (source / name).is_file(), f"Missing from sdist: {name}"
    for suite in ("core", "gui", "packaging"):
        assert list((source / "tests" / suite).glob("test_*.py"))
    assert (source / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_wheel_contains_license_resources_and_correct_metadata(distributions):
    _, _, wheel = distributions
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        assert metadata["Name"] == "dichromatic-map"
        assert metadata["License-Expression"] == "MIT"
        assert "LICENSE" in metadata.get_all("License-File", [])
        assert "gui" in metadata.get_all("Provides-Extra", [])
        python_versions = SpecifierSet(metadata["Requires-Python"])
        assert "3.10" in python_versions and "3.9" not in python_versions
        dependencies = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
        assert any(item.name.lower() == "numpy" and item.marker is None for item in dependencies)
        for name in ("pyside6", "pyqtgraph"):
            assert any(
                item.name.lower() == name and item.marker is not None
                and item.marker.evaluate({"extra": "gui"})
                and not item.marker.evaluate({"extra": ""})
                for item in dependencies
            ), f"Missing optional GUI dependency: {name}"
        license_path = next(name for name in names if name.endswith(".dist-info/licenses/LICENSE"))
        assert archive.read(license_path) == (ROOT / "LICENSE").read_bytes()
        for resource in ("theme.qss", "icons/chevron-down.svg", "icons/chevron-up.svg"):
            assert archive.read("dichromatic_map/ui/resources/" + resource)
        assert not any(name.split("/")[0] in {"tests", "scripts", "docs"} for name in names)


def test_installed_wheel_api_workers_and_console_help(distributions):
    workspace, _, wheel = distributions
    installed = workspace / "installed"
    outside = workspace / "outside-checkout"
    outside.mkdir()
    run(
        [sys.executable, "-B", "-m", "pip", "install", "--disable-pip-version-check",
         "--no-deps", "--no-compile", "--no-index", "--target", str(installed), str(wheel)],
        cwd=outside,
    )
    # Exclude the checkout and its editable import path; verify every package origin below.
    environment = dict(os.environ, PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE="1")
    program = outside / "verify_installed.py"
    program.write_text(r"""
import importlib.abc
import multiprocessing
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6', 'pyqtgraph', 'matplotlib'}:
            raise AssertionError('Unexpected GUI dependency: ' + fullname)

sys.meta_path.insert(0, NoGui())
import numpy as np
import dichromatic_map
from dichromatic_map import count_cell_atoms, get_geometry, local_near_pairs, projected_columns
from dichromatic_map.compute import generate_grain_worker, worker_initializer
from dichromatic_map.__main__ import main as cli_main


def worker_origin():
    return str(Path(dichromatic_map.__file__).resolve())


def main():
    target = Path(sys.argv[1]).resolve()
    assert Path(dichromatic_map.__file__).resolve().is_relative_to(target)
    assert get_geometry('FCC', '110').layer_count == 2
    grain = projected_columns(4, 3, 0, lattice='BCC', axis='100')
    assert len(grain.positions) > 0
    assert len(local_near_pairs(grain, grain, 0.05).layers) == 0
    vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    counts = count_cell_atoms(vertices, 0, (np.eye(2), np.eye(2)), 'BCC', '100', layer=0)
    np.testing.assert_array_equal(counts.half_open, [[4, 0], [4, 0]])
    np.testing.assert_array_equal(counts.interior + counts.boundary, [[9, 0], [9, 0]])
    assert get_geometry('SC', '100').layer_count == 1
    sc_grain = projected_columns(4, 3, 0, lattice='SC', axis='100')
    assert len(sc_grain.positions) > 0
    np.testing.assert_array_equal(sc_grain.layers, np.zeros(len(sc_grain.positions), dtype=int))
    np.testing.assert_allclose(sc_grain.positions, np.rint(sc_grain.positions), atol=1e-12)
    sc_counts = count_cell_atoms(vertices, 0, (np.eye(2), np.eye(2)), 'SC', '100', layer=0)
    np.testing.assert_array_equal(sc_counts.half_open, [[4], [4]])
    np.testing.assert_array_equal(sc_counts.interior + sc_counts.boundary, [[9], [9]])
    arguments = (4.0, 3.0, 0.0, (0.0, 0.0), None, 'BCC', '100')
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_initializer) as executor:
        pid, generated = executor.submit(generate_grain_worker, *arguments).result(30)
        assert pid != os.getpid()
        np.testing.assert_allclose(generated.positions, grain.positions)
        sc_arguments = (4.0, 3.0, 0.0, (0.0, 0.0), None, 'SC', '100')
        sc_pid, sc_generated = executor.submit(generate_grain_worker, *sc_arguments).result(30)
        assert sc_pid != os.getpid()
        assert sc_generated.layer_count == 1
        np.testing.assert_allclose(sc_generated.positions, sc_grain.positions)
        np.testing.assert_array_equal(sc_generated.layers, sc_grain.layers)
        assert Path(executor.submit(worker_origin).result(30)).is_relative_to(target)
    sys.argv = ['dichromatic-map', '--help']
    try:
        cli_main()
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError('CLI help did not exit')
    print('Installed API and spawned workers verified without GUI imports.')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
""", encoding="utf-8")
    result = run([sys.executable, "-B", str(program), str(installed)], cwd=outside, env=environment)
    assert "Installed API and spawned workers verified" in result.stdout
    scripts = [
        installed / directory / filename
        for directory in ("bin", "Scripts")
        for filename in ("dichromatic-map", "dichromatic-map.exe")
    ]
    entry_point = next((path for path in scripts if path.is_file()), None)
    assert entry_point is not None, "pip did not install the console entry point"
    help_result = run([str(entry_point), "--help"], cwd=outside, env=environment)
    assert "--workers" in help_result.stdout and "--save" in help_result.stdout
    lattice_choices = re.search(r"--lattice \{([^}]+)\}", help_result.stdout)
    assert lattice_choices is not None, help_result.stdout
    assert set(lattice_choices.group(1).split(",")) == {"FCC", "BCC", "SC"}


def document_anchors(content):
    anchors = set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', content))
    without_code = re.sub(r"^```.*?^```[^\n]*$", "", content, flags=re.M | re.S)
    duplicates = {}
    for heading in re.findall(r"^#{1,6} (.+)$", without_code, flags=re.M):
        slug = re.sub(r"[^\w\s-]", "", heading.strip().lower()).replace(" ", "-")
        count = duplicates.get(slug, 0)
        duplicates[slug] = count + 1
        anchors.add(f"{slug}-{count}" if count else slug)
    return anchors


@pytest.mark.parametrize("document", [
    "README.md", "CONTRIBUTING.md", "docs/en/README.md", "docs/zh/README.md",
    "docs/en/development.md", "docs/zh/development.md",
])
def test_documentation_links_and_python_examples(document):
    path = ROOT / document
    content = path.read_text(encoding="utf-8")
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", content):
        link = urlsplit(target)
        if link.scheme or link.netloc:
            # The PyPI README uses absolute repository URLs. Validate their
            # current-branch targets locally without fetching external pages.
            prefixes = {
                "github.com": "/Yazhuo-Liu/DichromaticMap/blob/main/",
                "raw.githubusercontent.com": "/Yazhuo-Liu/DichromaticMap/main/",
            }
            prefix = prefixes.get(link.netloc.lower())
            if link.scheme not in {"http", "https"} or not prefix or not link.path.startswith(prefix):
                continue
            destination = ROOT / unquote(link.path[len(prefix):])
        else:
            destination = path.parent / unquote(link.path) if link.path else path
        assert destination.exists(), f"Broken link: {document}: {target}"
        if link.fragment and destination.suffix == ".md":
            assert unquote(link.fragment) in document_anchors(destination.read_text(encoding="utf-8")), (
                f"Broken anchor: {document}: {target}"
            )
    for index, example in enumerate(re.findall(r"^```python\n(.*?)^```", content, re.M | re.S)):
        with redirect_stdout(io.StringIO()):
            exec(compile(example, f"{document}:example-{index}", "exec"), {})
