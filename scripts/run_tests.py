"""Run the supported test suites with bounded numerical thread usage.

Examples:
    python scripts/run_tests.py
    python scripts/run_tests.py --suite core
    python scripts/run_tests.py --suite gui -- -k restore -v
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


SUITES = {
    "all": ("tests/core", "tests/gui", "tests/packaging"),
    "core": ("tests/core",),
    "gui": ("tests/gui",),
    "packaging": ("tests/packaging",),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="all")
    arguments, pytest_arguments = parser.parse_known_args()
    if pytest_arguments[:1] == ["--"]:
        pytest_arguments.pop(0)

    required = ["pytest", "numpy"]
    if arguments.suite in ("all", "gui"):
        required.extend(("PySide6", "pyqtgraph"))
    if arguments.suite in ("all", "packaging"):
        required.extend(("build", "setuptools", "wheel"))
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        extra = "test,gui" if arguments.suite in ("all", "gui") else "test"
        parser.error(
            f"Missing test dependencies: {', '.join(missing)}. "
            f'Install from the project root with: python -m pip install -e ".[{extra}]"'
        )

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    environment.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, "-B", "-m", "pytest",
        *SUITES[arguments.suite], *pytest_arguments,
    ]
    return subprocess.run(command, cwd=root, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
