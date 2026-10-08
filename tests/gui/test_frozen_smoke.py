"""Release smoke checks must distinguish threads from actual spawned workers."""

import os
from pathlib import Path
import runpy

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytestmark = pytest.mark.gui

check_spawned_workers = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/frozen_smoke.py")
)["check_spawned_workers"]


@pytest.mark.parametrize("lattice", ("FCC", "BCC", "SC"))
def test_release_smoke_explicitly_checks_processes_after_thread_render(gui, lattice):
    window = gui.window(workers=2, lattice=lattice, axis="100", width=6, height=4)
    window._start_parallel_regeneration(compute_coincidences=True)
    gui.settle(window)
    assert window.compute.worker_process_ids == {os.getpid()}
    workers = check_spawned_workers(window)
    assert workers and os.getpid() not in workers
