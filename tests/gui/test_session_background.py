"""Session transactions stay responsive and publish only validated snapshots."""

import threading

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from dichromatic_map import exports, session as session_files
from dichromatic_map.state import PatternParameters
from dichromatic_map.ui import session as session_module
from dichromatic_map.ui import window as window_module
from test_workflows import pick_exact_cell

pytestmark = pytest.mark.gui


def test_initial_grains_are_generated_off_main_thread(gui, monkeypatch):
    release = threading.Event()
    started = threading.Event()
    responsive = threading.Event()
    threads = []
    original = window_module.generate_grain_worker

    def worker(*args):
        threads.append(threading.get_ident())
        started.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(window_module, "generate_grain_worker", worker)
    main = threading.get_ident()
    window = window_module.DichromaticPatternWindow(
        PatternParameters(), worker_count=1, show_close_citation=False,
    )
    gui.windows.append(window)
    try:
        assert started.wait(10)
        assert window.state.grains == []
        gui.QtCore.QTimer.singleShot(0, responsive.set)
        gui.app.processEvents()
        assert responsive.is_set()
        release.set()
        window.show()
        gui.settle(window)
        assert len(window.state.grains) == 2
        assert all(thread != main for thread in threads)
    finally:
        release.set()


def test_import_reuses_two_background_preflight_grains(gui, monkeypatch, tmp_path):
    source = gui.window(angle_deg=13)
    path = tmp_path / "source.dmap"
    source.save_session(path)
    window = gui.window(angle_deg=22)
    calls = []
    original = session_module.projected_columns
    main = threading.get_ident()

    def project(*args, **kwargs):
        grain = original(*args, **kwargs)
        calls.append((threading.get_ident(), grain))
        return grain

    def regenerate(*_args):
        pytest.fail("Import must reuse its validated preflight grains")

    monkeypatch.setattr(session_module, "projected_columns", project)
    monkeypatch.setattr(window_module, "generate_grain_worker", regenerate)
    window.load_session(path)
    assert window.state.angle_deg == 13  # Public API returns after publication.
    gui.settle(window)
    assert len(calls) == 2
    assert all(thread != main for thread, _grain in calls)
    assert all(actual is expected for actual, (_, expected) in zip(window.state.grains, calls))


def test_save_counts_use_background_frozen_inputs_while_qt_runs(gui, monkeypatch, tmp_path):
    window = gui.window(lattice="BCC", axis="100")
    pick_exact_cell(gui, window)
    path = tmp_path / "frozen.dmap"
    release = threading.Event()
    started = threading.Event()
    original = exports.count_cell_atoms
    main = threading.get_ident()
    observed = []

    def count(*args, **kwargs):
        assert threading.get_ident() != main
        started.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    def change_live_inputs():
        if not started.is_set():
            return
        observed.append((path.exists(), window.session.pending_future is not None))
        window.controls.angle_spin.setValue(23)
        release.set()
        timer.stop()

    monkeypatch.setattr(exports, "count_cell_atoms", count)
    timer = gui.QtCore.QTimer()
    timer.setInterval(5)
    timer.timeout.connect(change_live_inputs)
    timer.start()
    try:
        saved_angle = window.state.angle_deg
        window.save_session(path)
        assert observed == [(False, True)]
        saved = session_files.load_session(path)
        assert saved.state.angle_deg == saved_angle
        assert len(saved.state.manual_vertices) == 4
        gui.settle(window)
        assert window.state.angle_deg == 23
    finally:
        timer.stop()
        release.set()


@pytest.mark.parametrize("action", ["edit", "close"])
def test_import_does_not_publish_after_reentrant_edit_or_close(gui, monkeypatch, tmp_path, action):
    source = gui.window(angle_deg=13)
    path = tmp_path / "source.dmap"
    source.save_session(path)
    window = gui.window(angle_deg=22)
    original_state = window.state
    release = threading.Event()
    started = threading.Event()
    original = session_module.SessionController._prepare_load

    def prepare(path):
        started.set()
        assert release.wait(10)
        return original(path)

    def interrupt():
        if not started.is_set():
            return
        timer.stop()
        if action == "close":
            window.close()
        else:
            window.controls.angle_spin.setValue(23)
        release.set()

    monkeypatch.setattr(window.session, "_prepare_load", prepare)
    timer = gui.QtCore.QTimer()
    timer.setInterval(5)
    timer.timeout.connect(interrupt)
    timer.start()
    try:
        with pytest.raises(RuntimeError):
            window.load_session(path)
        assert window.state is original_state
        if action == "edit":
            gui.settle(window)
            assert window.state.angle_deg == 23
    finally:
        timer.stop()
        release.set()


def test_failed_widget_commit_rolls_back_to_original_session(gui, monkeypatch, tmp_path):
    source = gui.window(lattice="SC", axis="111", angle_deg=13)
    path = tmp_path / "source.dmap"
    source.save_session(path)
    window = gui.window(lattice="BCC", axis="100", angle_deg=22)
    state = window.state
    original = window.plot._create_layer_items
    calls = []

    def create():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("widget creation failed")
        return original()

    monkeypatch.setattr(window.plot, "_create_layer_items", create)
    with pytest.raises(RuntimeError, match="widget creation failed"):
        window.load_session(path)
    gui.settle(window)
    assert window.state is state
    assert window.state.geometry.lattice == "BCC"
    assert window.state.angle_deg == 22
    assert window.state.grain_signature == window._geometry_signature()
