"""Exercise bundled resources, GUI exports, sessions and real spawned workers."""

import json
import os
from pathlib import Path
import sys
import time


def check_spawned_workers(window):
    """Verify the frozen process entry independently of interactive scheduling."""
    import numpy as np
    from dichromatic_map.compute import generate_grain_worker, coincidence_layers_worker

    executor = window.compute.executor
    assert executor is not None, "No process executor configured for the smoke check"
    center, width, height, _bounds = window.plot._buffer_geometry()
    state = window.state
    futures = [executor.submit(
        generate_grain_worker, width, height, sign * state.angle_deg / 2,
        tuple(center), state.deformations[grain], state.geometry.lattice,
        state.geometry.axis, state.translations[grain],
    ) for grain, sign in enumerate((1, -1))]
    results = [future.result(timeout=60) for future in futures]
    workers = {pid for pid, _grain in results}
    assert workers and os.getpid() not in workers, "Process work ran in the viewer process"
    for (_pid, generated), expected in zip(results, state.grains):
        for field in ("positions", "layers", "half_indices"):
            np.testing.assert_array_equal(getattr(generated, field), getattr(expected, field))
        assert generated.layer_count == expected.layer_count
    pid, sites = executor.submit(
        coincidence_layers_worker, results[0][1], results[1][1], 1e-6,
        tuple(range(state.geometry.layer_count)),
    ).result(timeout=60)
    assert pid != os.getpid(), "Matching work ran in the viewer process"
    workers.add(pid)
    for layer, points in sites:
        np.testing.assert_array_equal(points, state.coincident_points[layer])
    return workers


def run(output):
    import numpy as np
    from dichromatic_map.completion import cell_completion_candidates
    from dichromatic_map.state import PatternParameters
    from dichromatic_map.ui import controls
    from dichromatic_map.ui._qt import QtCore, QtGui
    from dichromatic_map.ui.completion import CellCompletionDialog
    from dichromatic_map.ui.window import DichromaticPatternWindow

    if not getattr(sys, "frozen", False):
        raise RuntimeError("Run this check through the built executable")
    app = controls.create_application()
    resources = Path(controls.__file__).with_name("resources")
    assert (resources / "theme.qss").read_text(encoding="utf-8")
    for name in ("chevron-down.svg", "chevron-up.svg"):
        assert not QtGui.QIcon(str(resources / "icons" / name)).pixmap(16, 16).isNull()

    def settle(window):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 10)
            if (window.compute.parallel_stage is None
                    and not window.state.csl_updating
                    and not window.view_refresh_timer.isActive()
                    and not window.coincidence_timer.isActive()):
                assert window.state.render_error is None
                assert window.compute.worker_count == 2, "Parallel computation fell back to serial"
                return
            time.sleep(0.01)
        raise TimeoutError(window.controls.status_label.text())

    workers = set()
    for lattice in ("FCC", "BCC", "SC"):
        window = DichromaticPatternWindow(
            PatternParameters(lattice=lattice, axis="100", width=6, height=4),
            worker_count=2, show_close_citation=False,
        )
        try:
            window.show()
            # Small interactive buffers use the thread pipeline. Exercise it
            # separately from the explicit frozen process entry check below.
            window._start_parallel_regeneration(compute_coincidences=True)
            settle(window)
            assert all(len(grain.positions) for grain in window.state.grains)
            assert window.compute.worker_process_ids == {os.getpid()}, "Small buffer left the thread pipeline"
            workers.update(check_spawned_workers(window))
            window.save(output / f"{lattice}.png")
            window.plot.save(output / f"{lattice}-clean.png", clean=True)
            session = output / f"{lattice}.dmap"
            window.save_session(session)
            window.load_session(session)
            window._start_parallel_regeneration(compute_coincidences=True)
            settle(window)
            assert window.state.geometry.lattice == lattice
        finally:
            window.close()
            app.processEvents()

    edge = np.array([[[0., 0.], [1., 0.]], [[0., 0.], [1., 0.]]])
    candidates = cell_completion_candidates(edge, 0, lattice="SC", axis="100")
    assert candidates and candidates[0].fit is not None
    dialog = CellCompletionDialog(candidates, 2, ("#2563eb", "#ef4444"), 0)
    dialog.show()
    app.processEvents()
    assert dialog.use_button.isEnabled()
    assert dialog.grab().save(str(output / "completion.png"))
    dialog.reject()
    app.processEvents()
    (output / "result.json").write_text(json.dumps({
        "status": "ok", "frozen": True, "worker_pids": sorted(workers),
        "lattices": ["FCC", "BCC", "SC"],
    }, indent=2), encoding="utf-8")
