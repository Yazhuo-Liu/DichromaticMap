"""Failed background searches and matching pools remain retryable."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from dichromatic_map import compute, matching, strain

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("failure", [ValueError, CancelledError])
def test_near_search_reports_empty_message_failure_and_retries(gui, monkeypatch, failure):
    window = gui.window(angle_deg=22)
    futures = []
    grains = window.state.grains

    def submit(*_args, **_kwargs):
        future = Future()
        futures.append(future)
        return future

    strain.clear_strain_caches()
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(executor, "submit", submit)
        search = window.compute.near_search = compute.NearSearch(1, executor)
        window.state.near_enabled = True
        window.state.near_method = "strain"
        try:
            window._start_near_search()
            if failure is CancelledError:
                assert futures[-1].cancel()
            else:
                futures[-1].set_exception(failure())
            window._poll_near_search()
            assert window.controls.near_info.toPlainText() == (
                "Near-CSL search failed: " + failure.__name__
            )
            assert not search.busy
            assert not window.near_poll_timer.isActive()
            assert all(actual is original for actual, original in zip(window.state.grains, grains))
            assert strain.get_cached_cell_search(*search.args) is None

            window._start_near_search()
            assert len(futures) == 2  # A failed request did not cache no solutions.
            vectors = np.empty((0, 2), dtype=int)
            futures[-1].set_result((vectors, vectors))
            window._poll_near_search()
            assert search.error is None
            assert not window.near_poll_timer.isActive()
            assert "No compatible cell found" in window.controls.near_info.toPlainText()
            assert strain.get_cached_cell_search(*search.args) == []
            gui.settle(window)
        finally:
            search.close()
            strain.clear_strain_caches()


@pytest.mark.parametrize("stage", ["local", "exact"])
@pytest.mark.parametrize("failure", ["submit", "result"])
def test_broken_matching_pool_retires_and_retries_on_threads(gui, monkeypatch, stage, failure):
    window = gui.window(workers=2, angle_deg=22)
    grains = window.state.grains
    reference = (
        matching.local_near_pairs(*grains, distance=window.controls.local_distance_spin.value())
        if stage == "local" else
        matching.same_layer_coincidence_sites(*grains, 1e-6)
    )

    class BrokenPool:
        submissions = shutdowns = 0

        def submit(self, *_args):
            self.submissions += 1
            error = BrokenProcessPool("worker exited unexpectedly")
            if failure == "submit":
                raise error
            future = Future()
            future.set_exception(error)
            return future

        def shutdown(self, **kwargs):
            assert kwargs == {"wait": False, "cancel_futures": True}
            self.shutdowns += 1

    window.compute.executor.shutdown(wait=False, cancel_futures=True)
    pool = BrokenPool()
    window.compute.executor = pool
    ready = Future()
    ready.set_result(123)
    window.compute.matching_warmup_futures = [ready]
    monkeypatch.setattr(compute, "PROCESS_MATCH_POINTS", 0)
    monkeypatch.setattr(compute, "PROCESS_MATCH_LAYERS", 1)

    def synchronous(*_args, **_kwargs):
        pytest.fail("Broken matching processes must retry in the background")

    monkeypatch.setattr(window, "_regenerate_buffer", synchronous)
    if stage == "local":
        window.state.near_enabled = True
        window.state.near_method = "local"
        window._start_local_matching()
    else:
        window._start_parallel_coincidences()
    gui.settle(window)
    assert window.compute.executor is None
    assert window.compute.matching_warmup_failed
    assert window.compute.matching_warmup_futures == []
    assert window.compute.worker_count == 2
    assert pool.shutdowns == 1
    submissions = pool.submissions
    assert all(actual is original for actual, original in zip(window.state.grains, grains))
    assert os.getpid() in window.compute.worker_process_ids
    if stage == "local":
        assert len(reference.layers) > 0
        for field in ("first", "second", "layers"):
            np.testing.assert_array_equal(getattr(window.state.local_pairs, field), getattr(reference, field))
        window._start_local_matching()
    else:
        for actual, expected in zip(window.state.coincident_points, reference):
            np.testing.assert_array_equal(actual, expected)
        window._start_parallel_coincidences()
    gui.settle(window)
    assert pool.submissions == submissions
