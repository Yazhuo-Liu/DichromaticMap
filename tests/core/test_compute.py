"""Background scheduling must progress and invalidate work without UI polls."""

from concurrent.futures import ThreadPoolExecutor
import os
import threading

import numpy as np

from dichromatic_map import compute, strain


def test_near_search_finishes_without_polling_and_matches_serial(monkeypatch):
    finalized = threading.Event()
    original_pareto = strain.pareto_cells

    def pareto(cells):
        result = original_pareto(cells)
        finalized.set()
        return result

    monkeypatch.setattr(compute, "pareto_cells", pareto)
    first, second = strain.candidate_vectors(39.5, 2, 12)
    # Preserve the historical eight-row solve batches. A single larger SVD
    # batch can select a different equivalent cell at roundoff-scale ties.
    expected = original_pareto([
        cell
        for start in range(0, len(first), 8)
        for cell in strain.solve_cells_chunk(
            39.5, 2, first, second, start, start + 8,
        )[1]
    ])
    search = compute.NearSearch(1)
    try:
        search.request(39.5, 2, 12)
        # No poll calls may be needed to start or refill numerical workers.
        assert finalized.wait(10)
        assert search.busy  # The UI has not consumed the result yet.
        actual = search.poll()
        assert not search.busy
        assert search.poll() is None
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            assert result.atoms == reference.atoms
            np.testing.assert_array_equal(result.m1, reference.m1)
            np.testing.assert_array_equal(result.m2, reference.m2)
            np.testing.assert_allclose(result.f1, reference.f1, atol=1e-12)
            np.testing.assert_allclose(result.f2, reference.f2, atol=1e-12)
    finally:
        executor = search.executor
        search.close()
        executor.shutdown(wait=True, cancel_futures=True)


def test_near_search_bounds_stale_work_and_publishes_latest_in_order(monkeypatch):
    release_old = threading.Event()
    old_started = threading.Event()
    latest_tail_started = threading.Event()
    finalized = threading.Event()
    lock = threading.Lock()
    active = maximum = old_count = 0
    prepared = []

    def prepare(angle, *_args):
        prepared.append(angle)
        return np.zeros((24, 2), dtype=int), np.zeros((24, 2), dtype=int)

    def solve(angle, _percent, _first, _second, start, *_args):
        nonlocal active, maximum, old_count
        with lock:
            active += 1
            maximum = max(active, maximum)
            if angle == 1:
                old_count += 1
                if old_count == 2:
                    old_started.set()
        try:
            if angle == 1:
                assert release_old.wait(10)
            elif start == 0:
                # Complete the latest chunks out of their serial order.
                assert latest_tail_started.wait(10)
            elif start == 16:
                latest_tail_started.set()
            return os.getpid(), [(angle, start)]
        finally:
            with lock:
                active -= 1

    def pareto(cells):
        finalized.set()
        return cells

    monkeypatch.setattr(compute, "candidate_vectors", prepare)
    monkeypatch.setattr(compute, "solve_cells_chunk", solve)
    monkeypatch.setattr(compute, "pareto_cells", pareto)
    with ThreadPoolExecutor(max_workers=2) as executor:
        search = compute.NearSearch(2, executor)
        try:
            search.request(1, 2, 12)
            assert old_started.wait(10)
            search.request(2, 2, 12)
            search.request(3, 2, 12)
            assert search.poll() is None
            assert prepared == [1]
            release_old.set()
            assert finalized.wait(10)
            assert search.poll() == [(3, 0), (3, 8), (3, 16)]
            assert prepared == [1, 3]
            assert maximum == 2
            assert search.completed == search.total == 3
        finally:
            release_old.set()
            latest_tail_started.set()
            search.close()


def test_closed_search_does_not_restart_pending_work(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def prepare(angle, *_args):
        calls.append(angle)
        started.set()
        assert release.wait(10)
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    monkeypatch.setattr(compute, "candidate_vectors", prepare)
    with ThreadPoolExecutor(max_workers=1) as executor:
        search = compute.NearSearch(1, executor)
        search.request(1, 2, 12)
        assert started.wait(10)
        search.request(2, 2, 12)
        search.close()
        release.set()
    assert calls == [1]
    assert search.poll() is None
    assert not search.busy
