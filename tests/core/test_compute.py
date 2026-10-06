"""Background scheduling must progress and invalidate work without UI polls."""

from concurrent.futures import Future, ThreadPoolExecutor
import os
import threading

import numpy as np
import pytest

from dichromatic_map import compute, strain
from dichromatic_map.crystal import ProjectedGrain, projected_columns


@pytest.fixture(autouse=True)
def isolated_search_caches():
    strain.clear_strain_caches()
    yield
    strain.clear_strain_caches()


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
    monkeypatch.setattr(compute, "NEAR_PARALLEL_CANDIDATES", 16)
    monkeypatch.setattr(compute, "get_cached_cell_search", lambda *_args: None)
    monkeypatch.setattr(compute, "cache_cell_search", lambda *_args: None)
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


@pytest.mark.parametrize("candidate_count,capacity,expected", [(120, 2, 1), (280, 2, 2), (280, 1, 1)])
def test_near_search_concurrency_matches_size_and_thread_capacity(
    monkeypatch, candidate_count, capacity, expected,
):
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    active = maximum = 0
    lock = threading.Lock()

    def solve(*_args):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == expected:
                started.set()
        try:
            assert release.wait(10)
            return os.getpid(), []
        finally:
            with lock:
                active -= 1

    def pareto(_cells):
        completed.set()
        return []

    monkeypatch.setattr(compute, "candidate_vectors", lambda *_args: (
        np.zeros((candidate_count, 2)), np.zeros((candidate_count, 2)),
    ))
    monkeypatch.setattr(compute, "solve_cells_chunk", solve)
    monkeypatch.setattr(compute, "pareto_cells", pareto)
    with ThreadPoolExecutor(max_workers=capacity) as executor:
        search = compute.NearSearch(4, executor)
        try:
            search.request(39.5, 2, 12)
            assert started.wait(10)
            assert search.active_workers == expected
            with search._lock:
                assert len(search.running) == expected
            release.set()
            assert completed.wait(10)
            assert search.poll() == []
            assert maximum == expected
        finally:
            release.set()
            search.close()


def test_near_search_reuses_complete_cache_without_starting_workers(monkeypatch):
    cells = strain.find_strained_cells(39.5, 2, 12)
    assert cells
    expected = cells[0].m1.copy()

    def unexpected(*_args):
        pytest.fail("A complete cached search must not start numerical jobs")

    monkeypatch.setattr(compute, "candidate_vectors", unexpected)
    search = compute.NearSearch(4)
    try:
        search.request(39.5, 2, 12)
        assert search.busy
        assert search.executor._max_workers == 2
        assert not search.executor._threads
        result = search.poll()
        assert not search.busy
        result[0].m1[:] = 0
        result.clear()
        search.request(39.5, 2, 12)
        np.testing.assert_array_equal(search.poll()[0].m1, expected)
        strain.cache_cell_search([], 12, 2, 12)
        search.request(12, 2, 12)
        assert search.poll() == []
    finally:
        search.close()


def test_cached_latest_search_discards_running_stale_preparation(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    strain.cache_cell_search([], 2, 2, 12)

    def prepare(*_args):
        started.set()
        assert release.wait(10)
        return np.empty((0, 2)), np.empty((0, 2))

    monkeypatch.setattr(compute, "candidate_vectors", prepare)
    with ThreadPoolExecutor(max_workers=1) as executor:
        search = compute.NearSearch(4, executor)
        try:
            search.request(1, 2, 12)
            assert started.wait(10)
            search.request(2, 2, 12)
            assert search.poll() == []
            assert not search.busy
        finally:
            release.set()
            search.close()
    assert strain.get_cached_cell_search(1, 2, 12) is None


def test_layer_packets_preserve_phase_ids_order_and_contiguous_views():
    grain = ProjectedGrain(
        np.arange(20).reshape(10, 2), np.repeat(np.arange(5), 2),
        np.arange(30).reshape(10, 3), 5,
    )
    adjacent = compute.grain_layer_subset(grain, [1, 2])
    assert adjacent.layer_count == 5
    assert np.shares_memory(adjacent.positions, grain.positions)
    np.testing.assert_array_equal(adjacent.layers, [1, 1, 2, 2])
    assert compute.grain_layer_subset(grain, range(5)) is grain
    disjoint = compute.grain_layer_subset(grain, [4, 0])
    np.testing.assert_array_equal(disjoint.half_indices, grain.half_indices[[0, 1, 8, 9]])
    shuffled = np.array([9, 0, 4, 1, 7, 2, 6, 5, 3, 8])
    unordered = ProjectedGrain(
        grain.positions[shuffled], grain.layers[shuffled], grain.half_indices[shuffled], 5,
    )
    selected = compute.grain_layer_subset(unordered, [3, 0])
    expected = np.isin(unordered.layers, [3, 0])
    np.testing.assert_array_equal(selected.positions, unordered.positions[expected])
    np.testing.assert_array_equal(selected.half_indices, unordered.half_indices[expected])
    assert len(compute.grain_layer_subset(unordered, [99]).positions) == 0


def test_layer_packets_match_full_grains_and_index_each_batch_once(monkeypatch):
    grains = [projected_columns(10, 8, angle, lattice="SC", axis="2 3 5") for angle in (7.5, 7.5)]
    layers = np.array([1, 3, 5])
    packets = [compute.grain_layer_subset(grain, layers) for grain in grains]
    _, full = compute.coincidence_layers_worker(*grains, 1e-6, layers)
    _, partial = compute.coincidence_layers_worker(*packets, 1e-6, layers)
    assert all(len(points) > 0 for _layer, points in full)
    for (layer, reference), (actual_layer, points) in zip(full, partial):
        assert actual_layer == layer
        np.testing.assert_array_equal(points, reference)
    shifted = ProjectedGrain(
        grains[1].positions + [0.025, 0], grains[1].layers,
        grains[1].half_indices, grains[1].layer_count,
    )
    _, full_pairs = compute.local_match_layers_worker(grains[0], shifted, 0.1, layers)
    _, packet_pairs = compute.local_match_layers_worker(
        packets[0], compute.grain_layer_subset(shifted, layers), 0.1, layers,
    )
    assert len(full_pairs.layers) > 0
    for field in ("first", "second", "layers"):
        np.testing.assert_array_equal(getattr(packet_pairs, field), getattr(full_pairs, field))
    original = ProjectedGrain.layer_selections
    calls = []

    def selections(grain):
        calls.append(id(grain))
        return original(grain)

    monkeypatch.setattr(ProjectedGrain, "layer_selections", selections)
    compute.coincidence_layers_worker(*grains, 1e-6, layers)
    assert calls.count(id(grains[0])) == calls.count(id(grains[1])) == 1


class ControlledPool:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.futures = []
        self.shutdown_calls = []

    def submit(self, function):
        assert function is compute.worker_ready
        future = Future()
        self.futures.append(future)
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


def test_matching_threads_first_then_warm_multilayer_processes(monkeypatch):
    monkeypatch.setattr(compute, "ProcessPoolExecutor", ControlledPool)
    session = compute.ComputeSession()
    session.configure(8)
    pool = session.executor
    try:
        assert pool.kwargs["max_workers"] == 4
        assert session.matching_executor(300_000, 38) == (session.thread_executor(), 1)
        session.warm_matching_pool(30_000, 38)
        session.warm_matching_pool(300_000, 2)
        assert not pool.futures
        session.warm_matching_pool(300_000, 38)
        assert len(pool.futures) == 4
        session.warm_matching_pool(300_000, 38)
        assert len(pool.futures) == 4  # A pending warmup is never duplicated.
        assert session.matching_executor(300_000, 38)[1] == 1
        for future in pool.futures:
            future.set_result(123)
        assert session.matching_executor(300_000, 38) == (pool, 4)
        assert session.matching_executor(300_000, 2)[1] == 1
        assert session.matching_executor(30_000, 38)[1] == 1
    finally:
        session.close()
    assert pool.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    session.warm_matching_pool(300_000, 38)
    assert len(pool.futures) == 4


@pytest.mark.parametrize("failure", ["error", "cancelled", "submit"])
def test_matching_warmup_failure_falls_back_and_configure_cancels(monkeypatch, failure):
    monkeypatch.setattr(compute, "ProcessPoolExecutor", ControlledPool)
    session = compute.ComputeSession()
    session.configure(4)
    pool = session.executor
    try:
        if failure == "submit":
            monkeypatch.setattr(pool, "submit", lambda *_args: (_ for _ in ()).throw(RuntimeError("closed")))
        session.warm_matching_pool(300_000, 38)
        for index, future in enumerate(pool.futures):
            if index == 0 and failure == "error":
                future.set_exception(RuntimeError("startup failed"))
            elif index == 0 and failure == "cancelled":
                future.cancel()
            else:
                future.set_result(123)
        assert session.matching_executor(300_000, 38)[1] == 1
        session.configure(2)
        assert session.matching_warmup_futures == []
        assert not session.matching_warmup_failed
        assert pool.shutdown_calls
        next_pool = session.executor
        session.warm_matching_pool(300_000, 38)
        assert len(next_pool.futures) == 2
        session.close()
        assert all(future.cancelled() for future in next_pool.futures)
    finally:
        session.close()
