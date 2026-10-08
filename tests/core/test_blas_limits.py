"""Numerical thread tasks share and restore the process-wide BLAS limit."""

from concurrent.futures import ThreadPoolExecutor
import builtins
import threading

import numpy as np
import pytest

from dichromatic_map import compute, strain


class FakeController:
    def __init__(self):
        self.threads = 5
        self.limits = 0
        self.restores = 0

    def limit(self, *, limits, user_api):
        assert limits == 1
        assert user_api == "blas"
        original = self.threads
        self.threads = limits
        self.limits += 1
        controller = self

        class Limit:
            def restore_original_limits(self):
                controller.threads = original
                controller.restores += 1

        return Limit()


@pytest.fixture
def fake_blas(monkeypatch):
    controller = FakeController()
    discoveries = []

    def discover():
        discoveries.append(True)
        return controller

    monkeypatch.setattr(compute, "_blas_controller", discover)
    monkeypatch.setattr(compute, "_thread_blas_limit", compute._SharedBlasLimit())
    return controller, discoveries


def test_nested_and_failed_tasks_restore_only_after_the_last_scope(fake_blas):
    controller, discoveries = fake_blas
    with pytest.raises(ValueError, match="failed task"):
        with compute._thread_blas_limit.active():
            assert controller.threads == 1
            with compute._thread_blas_limit.active():
                assert controller.limits == 1
            assert controller.threads == 1
            assert controller.restores == 0
            raise ValueError("failed task")
    assert controller.threads == 5
    assert controller.restores == 1
    # Each nonoverlapping task captures the current external setting.
    controller.threads = 3
    assert compute.run_numerical_task(lambda: controller.threads) == 1
    assert controller.threads == 3
    assert controller.restores == 2
    assert discoveries == [True]


def test_sessions_share_limits_until_running_tasks_finish_after_close(fake_blas):
    controller, discoveries = fake_blas
    first = compute.ComputeSession()
    second = compute.ComputeSession()
    first_executor = first.thread_executor()
    second_executor = second.thread_executor()
    started = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    assert not discoveries  # An idle session does not change external NumPy.

    def task(index):
        assert controller.threads == 1
        started[index].set()
        assert release[index].wait(10)
        assert controller.threads == 1
        if index == 1:
            raise ValueError("failed overlapping task")
        return index

    try:
        one = first_executor.submit(task, 0)
        assert started[0].wait(10)
        two = second_executor.submit(task, 1)
        assert started[1].wait(10)
        assert controller.limits == 1
        first.close()
        second.close()
        assert controller.threads == 1
        release[0].set()
        assert one.result(timeout=10) == 0
        assert controller.restores == 0
        assert controller.threads == 1
        release[1].set()
        with pytest.raises(ValueError, match="failed overlapping task"):
            two.result(timeout=10)
        assert controller.threads == 5
        assert controller.restores == 1
        assert discoveries == [True]
    finally:
        for event in release:
            event.set()
        first.close()
        second.close()
        first_executor.shutdown(wait=True, cancel_futures=True)
        second_executor.shutdown(wait=True, cancel_futures=True)


def test_near_search_limits_a_caller_supplied_thread_executor(monkeypatch, fake_blas):
    controller, _discoveries = fake_blas
    prepared = threading.Event()

    def prepare(*_args):
        assert controller.threads == 1
        prepared.set()
        return np.empty((0, 2)), np.empty((0, 2))

    monkeypatch.setattr(compute, "candidate_vectors", prepare)
    monkeypatch.setattr(compute, "get_cached_cell_search", lambda *_args: None)
    monkeypatch.setattr(compute, "cache_cell_search", lambda *_args: None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        search = compute.NearSearch(1, executor)
        try:
            search.request(39.5, 2, 12)
            assert prepared.wait(10)
        finally:
            search.close()
    assert controller.threads == 5
    assert controller.limits == controller.restores == 1


def test_numpy_only_install_still_runs_thread_tasks_and_worker_initializer(monkeypatch):
    original_import = builtins.__import__
    imports = []

    def without_threadpoolctl(name, *args, **kwargs):
        if name == "threadpoolctl":
            imports.append(name)
            raise ImportError("optional threadpoolctl is absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_threadpoolctl)
    monkeypatch.setattr(compute, "_thread_blas_limit", compute._SharedBlasLimit())
    session = compute.ComputeSession()
    executor = session.thread_executor()
    try:
        assert executor.submit(lambda value: value + 1, value=2).result(timeout=10) == 3
        with pytest.raises(ValueError, match="failed without optional dependency"):
            executor.submit(
                lambda: (_ for _ in ()).throw(ValueError("failed without optional dependency")),
            ).result(timeout=10)
        assert imports == ["threadpoolctl"]
        compute.worker_initializer()
        assert imports == ["threadpoolctl", "threadpoolctl"]
    finally:
        session.close()
        executor.shutdown(wait=True, cancel_futures=True)


def test_process_initializer_keeps_a_blas_only_limit(fake_blas, monkeypatch):
    controller, discoveries = fake_blas
    monkeypatch.setattr(compute, "_blas_limit", None, raising=False)
    compute.worker_initializer()
    assert controller.threads == 1
    assert controller.limits == 1
    assert controller.restores == 0
    assert discoveries == [True]


def test_real_blas_limit_restores_external_configuration_and_scientific_results(monkeypatch):
    threadpoolctl = pytest.importorskip("threadpoolctl")
    monkeypatch.setattr(compute, "_thread_blas_limit", compute._SharedBlasLimit())
    session = compute.ComputeSession()
    executor = session.thread_executor()
    strain.clear_strain_caches()

    def calculate():
        pools = [pool for pool in threadpoolctl.threadpool_info() if pool["user_api"] == "blas"]
        _pid, grain = compute.generate_grain_worker(16, 12, 19.75, (0, 0))
        first, second = strain.candidate_vectors(39.5, 2, 12)
        cells = strain.pareto_cells([
            cell
            for start in range(0, len(first), 8)
            for cell in strain.solve_cells_chunk(39.5, 2, first, second, start, start + 8)[1]
        ])
        return pools, grain, cells

    try:
        with threadpoolctl.threadpool_limits(limits=2, user_api="blas"):
            before, expected_grain, expected_cells = calculate()
            assert before and all(pool["num_threads"] == 2 for pool in before)
            during, actual_grain, actual_cells = executor.submit(calculate).result(timeout=10)
            assert all(pool["num_threads"] == 1 for pool in during)
            restored = [pool for pool in threadpoolctl.threadpool_info() if pool["user_api"] == "blas"]
            assert all(pool["num_threads"] == 2 for pool in restored)
        np.testing.assert_array_equal(actual_grain.half_indices, expected_grain.half_indices)
        np.testing.assert_array_equal(actual_grain.layers, expected_grain.layers)
        np.testing.assert_allclose(actual_grain.positions, expected_grain.positions, rtol=0, atol=1e-12)
        assert actual_cells and len(actual_cells) == len(expected_cells)
        for actual, expected in zip(actual_cells, expected_cells):
            assert actual.atoms == expected.atoms
            np.testing.assert_array_equal(actual.m1, expected.m1)
            np.testing.assert_array_equal(actual.m2, expected.m2)
            np.testing.assert_allclose(actual.f1, expected.f1, rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual.f2, expected.f2, rtol=0, atol=1e-12)
    finally:
        session.close()
        executor.shutdown(wait=True, cancel_futures=True)
        strain.clear_strain_caches()
