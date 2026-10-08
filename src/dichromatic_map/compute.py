"""Qt-free worker entry points, executor ownership and asynchronous search.

Workers are module-level functions so a spawned process imports numerical
modules only. The UI coordinates operations and polls completed futures.
"""

from __future__ import annotations
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os
import threading
import numpy as np
from .crystal import ProjectedGrain, projected_columns, get_geometry
from .cells import count_cell_atoms
from .matching import local_near_pairs, same_layer_coincidence_sites
from .strain import (
    candidate_vectors, solve_cells_chunk, pareto_cells,
    get_cached_cell_search, cache_cell_search,
)


# Small interactive buffers avoid process startup and full-array transfers.
PROCESS_MATCH_POINTS = 100_000
PROCESS_MATCH_LAYERS = 4
NEAR_PARALLEL_CANDIDATES = 256


def _blas_controller():
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        return None  # The numerical package still supports NumPy-only installs.
    return ThreadpoolController()


class _SharedBlasLimit:
    """Coordinate the process-wide BLAS setting across numerical tasks.

    Per-thread limits contexts can restore another task's setting. Instead,
    overlapping tasks share one limit, and the last task restores the setting
    captured before the first. Idle viewers and importing this module leave
    an embedding program's NumPy configuration alone.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._users = 0
        self._controller = None
        self._loaded = False
        self._limit = None

    @contextmanager
    def active(self):
        with self._lock:
            if not self._loaded:
                self._controller = _blas_controller()
                self._loaded = True
            if self._users == 0 and self._controller is not None:
                # Cache library discovery; short solve chunks only adjust the
                # already loaded BLAS libraries, rather than scanning each time.
                self._limit = self._controller.limit(limits=1, user_api="blas")
            self._users += 1
        try:
            yield
        finally:
            with self._lock:
                self._users -= 1
                if self._users == 0 and self._limit is not None:
                    limit, self._limit = self._limit, None
                    limit.restore_original_limits()


_thread_blas_limit = _SharedBlasLimit()


def run_numerical_task(function, /, *args, **kwargs):
    """Run an owned background task under the shared, temporary BLAS limit."""
    with _thread_blas_limit.active():
        return function(*args, **kwargs)


class _NumericalThreadPoolExecutor(ThreadPoolExecutor):
    def submit(self, function, /, *args, **kwargs):
        return super().submit(run_numerical_task, function, *args, **kwargs)


def worker_initializer():
    """Avoid each process starting its own BLAS thread team when available."""
    global _blas_limit
    controller = _blas_controller()
    if controller is not None:
        # A spawned numerical process owns this setting until it exits.
        _blas_limit = controller.limit(limits=1, user_api="blas")


def worker_ready():
    """Start a numerical worker without transferring projected grains."""
    return os.getpid()


def local_match_layers_worker(grain1, grain2, distance, layers):
    return os.getpid(), local_near_pairs(grain1, grain2, distance, layers)


def cell_count_worker(*args):
    return os.getpid(), count_cell_atoms(*args)


def generate_grain_worker(
    width: float,
    height: float,
    rotation_deg: float,
    center: tuple[float, float],
    deformation: np.ndarray | None = None,
    lattice: str = "FCC",
    axis: str = "110",
    translation: np.ndarray | None = None,
) -> tuple[int, ProjectedGrain]:
    """Process-pool entry point for one projected grain."""

    return (
        os.getpid(),
        projected_columns(
            width, height, rotation_deg, center, deformation, lattice, axis, translation
        ),
    )


def coincidence_layer_worker(
    grain_1: ProjectedGrain,
    grain_2: ProjectedGrain,
    tolerance: float,
    layer: int,
) -> tuple[int, np.ndarray]:
    """Process-pool entry point for one stacking-layer coincidence search."""

    only_first = grain_1.layer_selections().get(layer, slice(0, 0))
    only_second = grain_2.layer_selections().get(layer, slice(0, 0))
    return os.getpid(), _coincidence_selection(
        grain_1, grain_2, tolerance, only_first, only_second,
    )


def _coincidence_selection(grain_1, grain_2, tolerance, only_first, only_second):
    first_positions = grain_1.positions[only_first]
    second_positions = grain_2.positions[only_second]
    first = ProjectedGrain(
        first_positions,
        np.zeros(len(first_positions), dtype=np.int16),
        grain_1.half_indices[only_first], 1,
    )
    second = ProjectedGrain(
        second_positions,
        np.zeros(len(second_positions), dtype=np.int16),
        grain_2.half_indices[only_second], 1,
    )
    return same_layer_coincidence_sites(first, second, tolerance)[0]


def coincidence_layers_worker(grain_1, grain_2, tolerance, layers):
    """A bounded batch of phases, avoiding one full-array transfer per layer."""
    first = grain_1.layer_selections()
    second = grain_2.layer_selections()
    return os.getpid(), tuple(
        (
            int(layer),
            _coincidence_selection(
                grain_1, grain_2, tolerance,
                first.get(int(layer), slice(0, 0)),
                second.get(int(layer), slice(0, 0)),
            ),
        )
        for layer in layers
    )


def grain_layer_subset(grain, layers, selections=None):
    """Transfer only requested phases, preserving physical IDs and row order."""
    selections = grain.layer_selections() if selections is None else selections
    layers = set(map(int, layers))
    if selections.keys() <= layers:
        return grain
    selected = [selection for layer, selection in selections.items() if layer in layers]
    if not selected:
        selected = [slice(0, 0)]
    if len(selected) == 1:
        indices = selected[0]
    elif all(isinstance(selection, slice) for selection in selected):
        selected.sort(key=lambda selection: selection.start)
        if all(first.stop == second.start for first, second in zip(selected, selected[1:])):
            indices = slice(selected[0].start, selected[-1].stop)
        else:
            indices = np.concatenate([
                np.arange(selection.start, selection.stop) for selection in selected
            ])
    else:
        indices = np.sort(np.concatenate([
            np.arange(selection.start, selection.stop)
            if isinstance(selection, slice) else selection
            for selection in selected
        ]))
    return ProjectedGrain(
        grain.positions[indices], grain.layers[indices], grain.half_indices[indices],
        grain.layer_count,
    )


class NearSearch:
    """Bounded asynchronous search; stale requests never publish results.

    Preparation and pair solves run off the UI thread. At most workers jobs
    are in flight. Invalidation drops pending jobs and lets bounded running
    chunks finish before submitting the next request.
    """

    def __init__(self, workers=1, executor=None):
        self.workers = max(1, int(workers))
        # Cold process startup outweighs bounded interactive cell searches.
        # Larger candidate sets benefit from two threads without that startup.
        self.executor = executor if isinstance(executor, ThreadPoolExecutor) else None
        self.owns_executor = self.executor is None
        self.active_workers = 1
        self._executor_capacity = (
            min(self.workers, self.executor._max_workers)
            if self.executor is not None else min(2, self.workers)
        )
        self.generation = 0
        self.jobs = deque()
        self.running = {}
        self.parts = {}
        self.total = self.completed = 0
        self.busy = False
        self.error = None
        self.process_ids = set()
        self._lock = threading.RLock()
        self._result = None
        self._closed = False

    def cancel(self):
        with self._lock:
            self._cancel_locked()

    def _cancel_locked(self):
        self.generation += 1
        self.jobs.clear()
        self.parts.clear()
        self.busy = False
        self.error = None
        self.total = self.completed = 0
        self._result = None
        for future in list(self.running):
            future.cancel()

    def request(self, angle, percent, extent, lattice="FCC", axis="110"):
        geometry = get_geometry(lattice, axis)
        with self._lock:
            if self._closed:
                raise RuntimeError("Near-CSL search is closed")
            self._cancel_locked()
            self.args = (angle, percent, extent, geometry.lattice, geometry.axis)
            self.busy = True
            self.active_workers = 1
            if self.executor is None:
                self.executor = _NumericalThreadPoolExecutor(
                    max_workers=self._executor_capacity,
                )
            cached = get_cached_cell_search(*self.args)
            if cached is not None:
                self._result = cached
                return
            self.jobs.append(("prepare", candidate_vectors, self.args))
            self._submit_locked()

    def _submit_locked(self):
        """Refill workers on completion, independently of GUI progress polling."""
        try:
            while not self._closed and self.jobs and len(self.running) < self.active_workers:
                stage, function, args = self.jobs.popleft()
                if isinstance(self.executor, _NumericalThreadPoolExecutor):
                    future = self.executor.submit(function, *args)
                else:
                    future = self.executor.submit(run_numerical_task, function, *args)
                self.running[future] = (
                    self.generation, stage, args[4] if stage == "solve" else -1
                )
                future.add_done_callback(self._completed)
            if (
                self.busy and not self.error and self._result is None
                and not self.jobs and not self.running
            ):
                # Equal-area/strain candidates retain serial search order,
                # independently of which process happens to finish first.
                self._result = pareto_cells([
                    cell for start in sorted(self.parts) for cell in self.parts[start]
                ])
                cache_cell_search(self._result, *self.args)
                self.parts.clear()
        except Exception as error:
            self._cancel_locked()
            # Keep busy until the UI has observed the error, as with results.
            self.busy = True
            self.error = str(error)

    def _completed(self, future):
        with self._lock:
            generation, stage, start = self.running.pop(future)
            if generation == self.generation:
                try:
                    result = future.result()
                    if stage == "prepare":
                        i, j = result
                        self.active_workers = (
                            min(2, self._executor_capacity)
                            if len(i) >= NEAR_PARALLEL_CANDIDATES else 1
                        )
                        for start in range(0, len(i), 8):
                            self.jobs.append(("solve", solve_cells_chunk, (
                                self.args[0], self.args[1], i, j, start, start + 8,
                                self.args[3], self.args[4],
                            )))
                        self.total = len(self.jobs)
                    else:
                        pid, cells = result
                        self.process_ids.add(pid)
                        self.parts[start] = cells
                        self.completed += 1
                except Exception as error:
                    self._cancel_locked()
                    self.busy = True
                    self.error = str(error)
            self._submit_locked()

    def poll(self):
        with self._lock:
            if self.error:
                self.busy = False
            if self._result is not None:
                result, self._result = self._result, None
                self.busy = False
                return result
        return None

    def close(self):
        with self._lock:
            self._closed = True
            self._cancel_locked()
            executor, self.executor = self.executor, None
        if self.owns_executor and executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


class ComputeSession:
    """Own the shared executors and pending work for one viewer session."""

    def __init__(self):
        self.worker_count = 1
        self.executor = None
        self.local_thread_executor = None
        self.near_search = None
        self.parallel_stage = None
        self.parallel_futures = []
        self.parallel_retired = []
        self.regeneration_pending = None
        self.parallel_payload = {}
        self.parallel_generation = 0
        self.worker_process_ids = set()
        self.manual_count_future = None
        self.manual_count_pending = None
        self.manual_count_running_key = None
        self.matching_warmup_futures = []
        self.matching_warmup_failed = False

    def configure(self, worker_count):
        for future in self.matching_warmup_futures:
            future.cancel()
        self.matching_warmup_futures = []
        self.matching_warmup_failed = False
        old_executor, self.executor = self.executor, None
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=True)
        self.worker_count = int(worker_count)
        if self.local_thread_executor is not None:
            self.local_thread_executor.shutdown(wait=False, cancel_futures=True)
            self.local_thread_executor = None
        if self.worker_count > 1:
            self.executor = ProcessPoolExecutor(
                max_workers=min(4, self.worker_count),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initializer,
            )

    def background_executor(self):
        return self.thread_executor()

    def thread_executor(self):
        if self.local_thread_executor is None:
            self.local_thread_executor = _NumericalThreadPoolExecutor(
                max_workers=min(2, self.worker_count),
            )
        return self.local_thread_executor

    def _large_multilayer_match(self, point_count, layer_count):
        return (
            self.executor is not None
            and point_count >= PROCESS_MATCH_POINTS
            and layer_count >= PROCESS_MATCH_LAYERS
        )

    def warm_matching_pool(self, point_count, layer_count):
        """Warm after the first substantial match has already been published.

        Even a 300k-point match is slower with cold spawned workers. Starting
        them after publication keeps the first result responsive; repeated
        large, many-layer searches can then amortize startup and transfers.
        """
        if (
            not self._large_multilayer_match(point_count, layer_count)
            or self.matching_warmup_futures or self.matching_warmup_failed
        ):
            return
        try:
            for _ in range(min(4, self.worker_count)):
                self.matching_warmup_futures.append(self.executor.submit(worker_ready))
        except Exception:
            self.matching_warmup_failed = True
            for future in self.matching_warmup_futures:
                future.cancel()

    def matching_executor(self, point_count, layer_count):
        if (
            self._large_multilayer_match(point_count, layer_count)
            and self.matching_warmup_futures and not self.matching_warmup_failed
            and all(future.done() for future in self.matching_warmup_futures)
        ):
            if all(
                not future.cancelled() and future.exception() is None
                for future in self.matching_warmup_futures
            ):
                return self.executor, min(4, self.worker_count)
            self.matching_warmup_failed = True
        return self.thread_executor(), 1

    def cancel_parallel(self):
        self.parallel_generation += 1
        for future in self.parallel_futures:
            if not future.cancel() and not future.done():
                self.parallel_retired.append(future)
        self.parallel_futures = []
        self.parallel_retired = [
            future for future in self.parallel_retired if not future.done()
        ]
        self.regeneration_pending = None
        self.parallel_stage = None
        self.parallel_payload = {}

    def retire_matching_pool(self):
        """Disable a failed process pool while keeping healthy threads available."""
        executor, self.executor = self.executor, None
        self.matching_warmup_failed = True
        for future in self.matching_warmup_futures:
            future.cancel()
        self.matching_warmup_futures = []
        if executor is None:
            return False
        executor.shutdown(wait=False, cancel_futures=True)
        return True

    def close(self):
        for future in self.matching_warmup_futures:
            future.cancel()
        self.matching_warmup_futures = []
        self.manual_count_pending = None
        if self.manual_count_future is not None:
            self.manual_count_future.cancel()
        if self.near_search is not None:
            self.near_search.close()
        self.cancel_parallel()
        executor, self.executor = self.executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if self.local_thread_executor is not None:
            self.local_thread_executor.shutdown(wait=False, cancel_futures=True)
            self.local_thread_executor = None
