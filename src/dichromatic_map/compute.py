"""Qt-free worker entry points, executor ownership and asynchronous search.

Workers are module-level functions so a spawned process imports numerical
modules only. The UI coordinates operations and polls completed futures.
"""

from __future__ import annotations
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import os
import numpy as np
from .crystal import ProjectedGrain, projected_columns, get_geometry
from .cells import count_cell_atoms
from .matching import local_near_pairs, same_layer_coincidence_sites
from .strain import candidate_vectors, solve_cells_chunk, pareto_cells


def worker_initializer():
    """Avoid each process starting its own BLAS thread team when available."""
    global _blas_limit
    try:
        from threadpoolctl import threadpool_limits

        _blas_limit = threadpool_limits(limits=1)
    except ImportError:
        pass  # NumPy-only installations remain supported.


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

    only_first = grain_1.layers == layer
    only_second = grain_2.layers == layer
    first = ProjectedGrain(
        grain_1.positions[only_first],
        np.zeros(np.count_nonzero(only_first), dtype=np.int16),
        grain_1.half_indices[only_first],
    )
    second = ProjectedGrain(
        grain_2.positions[only_second],
        np.zeros(np.count_nonzero(only_second), dtype=np.int16),
        grain_2.half_indices[only_second],
    )
    return os.getpid(), same_layer_coincidence_sites(first, second, tolerance)[0]


def coincidence_layers_worker(grain_1, grain_2, tolerance, layers):
    """A bounded batch of phases, avoiding one full-array transfer per layer."""
    return os.getpid(), tuple(
        (
            int(layer),
            coincidence_layer_worker(grain_1, grain_2, tolerance, int(layer))[1],
        )
        for layer in layers
    )


class NearSearch:
    """Bounded asynchronous search; stale requests never publish results.

    Preparation and pair solves run off the UI thread. At most workers jobs
    are in flight. Invalidation drops pending jobs and lets bounded running
    chunks finish before submitting the next request.
    """

    def __init__(self, workers=1, executor=None):
        self.workers = max(1, int(workers))
        self.executor = executor
        self.owns_executor = executor is None
        self.generation = 0
        self.jobs = deque()
        self.running = {}
        self.parts = []
        self.total = self.completed = 0
        self.busy = False
        self.error = None
        self.process_ids = set()

    def cancel(self):
        self.generation += 1
        self.jobs.clear()
        self.parts.clear()
        self.busy = False
        self.error = None
        self.total = self.completed = 0
        for future in self.running:
            future.cancel()

    def request(self, angle, percent, extent, lattice="FCC", axis="110"):
        self.cancel()
        geometry = get_geometry(lattice, axis)
        if self.executor is None:
            self.executor = (
                ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=worker_initializer,
                )
                if self.workers > 1
                else ThreadPoolExecutor(max_workers=1)
            )
        self.args = (angle, percent, extent, geometry.lattice, geometry.axis)
        self.jobs.append(("prepare", candidate_vectors, self.args))
        self.busy = True

    def poll(self):
        try:
            for future in list(self.running):
                if not future.done():
                    continue
                generation, stage = self.running.pop(future)
                if generation != self.generation:
                    continue
                result = future.result()
                if stage == "prepare":
                    i, j = result
                    for start in range(0, len(i), 8):
                        self.jobs.append(
                            (
                                "solve",
                                solve_cells_chunk,
                                (
                                    self.args[0],
                                    self.args[1],
                                    i,
                                    j,
                                    start,
                                    start + 8,
                                    self.args[3],
                                    self.args[4],
                                ),
                            )
                        )
                    self.total = len(self.jobs)
                else:
                    pid, cells = result
                    self.process_ids.add(pid)
                    self.parts.extend(cells)
                    self.completed += 1
            while self.jobs and len(self.running) < self.workers:
                stage, function, args = self.jobs.popleft()
                future = self.executor.submit(function, *args)
                self.running[future] = (self.generation, stage)
            if self.busy and not self.jobs and not self.running:
                result = pareto_cells(self.parts)
                self.parts.clear()
                self.busy = False
                return result
        except Exception as error:
            self.cancel()
            self.error = str(error)
        return None

    def close(self):
        self.cancel()
        if self.owns_executor and self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = None


class ComputeSession:
    """Own the shared executors and pending work for one viewer session."""

    def __init__(self):
        self.worker_count = 1
        self.executor = None
        self.local_thread_executor = None
        self.near_search = None
        self.parallel_stage = None
        self.parallel_futures = []
        self.parallel_payload = {}
        self.parallel_generation = 0
        self.worker_process_ids = set()
        self.manual_count_future = None
        self.manual_count_pending = None
        self.manual_count_running_key = None

    def configure(self, worker_count):
        old_executor, self.executor = self.executor, None
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=True)
        self.worker_count = int(worker_count)
        if self.worker_count > 1:
            self.executor = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initializer,
            )

    def background_executor(self):
        if self.executor is not None:
            return self.executor
        if self.local_thread_executor is None:
            self.local_thread_executor = ThreadPoolExecutor(max_workers=1)
        return self.local_thread_executor

    def cancel_parallel(self):
        self.parallel_generation += 1
        for future in self.parallel_futures:
            future.cancel()
        self.parallel_futures = []
        self.parallel_stage = None
        self.parallel_payload = {}

    def close(self):
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
