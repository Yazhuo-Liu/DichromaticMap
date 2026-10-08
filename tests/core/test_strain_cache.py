"""Exact search keys, bounded resumable preparation and isolated cached arrays."""

from dataclasses import replace
import numpy as np
import pytest

from dichromatic_map import matching, strain


@pytest.fixture(autouse=True)
def empty_strain_caches():
    strain.clear_strain_caches()
    yield
    strain.clear_strain_caches()


def assert_same_cells(first, second):
    assert len(first) == len(second)
    for reference, result in zip(first, second):
        assert result.atoms == reference.atoms
        assert result.max_strain == reference.max_strain
        for field in ("m1", "m2", "f1", "f2", "cell"):
            np.testing.assert_array_equal(getattr(reference, field), getattr(result, field))


def test_candidate_cache_normalizes_crystal_and_returns_independent_arrays(monkeypatch):
    first, second = strain.candidate_vectors(39.5, 2, 12)
    expected = first.copy(), second.copy()
    first[:] = 999
    second[:] = -999

    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("Normalized equivalent crystal should reuse the candidates")

    monkeypatch.setattr(strain, "bases", unexpected_rebuild)
    repeated = strain.candidate_vectors(39.5, 2, 12, " fcc ", "2 2 0")
    for actual, reference in zip(repeated, expected):
        np.testing.assert_array_equal(actual, reference)
    repeated[0][:] = 123
    np.testing.assert_array_equal(strain.candidate_vectors(39.5, 2, 12)[0], expected[0])


def test_candidate_cache_never_rounds_angle_or_strain(monkeypatch):
    original_bases = strain.bases
    calls = []

    def observed_bases(*args):
        calls.append(args)
        return original_bases(*args)

    monkeypatch.setattr(strain, "bases", observed_bases)
    strain.candidate_vectors(39.5, 2, 12)
    strain.candidate_vectors(np.nextafter(39.5, np.inf), 2, 12)
    strain.candidate_vectors(39.5, np.nextafter(2.0, np.inf), 12)
    assert len(calls) == 3
    strain.candidate_vectors(np.float64(39.5), np.float64(2), 12, "fcc", "2 2 0")
    assert len(calls) == 3


def test_candidate_cache_is_bounded_and_evicts_least_recent_entry(monkeypatch):
    monkeypatch.setattr(strain, "_CANDIDATE_CACHE_LIMIT", 2)
    for angle in (22, 39.5, 22, 37.2):
        strain.candidate_vectors(angle, 2, 12)
    assert len(strain._candidate_cache) == 2
    assert strain._candidate_parameters(22, 2, 12, "FCC", "110") in strain._candidate_cache
    assert strain._candidate_parameters(39.5, 2, 12, "FCC", "110") not in strain._candidate_cache


def test_integer_grid_is_reused_and_cannot_be_made_writable():
    grid = strain._integer_translation_grid(12)
    assert strain._integer_translation_grid(12) is grid
    with pytest.raises(ValueError):
        grid.setflags(write=True)
    with pytest.raises(ValueError):
        grid[0] = 0
    np.testing.assert_array_equal(grid[:12], np.column_stack((np.zeros(12, int), np.arange(1, 13))))
    assert strain._integer_translation_grid.cache_info().maxsize == 39


def test_preparation_is_bounded_and_only_complete_results_enter_cache():
    preparation = strain.CandidatePreparation(22, 10, 12)
    assert not preparation.done
    assert preparation.completed_offsets == 0
    assert preparation.step(max_offsets=1) is None
    assert preparation.completed_offsets == 1
    assert not strain._candidate_cache
    abandoned = strain.CandidatePreparation(22, 10, 12)
    assert not abandoned.done
    while not preparation.done:
        before = preparation.completed_offsets
        result = preparation.step(max_offsets=7)
        assert preparation.completed_offsets - before <= 7
    assert result is not None
    cached = strain.CandidatePreparation(22, 10, 12)
    assert cached.done
    expected = tuple(array.copy() for array in result)
    result[0][:] = 111
    for actual, reference in zip(cached.step(), expected):
        np.testing.assert_array_equal(actual, reference)
    with pytest.raises(ValueError):
        preparation.step(max_offsets=0)


@pytest.mark.parametrize("block_size", [1, 7, 500])
def test_retained_prefix_has_bounded_memory_and_exact_stable_ties(monkeypatch, block_size):
    monkeypatch.setattr(strain, "MAX_VECTORS", 10)
    preparation = strain.CandidatePreparation(22, 10, 12)
    # Deliberate ties across batches, plus distinct adjacent floating values.
    size = np.tile([4.0, 1.0, 1.0, np.nextafter(1.0, 0.0), 2.0], 100)
    error = np.tile([0.1, 0.3, 0.1, 0.1, np.nextafter(0.1, 0.0)], 100)
    first = np.column_stack((np.arange(len(size)), np.zeros(len(size), int)))
    second = -first
    for start in range(0, len(size), block_size):
        stop = start + block_size
        preparation._retain(first[start:stop], second[start:stop],
                            error[start:stop], size[start:stop])
        assert len(preparation._n1) <= strain.MAX_VECTORS
        visited = min(stop, len(size))
        expected = np.unique(np.r_[
            np.argsort(size[:visited], kind="stable")[:5],
            np.argsort(error[:visited], kind="stable")[:5],
        ])
        np.testing.assert_array_equal(preparation._n1, first[expected])
        np.testing.assert_array_equal(preparation._n2, second[expected])
        np.testing.assert_array_equal(preparation._sizes, size[expected])
        np.testing.assert_array_equal(preparation._errors, error[expected])


def test_high_index_preparation_keeps_only_bounded_candidates_between_steps():
    preparation = strain.CandidatePreparation(22, 10, 12, "FCC", "1 1 15")
    while not preparation.done:
        before = preparation.completed_offsets
        preparation.step(32)
        assert preparation.completed_offsets - before <= 32
        assert len(preparation._n1) <= strain.MAX_VECTORS
        assert len(preparation._n2) <= strain.MAX_VECTORS
        assert len(preparation._errors) <= strain.MAX_VECTORS
        assert len(preparation._sizes) <= strain.MAX_VECTORS


@pytest.mark.parametrize("lattice,axis,angle", [
    ("FCC", "110", 22), ("FCC", "112", 0),
    ("BCC", "100", 37.2), ("SC", "1 -1 3", 21.4),
])
def test_incremental_candidates_match_ordered_exhaustive_physical_oracle(lattice, axis, angle):
    # Enumerate ALL integer translation pairs, independently of the production
    # radius/coordinate prefilters. Keep the documented dx,dy,integer order.
    from itertools import product

    extent, percent = 5, 3
    first = np.array([(x, y) for x, y in product(range(extent + 1),
        range(-extent, extent + 1)) if x > 0 or y > 0])
    second = np.array([(x, y) for x, y in product(
        range(-extent, extent + 1), repeat=2) if x or y])
    b1, b2 = strain.bases(angle, lattice, axis)
    v, w = first @ b1.T, second @ b2.T
    distance = np.linalg.norm(v[:, None] - w[None], axis=2)
    size = np.linalg.norm(v, axis=1)[:, None] + np.linalg.norm(w, axis=1)[None]
    rows, columns = np.nonzero(distance / size <= percent / 100 + 1e-12)
    nearest = np.rint(v @ np.linalg.inv(b2).T).astype(int)
    ordered = sorted(zip(rows, columns), key=lambda pair: (
        *(second[pair[1]] - nearest[pair[0]]), pair[0]))
    assert len(ordered) <= strain.MAX_VECTORS
    expected = (first[[row for row, _ in ordered]], second[[column for _, column in ordered]])
    for block_size in (1, 7, 32):
        strain.clear_strain_caches()
        preparation = strain.CandidatePreparation(angle, percent, extent, lattice, axis)
        while not preparation.done:
            result = preparation.step(block_size)
        for actual, reference in zip(result, expected):
            np.testing.assert_array_equal(actual, reference)


def test_search_cache_owns_its_arrays_and_returns_fresh_cells():
    args = 39.5, 2, 12
    results = strain.find_strained_cells(*args)
    expected = strain.get_cached_cell_search(*args)
    assert results and expected
    for field in ("m1", "m2", "f1", "f2", "cell"):
        getattr(results[0], field)[:] = -500
    results.clear()
    repeated = strain.get_cached_cell_search(*args, "fcc", "2 2 0")
    assert_same_cells(expected, repeated)
    strain.cache_cell_search(repeated, *args)
    repeated[0].f1[:] = 90
    repeated.append(repeated[0])
    assert_same_cells(expected, strain.get_cached_cell_search(*args))


def test_search_cache_uses_exact_parameters_and_chunk_policy():
    args = 39.5, 2, 12
    strain.cache_cell_search([], *args)
    assert strain.get_cached_cell_search(*args) == []
    assert strain.get_cached_cell_search(np.nextafter(39.5, np.inf), 2, 12) is None
    assert strain.get_cached_cell_search(39.5, np.nextafter(2.0, np.inf), 12) is None
    assert strain.get_cached_cell_search(39.5, 2, 13) is None
    assert strain.get_cached_cell_search(*args, rows_per_chunk=1) is None
    assert strain.get_cached_cell_search(*args, "BCC") is None


def test_cached_empty_search_does_not_repeat_numerical_work(monkeypatch):
    args = 13.25, 0.001, 3
    assert strain.find_strained_cells(*args) == []

    def unexpected_computation(*args, **kwargs):
        raise AssertionError("A complete empty result should be cached")

    monkeypatch.setattr(strain, "candidate_vectors", unexpected_computation)
    monkeypatch.setattr(strain, "solve_cells_chunk", unexpected_computation)
    assert strain.find_strained_cells(*args) == []


def test_failed_partial_search_does_not_publish_a_result(monkeypatch):
    calls = []

    def failed_chunk(*args):
        calls.append(args)
        if len(calls) == 1:
            return 0, []
        raise RuntimeError("Failed numerical chunk")

    monkeypatch.setattr(strain, "solve_cells_chunk", failed_chunk)
    with pytest.raises(RuntimeError, match="Failed numerical chunk"):
        strain.find_strained_cells(39.5, 2, 12)
    assert strain.get_cached_cell_search(39.5, 2, 12) is None


def test_search_cache_bounded_lru_and_invalid_results(monkeypatch):
    monkeypatch.setattr(strain, "_CELL_SEARCH_CACHE_LIMIT", 2)
    strain.cache_cell_search([], 22, 2, 12)
    strain.cache_cell_search([], 39.5, 2, 12)
    assert strain.get_cached_cell_search(22, 2, 12) == []
    strain.cache_cell_search([], 37.2, 2, 12)
    assert len(strain._cell_search_cache) == 2
    assert strain.get_cached_cell_search(39.5, 2, 12) is None
    cell = matching.exact_csl_cell(0)
    with pytest.raises(ValueError, match="does not match"):
        strain.cache_cell_search([cell], 0, 2, 12, "SC", "100")
    with pytest.raises(ValueError, match="at most 12"):
        strain.cache_cell_search([cell] * 13, 0, 2, 12)
    with pytest.raises(ValueError, match="2-by-2"):
        strain.cache_cell_search([replace(cell, cell=np.zeros((3, 3)))], 0, 2, 12)
    with pytest.raises(ValueError, match="positive integer"):
        strain.get_cached_cell_search(0, 2, 12, rows_per_chunk=1.5)


def test_shared_sync_search_retains_serial_order_and_complete_pareto_result():
    args = 39.5, 2, 12
    first, second = strain.candidate_vectors(*args)
    parts = []
    for start in range(0, len(first), 8):
        parts.extend(strain.solve_cells_chunk(args[0], args[1], first, second, start, start + 8)[1])
    expected = strain.pareto_cells(parts)
    assert_same_cells(expected, strain.find_strained_cells(*args))
    assert_same_cells(expected, strain.find_strained_cells(*args))
