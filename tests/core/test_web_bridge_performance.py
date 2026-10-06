"""Check the browser's buffered/cooperative adapter against its public JSON API."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def bridge():
    path = Path(__file__).resolve().parents[2] / "site/web_bridge.py"
    spec = importlib.util.spec_from_file_location("performance_web_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("overrides", [
    {},
    {"angle": 0},
    {"angle": 22, "local_matching": True, "local_distance": .1},
    {"axis": "1 -1 3", "width": 6, "height": 4},
    {"deformations": [[[1.002, .001], [.001, .998]], [[.999, 0], [0, 1.001]]],
     "translations": [[.125, -.0625], [-.08, .02]], "local_matching": True},
    {"center": [37.125, -32.875], "width": 6, "height": 4},
    {"center": [.123456, .234567], "width": .0001, "height": .0001,
     "local_matching": True, "local_distance": 1e-6},
])
def test_render_buffers_match_json_coordinates_indices_and_metadata(bridge, overrides):
    request = {"action": "render", "lattice": "FCC", "axis": "110",
               "angle": 38.94244126898139, "width": 12, "height": 9, **overrides}
    serialized = json.dumps(request)
    legacy = json.loads(bridge.web_dispatch(serialized))
    buffered = bridge.web_render(serialized)
    for grain, expected in zip(buffered["grains"], legacy["grains"]):
        assert grain.dtype == np.float64 and grain.flags.c_contiguous
        np.testing.assert_array_equal(grain.reshape(-1, 6), np.asarray(expected).reshape(-1, 6))
    np.testing.assert_array_equal(buffered["coincidences"].reshape(-1, 3),
                                  np.asarray(legacy["coincidences"]).reshape(-1, 3))
    if legacy["local"] is None:
        assert buffered["local"] is None
    else:
        np.testing.assert_array_equal(buffered["local"].reshape(-1, 5),
                                      np.asarray(legacy["local"]).reshape(-1, 5))
    assert json.loads(buffered["metadata"]) == {
        key: value for key, value in legacy.items() if key not in {"grains", "coincidences", "local"}}


@pytest.mark.parametrize("overrides", [
    {}, {"index": 12}, {"angle": 39.5, "percent": 2, "index": 40},
    {"percent": 3, "index": 10}, {"axis": "111"},
])
def test_fixed_search_batches_preserve_historical_matrices_and_tensor_readout(bridge, monkeypatch, overrides):
    request = {"action": "near_search", "lattice": "FCC", "axis": "110",
               "angle": 22, "percent": 2, "index": 8, **overrides}
    serialized = json.dumps(request)
    args = (request["angle"], request["percent"], request["index"],
            request["lattice"], request["axis"])
    first, second = bridge.candidate_vectors(*args)
    # This is the original synchronous loop. Keep its fixed boundaries so
    # equivalent-cell representatives at floating-point ties cannot depend
    # on a device's speed, message timing or cancellation checks.
    historical = []
    for start in range(0, len(first), 8):
        _, cells = bridge.solve_cells_chunk(args[0], args[1], first, second,
                                           start, start + 8, args[3], args[4])
        historical.extend(cells)
    expected = [{**bridge._cell(cell), "readout": bridge.tensor_readout(cell, args[0])}
                for cell in bridge.pareto_cells(historical)]
    spans = []
    original = bridge.solve_cells_chunk

    def record(*args):
        spans.append((args[4], args[5]))
        return original(*args)

    monkeypatch.setattr(bridge, "solve_cells_chunk", record)
    job = bridge.web_near_start(serialized)
    while not bridge.web_near_step(job):
        pass
    assert spans == [(start, start + 8) for start in range(0, len(first), 8)]
    actual = json.loads(bridge.web_near_finish(job))
    # Compare the complete selected representative, including integer m1/m2,
    # deformations, cell orientation, strain, label and tensor readout text.
    assert actual == expected


def test_search_step_returns_between_bounded_candidate_rows(bridge, monkeypatch):
    serialized = json.dumps({"angle": 22, "percent": 2, "index": 12,
                             "lattice": "FCC", "axis": "110"})
    job = bridge.web_near_start(serialized)
    calls = []
    original = bridge.solve_cells_chunk

    def record(*args):
        calls.append((args[4], args[5]))
        return original(*args)

    monkeypatch.setattr(bridge, "solve_cells_chunk", record)
    assert not bridge.web_near_step(job)
    assert calls == [(0, 8)]
    assert job["next"] == 8
    assert not bridge.web_near_step(job)
    assert calls == [(0, 8), (8, 16)]


def test_zero_candidate_search_finishes_without_solving(bridge, monkeypatch):
    monkeypatch.setattr(bridge, "candidate_vectors", lambda *args: (np.empty((0, 2), int), np.empty((0, 2), int)))
    monkeypatch.setattr(bridge, "solve_cells_chunk", lambda *args: pytest.fail("empty search must not solve"))
    job = bridge.web_near_start(json.dumps({"angle": 22, "percent": 2, "index": 8,
                                          "lattice": "FCC", "axis": "110"}))
    assert bridge.web_near_step(job)
    assert json.loads(bridge.web_near_finish(job)) == []
