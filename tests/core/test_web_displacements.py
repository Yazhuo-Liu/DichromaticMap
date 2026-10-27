"""The browser bridge computes and persists physically moved atoms."""

import base64
import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from .test_session import SETTINGS, VIEW
from .test_web_state_contract import bridge


def render_request():
    return {"lattice":"SC", "axis":"100", "angle":0,
            "width":4, "height":4, "center":[0, 0],
            "deformations":[np.eye(2).tolist(), np.eye(2).tolist()],
            "translations":[[0, 0], [0, 0]]}


def incoming_operation():
    return {"delta":[-20, 0], "bounds":[19.5, 20.5, -.5, .5]}


def test_browser_render_includes_atoms_moved_in_from_offscreen(bridge):
    request = render_request()
    before = bridge._render(request)
    far = bridge._render({**request, "center":[20, 0]})
    source = next(row for row in far["grains"][0] if row[:2] == [20, 0])
    request["grain_displacements"] = [[incoming_operation()], []]
    after = bridge._render(request)
    moved = next(row for row in after["grains"][0] if row[3:] == source[3:])
    assert moved[:2] == [0, 0]
    assert after["grains"][1] == before["grains"][1]
    assert sum(row[:2] == [0, 0] for row in after["grains"][0]) == 2
    assert after["exact_cell"] is None


def test_browser_counts_use_moved_positions_for_geometric_membership(bridge):
    polygon = [[-.25, -.25], [.25, -.25], [.25, .25], [-.25, .25]]
    request = {**render_request(), "polygons":[polygon, polygon], "layer":0,
               "grain_displacements":[[incoming_operation()], []]}
    counted = bridge._count(request)
    assert counted["interior"] == [[2], [1]]
    assert counted["half_open"][0] == [2]


@pytest.mark.parametrize("action", ["near_search", "complete", "fit_selected"])
def test_uniform_strain_analyses_reject_manual_moves_and_retained_custom_base(bridge, action):
    for parameters in ({"grain_displacements":[[incoming_operation()], []]}, {"custom_base":True}):
        with pytest.raises(ValueError, match="Reset grain moves"):
            bridge._ACTIONS[action]({**render_request(), **parameters})


def test_tiny_custom_strain_never_advertises_an_undeformed_exact_cell(bridge):
    request = render_request()
    request["deformations"][0][0][0] += 1e-6
    assert bridge._render(request)["exact_cell"] is None


def test_browser_session_roundtrip_retains_both_grain_histories(bridge):
    operations = [[incoming_operation()], [{"delta":[.125, -.25], "bounds":None}]]
    raw = {"lattice":"SC", "axis":"100", "angle":0, "a0":3.52,
           "colors":["#1677d2", "#e35d35"], "symbols":["o"], "sizes":[1],
           "visible_layers":[[0], [0]], "boundary":[], "atoms":[], "manual":[],
           "deformations":[np.eye(2).tolist(), np.eye(2).tolist()],
           "translations":[[0, 0], [0, 0]], "grain_displacements":operations}
    encoded = bridge._save_session(json.loads(json.dumps({"state":raw, "settings":SETTINGS, "view_range":VIEW})))
    with ZipFile(BytesIO(base64.b64decode(encoded))) as archive:
        payload = json.loads(archive.read("session.json"))
        assert payload["schema_version"] == 5
        assert payload["state"]["grain_displacements"] == operations
    loaded = bridge._load_session({"bytes":encoded})
    assert loaded["state"]["grain_displacements"] == operations
    request = render_request()
    expected = bridge._render({**request, "grain_displacements":operations})
    actual = bridge._render({**request, "grain_displacements":loaded["state"]["grain_displacements"]})
    assert actual["grains"] == expected["grains"]
