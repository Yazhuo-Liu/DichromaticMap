"""Scientific state crosses the browser bridge without implicit normalization."""
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dichromatic_map import crystal, session
from dichromatic_map.state import PatternParameters, PatternState, SelectedAtom
from .test_session import SETTINGS, VIEW


@pytest.fixture(scope="module")
def bridge():
    spec = importlib.util.spec_from_file_location("state_contract_bridge",
        Path(__file__).resolve().parents[2] / "site/web_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_preserves_imported_near_preset_angle_and_selected_atoms(bridge, tmp_path):
    angle = next(p.angle_deg for p in crystal.csl_presets("110") if p.name == "Σ9") + .001
    state = PatternState(PatternParameters(angle_deg=angle))
    state.angle_deg = angle  # Sessions deliberately retain their exact saved angle.
    grain = crystal.projected_columns(12, 9, angle / 2)
    for index in (len(grain.positions) // 3, len(grain.positions) // 2):
        state.selected_atoms.append(SelectedAtom(grain.positions[index], 0,
            int(grain.layers[index]), grain.half_indices[index]))
    path = tmp_path / "precise.dmap"
    session.save_session(path, state, SETTINGS, VIEW)
    loaded = json.loads(json.dumps(bridge._load_session({
        "bytes": base64.b64encode(path.read_bytes()).decode()})))
    rendered = json.loads(bridge.web_dispatch(json.dumps({"action":"render", "angle":angle,
        "width":12, "height":9, "lattice":"FCC", "axis":"110"})))
    assert rendered["angle"] == angle
    assert rendered["preset"] is None
    loaded["state"]["parameters"]["angle_deg"] = rendered["angle"]
    restored = session._snapshot_from_payload({"format":session.FORMAT,
        "schema_version":session.SCHEMA_VERSION, **loaded}).state
    for before, after in zip(state.selected_atoms, restored.selected_atoms):
        np.testing.assert_array_equal(before.position, after.position)
        np.testing.assert_array_equal(before.half_indices, after.half_indices)
