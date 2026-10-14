"""Large-coordinate fitted cells retain their physical state and exported counts."""

import base64
import csv
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from dichromatic_map import crystal, exports, session
from dichromatic_map.state import CellVertex, PatternParameters, PatternState
from dichromatic_map.strain import strain_selected_cell
from .test_session import SETTINGS


@pytest.mark.parametrize("shift", [0., 60_000_000.])
def test_bounded_translated_fit_roundtrips_through_archive_bridge_and_csv(tmp_path, shift):
    polygon = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    original = np.stack([polygon * side + shift for side in (49, 51)])
    fit = strain_selected_cell(original, 0, 2, "SC", "100", max_rotation_deg=0)
    state = PatternState(PatternParameters(lattice="SC", axis="100", angle_deg=0))
    state.manual_unstrained_vertices = [CellVertex(pair.mean(axis=0), 0, "closure", pair.copy())
        for pair in original.transpose(1, 0, 2)]
    state.manual_vertices = [CellVertex(pair.mean(axis=0), 0, "CSL", pair.copy())
        for pair in fit.vertices.transpose(1, 0, 2)]
    state.manual_strain_fit = fit
    state.near_enabled = True
    state.near_method = "local"
    state.near_cell = fit.cell
    state.deformations = (fit.cell.f1, fit.cell.f2)
    state.translations = fit.translations.copy()
    view = (shift - 5, shift + 56, shift - 5, shift + 56)
    path = tmp_path / "large-coordinate.dmap"
    session.save_session(path, state, SETTINGS, view)

    spec = importlib.util.spec_from_file_location("large_coordinate_bridge",
        Path(__file__).resolve().parents[2] / "site/web_bridge.py")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    loaded = json.loads(bridge.web_dispatch(json.dumps({"action":"load_session",
        "bytes":base64.b64encode(path.read_bytes()).decode()})))
    after = session._snapshot_from_payload({"format":session.FORMAT,
        "schema_version":session.SCHEMA_VERSION, **loaded}).state
    np.testing.assert_array_equal(after.deformations, state.deformations)
    np.testing.assert_array_equal(after.translations, state.translations)
    np.testing.assert_array_equal(after.manual_strain_fit.vertices, fit.vertices)
    assert after.manual_strain_fit.residual <= 1e-8

    # Check actual regenerated atoms, independently of the fit's stored points.
    for index, deformation in enumerate(after.deformations):
        center = fit.vertices[index].mean(axis=0)
        generated = crystal.projected_columns(54, 54, 0, center=tuple(center),
            lattice="SC", axis="100", deformation=deformation,
            translation=after.translations[index])
        for point in fit.vertices[index]:
            assert np.min(np.hypot(*(generated.positions - point).T)) <= 1e-8

    with zipfile.ZipFile(path) as archive:
        count_text = archive.read("counts.csv").decode()
    assert count_text == exports.build_export_tables(after, SETTINGS)["counts.csv"]
    counts = list(csv.DictReader(io.StringIO(count_text)))
    assert tuple(int(row["half_open"]) for row in counts) == (49 ** 2, 51 ** 2)
    assert tuple(int(row["closed"]) for row in counts) == (50 ** 2, 52 ** 2)
