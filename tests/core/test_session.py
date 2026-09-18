"""Real session archives, physical roundtrips and invalid-data isolation."""

import json
import struct
import zipfile

import numpy as np
import pytest

from dichromatic_map import crystal, session
from dichromatic_map.state import CellVertex, PatternParameters, PatternState, SelectedAtom
from dichromatic_map.strain import strain_selected_cell
from .helpers import fcc22_diamond_pairs

SETTINGS = dict(region_states=(True, False, False, True), manual_visible=True,
                show_common_cell=False, local_distance=0.05, strain_percent=2.0,
                search_index=12, manual_strain_percent=2.0, manual_rotation_deg=1.0)
VIEW = (-7.125, 9.875, -8.25, 5.75)


def populated_state():
    state = PatternState(PatternParameters(lattice="SC", axis="111", angle_deg=17.123456789,
                                           width=14, height=8, lattice_constant=4.1,
                                           marker_size=25, view_scale=1.5))
    state.display_rotation_deg = 33.7
    state.show_reference_axes = False
    state.selected_layer = 1
    state.visible_grain_layers = [{0, 2}, {1, 2}]
    state.visible_layers = {2}
    state.selected_points = [np.array([-1.2, 0.5]), np.array([3.7, 5.2])]
    state.deformations = (np.array([[1.01, 0.02], [0., 0.99]]), np.array([[1., 0.], [-0.01, 1.03]]))
    state.translations = np.array([[0.12, 0.25], [-0.4, 0.3]])
    state.axial_repeat = -3
    for grain, sign in enumerate((1, -1)):
        points = crystal.projected_columns(8, 8, sign * state.angle_deg / 2,
                                           lattice="SC", axis="111",
                                           deformation=state.deformations[grain],
                                           translation=state.translations[grain])
        index = np.flatnonzero(points.layers == 1)[3]
        state.selected_atoms.append(SelectedAtom(points.positions[index], grain, 1,
                                                 points.half_indices[index]))
    state.manual_vertices = [CellVertex(np.array([0.5, 0.75]), 1, "local",
                                       np.array([[0.49, 0.76], [0.51, 0.74]]))]
    state.manual_local_cutoff = 0.05
    state.near_enabled = True
    state.near_method = "local"
    state.interaction_mode = "cell"
    return state


def manual_strain_state():
    state = PatternState(PatternParameters(angle_deg=22))
    _, polygons = fcc22_diamond_pairs()
    # Shift the original cell to exercise nonzero post-deformation translations.
    from dichromatic_map.cells import bases
    for grain, basis in enumerate(bases(22)):
        polygons[grain] += basis @ [3, 1]
    original = [CellVertex(pair.mean(axis=0), 1, "local", pair.copy())
                for pair in polygons.transpose(1, 0, 2)]
    fit = strain_selected_cell(polygons, 22, layer=1)
    state.manual_unstrained_vertices = original
    state.manual_unstrained_cutoff = state.manual_local_cutoff = 0.05
    state.manual_strain_fit = fit
    state.near_enabled = True
    state.near_method = "local"
    state.near_cell = fit.cell
    state.deformations = (fit.cell.f1, fit.cell.f2)
    state.translations = fit.translations.copy()
    state.manual_vertices = [CellVertex(pair.mean(axis=0), 1, "CSL", pair.copy())
                             for pair in fit.vertices.transpose(1, 0, 2)]
    return state


def rewrite_metadata(path, mutate):
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    payload = json.loads(members["session.json"])
    mutate(payload)
    members["session.json"] = json.dumps(payload).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_roundtrip_full_state_and_partial_picking_without_cached_results(tmp_path):
    path = tmp_path / "with spaces.dmap"
    before = populated_state()
    session.save_session(path, before, SETTINGS, VIEW)
    snapshot = session.load_session(path)
    after = snapshot.state
    assert after is not before
    assert after.parameters == before.parameters
    assert after.angle_deg == 17.123456789
    assert after.display_rotation_deg == 33.7
    assert not after.show_reference_axes
    assert after.selected_layer == 1
    assert after.visible_grain_layers == [{0, 2}, {1, 2}]
    assert after.visible_layers == {2}
    assert after.axial_repeat == -3
    assert after.near_enabled and after.near_method == "local"
    assert after.manual_local_cutoff == 0.05
    assert after.interaction_mode == "cell"
    assert after.grains == [] and after.manual_counts is None and after.buffer_bounds is None
    assert snapshot.settings == SETTINGS
    assert snapshot.view_range == VIEW
    for name in ("selected_points", "deformations", "translations"):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    for actual, expected in zip(after.selected_atoms, before.selected_atoms):
        assert actual.grain_index == expected.grain_index and actual.layer == expected.layer
        np.testing.assert_array_equal(actual.position, expected.position)
        np.testing.assert_array_equal(actual.half_indices, expected.half_indices)
    assert len(after.manual_vertices) == 1
    np.testing.assert_array_equal(after.manual_vertices[0].grain_positions,
                                  before.manual_vertices[0].grain_positions)
    after.translations[0, 0] = 99
    assert before.translations[0, 0] == 0.12
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"session.json", "counts.csv", "vectors.csv", "strain.csv", "README.txt"}
        data = archive.read("session.json").decode()
        assert "grain_signature" not in data and "worker" not in data and "buffer_bounds" not in data
        assert archive.read("vectors.csv").decode().strip()


def test_roundtrip_applied_manual_strain_keeps_restore_original_data(tmp_path):
    state = manual_strain_state()
    path = tmp_path / "strained.dmap"
    session.save_session(path, state, SETTINGS, VIEW)
    after = session.load_session(path).state
    assert after.manual_strain_fit is not None
    assert len(after.near_solutions) == 1
    assert after.manual_unstrained_cutoff == 0.05
    for name in ("translations", "vertices", "origin", "rotations_deg", "stretches"):
        np.testing.assert_array_equal(getattr(after.manual_strain_fit, name),
                                      getattr(state.manual_strain_fit, name))
    for actual, expected in zip(after.manual_unstrained_vertices, state.manual_unstrained_vertices):
        np.testing.assert_array_equal(actual.grain_positions, expected.grain_positions)
        assert actual.source == "local" and actual.layer == 1
    assert all(vertex.source == "CSL" for vertex in after.manual_vertices)
    # Re-applying the preserved original pairs produces the same strain and shifts.
    polygons = np.stack([vertex.grain_positions for vertex in after.manual_unstrained_vertices], axis=1)
    repeated = strain_selected_cell(polygons, after.angle_deg, layer=1)
    np.testing.assert_allclose(repeated.translations, after.translations, atol=1e-12)
    np.testing.assert_allclose([repeated.cell.f1, repeated.cell.f2], after.deformations, atol=1e-12)


def test_save_uses_current_geometry_and_exact_angle_not_launch_parameters(tmp_path):
    state = PatternState(PatternParameters())
    state.geometry = crystal.get_geometry("BCC", "100")
    state.angle_deg = 2 * np.degrees(np.arctan(1 / 3))
    state.visible_grain_layers = [set(range(state.geometry.layer_count)) for _ in range(2)]
    path = tmp_path / "changed.dmap"
    session.save_session(path, state, SETTINGS, VIEW)
    after = session.load_session(path).state
    assert (after.parameters.lattice, after.parameters.axis) == ("BCC", "100")
    assert after.angle_deg == state.angle_deg
    assert state.parameters.lattice == "FCC"  # Saving did not mutate launch parameters.


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(format="another-format"),
    lambda p: p.update(schema_version=999),
    lambda p: p.update(schema_version=True),
    lambda p: p.update(view_range=[1, -1, 0, 2]),
    lambda p: p["state"]["parameters"].update(angle_deg=float("nan")),
    lambda p: p["state"]["parameters"].update(width=float("inf")),
    lambda p: p["state"]["parameters"].update(angle_deg=61),
    lambda p: p["state"]["parameters"].update(axis="0 0 0"),
    lambda p: p["state"].update(deformations=[[[1, 0], [0, -1]], [[1, 0], [0, 1]]]),
    lambda p: p["state"].update(translations=[[0, 0, 0], [0, 0, 0]]),
    lambda p: p["state"].update(translations=[[True, 0], [0, 0]]),
    lambda p: p["state"].update(selected_layer=99),
    lambda p: p["state"].update(visible_grain_layers=[[0, 0], [1]]),
    lambda p: p["state"].update(axial_repeat=5),
    lambda p: p["state"].update(show_reference_axes=1),
    lambda p: p["state"].update(near_method="unknown"),
    lambda p: p["state"].update(interaction_mode="unknown"),
    lambda p: p["state"]["manual_vertices"][0].update(grain_positions=None),
    lambda p: p["state"]["selected_atoms"][0].update(grain_index=3),
    lambda p: p["state"]["selected_atoms"][0].update(position=[0, 0]),
    lambda p: p["state"]["selected_atoms"][0].update(half_indices=[1, 0, 0]),
    lambda p: p["state"]["manual_vertices"][0].update(source="unknown"),
    lambda p: p["settings"].update(search_index=0),
    lambda p: p["settings"].update(region_states=[True, False]),
    lambda p: p["settings"].update(local_distance=-1),
    lambda p: p["settings"].update(unrecognized=True),
])
def test_invalid_metadata_is_rejected(tmp_path, mutate):
    path = tmp_path / "invalid.dmap"
    session.save_session(path, populated_state(), SETTINGS, VIEW)
    rewrite_metadata(path, mutate)
    with pytest.raises(ValueError):
        session.load_session(path)


@pytest.mark.parametrize("mutate", [
    lambda p: p["state"].update(manual_unstrained_vertices=None),
    lambda p: p["state"]["manual_strain_fit"].update(translations=[[0, 0], [0, 0]]),
    lambda p: p["state"]["manual_strain_fit"].update(stretches=[[1, 1], [1, 1]]),
    lambda p: p["state"]["near_cell"].update(cell=[[1, 0], [0, 1]]),
    lambda p: p["state"]["near_cell"].update(max_strain=0),
])
def test_inconsistent_applied_strain_is_rejected(tmp_path, mutate):
    path = tmp_path / "invalid-strain.dmap"
    session.save_session(path, manual_strain_state(), SETTINGS, VIEW)
    rewrite_metadata(path, mutate)
    with pytest.raises(ValueError):
        session.load_session(path)


def test_invalid_quad_is_rejected(tmp_path):
    path = tmp_path / "crossed.dmap"
    session.save_session(path, manual_strain_state(), SETTINGS, VIEW)
    def cross(payload):
        vertices = payload["state"]["manual_vertices"]
        vertices[1], vertices[2] = vertices[2], vertices[1]
    rewrite_metadata(path, cross)
    with pytest.raises(ValueError, match="convex"):
        session.load_session(path)


@pytest.mark.parametrize("content", [b"not a zip archive", b"PK\x03\x04truncated"])
def test_invalid_zip_is_rejected(tmp_path, content):
    path = tmp_path / "bad.dmap"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        session.load_session(path)


def test_oversized_or_unexpected_archive_members_are_rejected(tmp_path):
    path = tmp_path / "large.dmap"
    session.save_session(path, PatternState(PatternParameters()), SETTINGS, VIEW)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../../outside.txt", b"never extracted")
    with pytest.raises(ValueError, match="members"):
        session.load_session(path)
    session.save_session(path, PatternState(PatternParameters()), SETTINGS, VIEW)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["counts.csv"] = b"0" * (session.MAX_MEMBER_BYTES + 1)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="too large"):
        session.load_session(path)


def test_duplicate_json_fields_are_rejected(tmp_path):
    path = tmp_path / "duplicate.dmap"
    session.save_session(path, PatternState(PatternParameters()), SETTINGS, VIEW)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["session.json"] = b'{"schema_version":1,"schema_version":2}'
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="Duplicate"):
        session.load_session(path)


@pytest.mark.parametrize("failure", ["validation", "tables", "replace"])
def test_failed_save_preserves_existing_file_and_removes_temporary(tmp_path, monkeypatch, failure):
    path = tmp_path / "existing.dmap"
    path.write_bytes(b"existing user data")
    state = PatternState(PatternParameters())
    if failure == "validation":
        state.angle_deg = float("nan")
    elif failure == "tables":
        from dichromatic_map import exports
        def fail(*_args):
            raise ValueError("Unable to produce tables")
        monkeypatch.setattr(exports, "build_export_tables", fail)
    else:
        def fail(*_args):
            raise OSError("Cannot replace destination")
        monkeypatch.setattr(session.os, "replace", fail)
    with pytest.raises((ValueError, OSError)):
        session.save_session(path, state, SETTINGS, VIEW)
    assert path.read_bytes() == b"existing user data"
    assert list(tmp_path.iterdir()) == [path]


def test_corrupted_deflate_stream_is_rejected_as_invalid_session(tmp_path):
    path = tmp_path / "corrupted-stream.dmap"
    session.save_session(path, PatternState(PatternParameters()), SETTINGS, VIEW)
    with zipfile.ZipFile(path) as archive:
        metadata = archive.getinfo("session.json")
        assert metadata.compress_type == zipfile.ZIP_DEFLATED
        offset = metadata.header_offset
    damaged = bytearray(path.read_bytes())
    # The local header contains filename/extra lengths at bytes 26 and 28.
    name_length, extra_length = struct.unpack_from("<HH", damaged, offset + 26)
    start = offset + 30 + name_length + extra_length
    damaged[start] = 0x07  # A final deflate block with reserved block type 3.
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="Invalid DichromaticMap session"):
        session.load_session(path)
