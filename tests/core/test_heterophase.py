"""Two different grains: per-grain lattice, tilt axis and lattice constant."""

import csv
import io
import json
import zipfile

import numpy as np
import pytest

from dichromatic_map import crystal, session
from dichromatic_map.cells import bases, count_cell_atoms
from dichromatic_map.completion import cell_completion_candidates
from dichromatic_map.exports import build_export_tables
from dichromatic_map.matching import (
    exact_csl_cell, local_near_pairs, projected_coincidence_sites, projected_near_pairs,
)
from dichromatic_map.state import CellVertex, PatternParameters, PatternState
from dichromatic_map.strain import find_strained_cells, strain_selected_cell, tensor_readout

FE = dict(lattice=("FCC", "BCC"), axis=("110", "111"), scale=2.87 / 3.59)
# Kurdjumov-Sachs about FCC [110] || BCC [111], exactly as the viewer snaps it.
KS = crystal.angle_presets(FE["lattice"], FE["axis"])[0].angle_deg
SETTINGS = dict(region_states=(True, True, True, True), manual_visible=False,
                show_common_cell=True, local_distance=0.1, strain_percent=2.0,
                search_index=12, manual_strain_percent=2.0, manual_rotation_deg=1.0)
VIEW = (-6.0, 6.0, -4.5, 4.5)


def rotation3(angle):
    theta = np.deg2rad(angle)
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def test_bicrystal_normalizes_shared_and_per_grain_fields():
    shared = crystal.bicrystal("fcc", (1, 1, 0))
    assert shared.homophase and shared.lattice == "FCC" and shared.axis == "110"
    pair = crystal.bicrystal(("FCC", "bcc"), ("[110]", "1 1 1"), 0.8)
    assert not pair.homophase
    assert pair.lattices == ("FCC", "BCC") and pair.axes == ("110", "111")
    assert pair.layer_counts == (2, 3) and pair.layer_count == 3
    assert crystal.bicrystal(pair) is pair
    # Same lattice and axis with another lattice constant is still two phases.
    assert not crystal.bicrystal("FCC", "110", 0.9).homophase
    for bad in (dict(scale=0), dict(scale=np.inf), dict(scale=True),
                dict(lattice=("FCC", "HCP")), dict(axis=("110", "111", "100"))):
        with pytest.raises(ValueError):
            crystal.bicrystal(**{"lattice": "FCC", "axis": "110", **bad})


@pytest.mark.parametrize("axes,period,maximum", [
    (("110", "111"), 60.0, 30.0),
    (("110", "001"), 90.0, 45.0),
    (("112", "110"), 180.0, 90.0),
])
def test_two_grain_angle_range_uses_both_axial_symmetries(axes, period, maximum):
    bounds = crystal.misorientation_range(axes, ("FCC", "BCC"))
    assert (bounds.period_deg, bounds.maximum_deg) == (period, maximum)
    assert bounds.sign_equivalent


def test_without_grain_exchange_or_shared_twofold_the_whole_period_is_kept():
    homophase = crystal.misorientation_range("1 2 3", "FCC")
    assert homophase.maximum_deg == 180.0  # Grain exchange identifies ±theta.
    different = crystal.misorientation_range("1 2 3", "FCC", scale=1.1)
    assert different.maximum_deg == 360.0 and not different.sign_equivalent


@pytest.mark.parametrize("axes,expected", [
    (("110", "111"), {"KS"}),
    (("110", "001"), {"Bain", "NW"}),
    (("111", "110"), {"KS", "NW"}),
    (("110", "110"), set()),
])
def test_orientation_presets_reproduce_parallel_planes_and_directions(axes, expected):
    pair = crystal.bicrystal(("FCC", "BCC"), axes)
    presets = crystal.angle_presets(pair)
    assert {preset.name for preset in presets} == expected
    relationships = {name: rest for name, *rest in crystal.ORIENTATION_RELATIONSHIPS}
    first, second = pair.geometries
    for preset in presets:
        # Cubic components of grain 1 / grain 2 vectors in analysis axes.
        q1 = rotation3(preset.angle_deg / 2) @ first.frame.T
        q2 = rotation3(-preset.angle_deg / 2) @ second.frame.T
        fcc_plane, bcc_plane, fcc_line, bcc_line = relationships[preset.name]
        # Some symmetry variant makes both pairs parallel.
        found = any(
            np.allclose(q1 @ (r1 @ fcc_plane), np.linalg.norm(fcc_plane) / np.linalg.norm(bcc_plane)
                        * (q2 @ (r2 @ bcc_plane)), atol=1e-9)
            and np.allclose(q1 @ (r1 @ fcc_line), np.linalg.norm(fcc_line) / np.linalg.norm(bcc_line)
                            * (q2 @ (r2 @ bcc_line)), atol=1e-9)
            for r1 in crystal.PROPER_CUBIC_ROTATIONS for r2 in crystal.PROPER_CUBIC_ROTATIONS
        )
        assert found, preset
        assert 0 <= preset.angle_deg <= crystal.misorientation_range(axes, ("FCC", "BCC")).maximum_deg


def test_homophase_presets_remain_csl():
    assert crystal.angle_presets("FCC", "110") == crystal.csl_presets("110")
    assert crystal.angle_presets(("FCC", "SC"), "110") == ()


def test_lattice_constant_ratio_scales_unstrained_positions_only():
    reference = crystal.projected_columns(6, 6, 12.0, lattice="BCC", axis="111")
    scaled = crystal.projected_columns(6 * 0.8, 6 * 0.8, 12.0, lattice="BCC", axis="111", scale=0.8)
    order = np.lexsort(reference.half_indices.T)
    scaled_order = np.lexsort(scaled.half_indices.T)
    np.testing.assert_array_equal(reference.half_indices[order], scaled.half_indices[scaled_order])
    np.testing.assert_allclose(scaled.positions[scaled_order], 0.8 * reference.positions[order],
                               atol=1e-12)


def test_rational_lattice_ratio_gives_an_exact_common_cell():
    cell = exact_csl_cell(0.0, lattice="FCC", axis="001", scale=0.8)
    assert cell is not None and cell.crystal == crystal.bicrystal("FCC", "001", 0.8)
    b1, b2 = bases(0.0, "FCC", "001", 0.8)
    np.testing.assert_allclose(b1 @ cell.m1, b2 @ cell.m2, atol=1e-12)
    assert cell.atoms == (32, 50)  # Four G1 repeats equal five G2 repeats.
    # A rounded experimental ratio is not commensurate within the bound.
    assert exact_csl_cell(24.7, **FE) is None
    # The homophase recognition path is unchanged.
    assert exact_csl_cell(38.94244126898139).atoms == (18, 18)


def test_projected_matching_compares_columns_across_all_layers():
    pair = crystal.bicrystal(**FE)
    angle = KS
    grains = [crystal.projected_columns(8, 8, sign * angle / 2, lattice=lattice, axis=axis,
                                        scale=scale)
              for sign, lattice, axis, scale in zip((1, -1), pair.lattices, pair.axes, pair.scales)]
    pairs = projected_near_pairs(*grains, 0.2)
    assert len(pairs.first) and set(np.unique(pairs.second_layers)) == {0, 1, 2}
    # Brute-force mutual nearest neighbours over every column of both grains.
    distances = np.linalg.norm(grains[0].positions[:, None] - grains[1].positions[None], axis=2)
    forward, reverse = distances.argmin(axis=1), distances.argmin(axis=0)
    expected = {
        (i, j) for i, j in enumerate(forward)
        if reverse[j] == i and 1e-6 < distances[i, j] <= 0.2
    }
    found = {
        (int(np.flatnonzero(np.all(grains[0].positions == first, axis=1))[0]),
         int(np.flatnonzero(np.all(grains[1].positions == second, axis=1))[0]))
        for first, second in zip(pairs.first, pairs.second)
    }
    assert found == expected
    np.testing.assert_array_equal(
        pairs.layers, [grains[0].layers[np.all(grains[0].positions == p, axis=1)][0] for p in pairs.first])
    np.testing.assert_array_equal(
        pairs.second_layers,
        [grains[1].layers[np.all(grains[1].positions == p, axis=1)][0] for p in pairs.second])
    sites, partners = projected_coincidence_sites(*grains, 1e-6)
    assert len(sites) == pair.layer_count == len(partners)
    np.testing.assert_allclose(np.concatenate(sites), [[0, 0]], atol=1e-12)
    assert np.concatenate(partners).tolist() == [0]


def test_same_layer_pairs_keep_both_layer_fields_equal():
    first = crystal.projected_columns(10, 10, 11, lattice="FCC", axis="110")
    second = crystal.projected_columns(10, 10, -11, lattice="FCC", axis="110")
    pairs = local_near_pairs(first, second, 0.05)
    np.testing.assert_array_equal(pairs.layers, pairs.second_layers)


@pytest.fixture(scope="module")
def ks_cell():
    cells = find_strained_cells(KS, 2.0, 12, FE["lattice"], FE["axis"], FE["scale"])
    assert cells
    return cells[0]


def test_two_grain_strain_search_certifies_common_translations(ks_cell):
    pair = ks_cell.crystal
    assert pair == crystal.bicrystal(**FE)
    b1, b2 = bases(KS, pair)
    np.testing.assert_allclose(ks_cell.f1 @ b1 @ ks_cell.m1, ks_cell.cell, atol=1e-8)
    np.testing.assert_allclose(ks_cell.f2 @ b2 @ ks_cell.m2, ks_cell.cell, atol=1e-8)
    assert ks_cell.max_strain <= 0.02 + 1e-12
    assert ks_cell.atoms == (2 * abs(round(np.linalg.det(ks_cell.m1))),
                             3 * abs(round(np.linalg.det(ks_cell.m2))))
    text = tensor_readout(ks_cell, KS)
    assert "G1 FCC [1 1 0]; G2 BCC [1 1 1]" in text and "no common C3" in text


def cell_polygons(cell, angle, origin_layers=(0, 0)):
    """Original atom vertices of a search cell, starting at each grain's layer origin."""
    pair = cell.crystal
    polygons = []
    for grain, (basis, matrix, layer) in enumerate(zip(bases(angle, pair), (cell.m1, cell.m2),
                                                      origin_layers)):
        geometry = pair.geometries[grain]
        offset = (pair.scales[grain] * crystal.rotation_matrix_2d((1 - 2 * grain) * angle / 2)
                  @ (geometry.layer_offsets_half_indices[layer] @ geometry.frame[:, :2] / 2))
        a, b = (basis @ matrix).T
        polygons.append(offset + np.array([[0, 0], a, a + b, b]))
    return np.array(polygons)


def test_selected_cell_fit_uses_one_layer_per_grain(ks_cell):
    angle = KS
    polygons = cell_polygons(ks_cell, angle, (1, 2))
    fit = strain_selected_cell(polygons, angle, 2.0, FE["lattice"], FE["axis"], layer=(1, 2),
                               max_rotation_deg=0, scale=FE["scale"])
    np.testing.assert_allclose(fit.cell.f1, ks_cell.f1, atol=1e-9)
    np.testing.assert_allclose(fit.cell.f2, ks_cell.f2, atol=1e-9)
    assert fit.residual <= 1e-8
    with pytest.raises(ValueError, match="same selected layer"):
        strain_selected_cell(polygons, angle, 2.0, FE["lattice"], FE["axis"], layer=(1, 1),
                             max_rotation_deg=0, scale=FE["scale"])
    with pytest.raises(ValueError, match="valid axial layer"):
        strain_selected_cell(polygons, angle, 2.0, FE["lattice"], FE["axis"], layer=2,
                             max_rotation_deg=0, scale=FE["scale"])


def test_counts_use_each_grains_own_layer(ks_cell):
    angle = KS
    polygons = cell_polygons(ks_cell, angle, (1, 2))
    counts = count_cell_atoms(polygons, angle, (np.eye(2), np.eye(2)), FE["lattice"], FE["axis"],
                              layer=(1, 2), scale=FE["scale"])
    assert counts.interior.shape == (2, 3)
    assert counts.half_open[0].tolist() == [0, abs(round(np.linalg.det(ks_cell.m1))), 0]
    assert counts.half_open[1].tolist() == [0, 0, abs(round(np.linalg.det(ks_cell.m2)))]


def test_three_vertex_closure_completes_a_two_grain_cell(ks_cell):
    angle = KS
    polygons = cell_polygons(ks_cell, angle)
    candidates = cell_completion_candidates(polygons[:, :3], angle, lattice=FE["lattice"],
                                            axis=FE["axis"], layer=(0, 0), scale=FE["scale"],
                                            max_rotation_deg=0)
    assert len(candidates) == 1 and candidates[0].fit is not None
    np.testing.assert_allclose(candidates[0].vertices, polygons, atol=1e-9)
    assert all("exchange" not in candidate.description for candidate in cell_completion_candidates(
        polygons[:, :2], angle, lattice=FE["lattice"], axis=FE["axis"], scale=FE["scale"]))


def test_vector_coordinates_use_each_grains_lattice_constant():
    pair = crystal.bicrystal(**FE)
    geometry = pair.geometries[1]
    # Two BCC atoms one primitive step apart along the axis: a0(G2)/2 [1 1 1].
    dz = crystal.axial_displacement(pair, 1, np.zeros(3, dtype=int), 1,
                                    np.zeros(3, dtype=int), 1)
    assert dz == pytest.approx(pair.scale * geometry.axial_period)
    result = crystal.crystal_vector_coordinates(np.r_[0, 0, dz], -10.0, None, "BCC", "111",
                                                scale=pair.scale)
    assert crystal.format_direction_components(result.current) == "a₀/2[1 1 1]"
    # Cross-grain heights come from each grain's own layer height.
    first = np.array([1, 1, 0])  # FCC layer at a0(G1)/2 [1 1 0] . e_z
    assert crystal.axial_displacement(pair, 0, first, 1, np.zeros(3, dtype=int)) == pytest.approx(
        -0.5 * first @ pair.geometries[0].frame[:, 2])


def heterophase_state(ks_cell):
    state = PatternState(PatternParameters(angle_deg=KS, lattice="FCC", axis="110",
                                           lattice2="BCC", axis2="111",
                                           lattice_constant=3.59, lattice_ratio=2.87 / 3.59))
    assert state.crystal == ks_cell.crystal
    state.near_enabled = True
    state.near_method = "strain"
    state.near_cell = ks_cell
    state.near_solutions = [ks_cell]
    state.deformations = (ks_cell.f1, ks_cell.f2)
    polygons = cell_polygons(ks_cell, KS, (1, 2))
    # Actual endpoints of the applied (translation-free) deformations.
    endpoints = np.stack([polygon @ f.T for polygon, f in zip(polygons, state.deformations)], axis=1)
    state.manual_vertices = [CellVertex(pair.mean(axis=0), 1, "local", pair.copy(), 2)
                             for pair in endpoints]
    state.visible_grain_layers = [{0, 1}, {1, 2}]
    state.visible_layers = {1}
    return state


def test_heterophase_state_parameters_and_ranges():
    state = PatternState(PatternParameters(lattice="FCC", axis="110", lattice2="BCC",
                                           axis2="111", lattice_ratio=2.87 / 3.59))
    assert state.crystal.layer_counts == (2, 3) and state.layer_count == 3
    assert state.angle_range.maximum_deg == 30.0
    assert [preset.name for preset in state.presets] == ["KS"]
    assert state.angle_deg == pytest.approx(state.presets[0].angle_deg)
    assert state.visible_grain_layers == [{0, 1}, {0, 1, 2}]
    assert len(state.layer_symbols) == 3 and len(state.coincident_points) == 3
    with pytest.raises(ValueError, match="between 0 and 30"):
        PatternParameters(angle_deg=40, lattice2="BCC", axis2="111").validate()
    with pytest.raises(ValueError, match="between 0.001 and 1000"):
        PatternParameters(lattice_ratio=0).validate()
    # Grain 2's controls alone leave the original single-structure state.
    assert PatternParameters(lattice2="FCC", axis2="110", lattice_ratio=1.0).crystal.homophase


def test_heterophase_session_roundtrip_and_exports(tmp_path, ks_cell):
    state = heterophase_state(ks_cell)
    path = tmp_path / "fcc-bcc.dmap"
    session.save_session(path, state, SETTINGS, VIEW)
    after = session.load_session(path).state
    assert after.crystal == state.crystal
    assert after.parameters.lattice2 == "BCC"
    assert after.parameters.lattice_ratio == 2.87 / 3.59
    assert [vertex.layers for vertex in after.manual_vertices] == [(1, 2)] * 4
    assert after.near_cell.crystal == state.crystal
    np.testing.assert_array_equal(after.near_cell.m2, ks_cell.m2)
    with zipfile.ZipFile(path) as archive:
        counts = list(csv.DictReader(io.StringIO(archive.read("counts.csv").decode())))
        strain_rows = list(csv.DictReader(io.StringIO(archive.read("strain.csv").decode())))
    assert [(row["grain"], row["layer"]) for row in counts] == [("G1", "1"), ("G2", "2")]
    assert [row["half_open"] for row in counts] == [
        str(abs(round(np.linalg.det(ks_cell.m1)))), str(abs(round(np.linalg.det(ks_cell.m2))))]
    constants = {row["grain"]: float(row["value"]) for row in strain_rows
                 if row["quantity"] == "lattice_constant"}
    assert constants == pytest.approx({"G1": 3.59, "G2": 2.87})
    tables = build_export_tables(after, SETTINGS)
    assert "a0_G2" in tables["README.txt"]


def test_homophase_vertices_reject_a_different_grain_two_layer(tmp_path):
    state = PatternState(PatternParameters(angle_deg=22))
    state.manual_vertices = [CellVertex(np.zeros(2), 0, "CSL", np.zeros((2, 2)))]
    path = tmp_path / "same.dmap"
    session.save_session(path, state, SETTINGS, VIEW)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    payload = json.loads(members["session.json"])
    payload["state"]["manual_vertices"][0]["second_layer"] = 1
    members["session.json"] = json.dumps(payload).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="one layer in both grains"):
        session.load_session(path)
