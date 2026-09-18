"""Physical numerical exports, independently checked against simple lattices."""

import csv
import io

import numpy as np
import pytest

from dichromatic_map.exports import build_export_tables
from dichromatic_map.state import CellVertex, PatternParameters, PatternState, SelectedAtom


def rotation(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def state(lattice="SC", axis="100", angle=0):
    return PatternState(PatternParameters(
        lattice=lattice, axis=axis, angle_deg=angle, lattice_constant=2.5,
    ))


def rows(tables, name):
    return list(csv.DictReader(io.StringIO(tables[name])))


def cell_vertices(polygons, layer=0):
    return [
        CellVertex(points.mean(axis=0), layer, "local", points.copy())
        for points in np.asarray(polygons).transpose(1, 0, 2)
    ]


def strain_value(rows, quantity, component="", grain="G1", frame=None, unit="1"):
    matches = [
        float(row["value"]) for row in rows
        if row["grain"] == grain and row["quantity"] == quantity
        and row["component"] == component and row["unit"] == unit
        and (frame is None or row["frame"] == frame)
    ]
    assert len(matches) == 1
    return matches[0]


def matrix(rows, quantity, axes="xy", grain="G1", frame=None):
    return np.array([
        [strain_value(rows, quantity, a + b, grain, frame) for b in axes]
        for a in axes
    ])


def test_sc_square_counts_recomputed_and_gb_filtered_without_mutation():
    current = state()
    polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    current.manual_vertices = cell_vertices(np.stack((polygon, polygon)))
    # Caches and view filtering must not substitute for physical enumeration.
    sentinel = object()
    current.manual_counts = sentinel
    current.visible_grain_layers = [set(), set()]
    current.selected_points = [np.array([1., -1.]), np.array([1., 3.])]
    settings = {"manual_visible": True, "region_states": (True, False, False, True)}
    result = rows(build_export_tables(current, settings), "counts.csv")
    assert [row["grain"] for row in result] == ["G1", "G2"]
    assert [int(row["closed"]) for row in result] == [6, 6]
    assert [int(row["interior"]) for row in result] == [1, 1]
    assert [int(row["boundary"]) for row in result] == [5, 5]
    assert [int(row["half_open"]) for row in result] == [4, 2]
    assert [int(row["half_open_edges"]) for row in result] == [2, 1]
    assert [int(row["half_open_corners"]) for row in result] == [1, 0]
    assert all(row["gb_side_filter_applied"] == "1" for row in result)
    assert [(row["gb_keep_left"], row["gb_keep_right"]) for row in result] == [("1", "0"), ("0", "1")]
    assert [float(result[0][key]) for key in ("gb_start_x_a0", "gb_start_y_a0", "gb_end_x_a0", "gb_end_y_a0")] == [1, -1, 1, 3]
    assert all(float(row["area_a0_squared"]) == 4 for row in result)
    assert all(float(row["area_angstrom_squared"]) == 25 for row in result)
    settings["manual_visible"] = False
    result = rows(build_export_tables(current, settings), "counts.csv")
    assert [int(row["closed"]) for row in result] == [9, 9]
    assert all(row["gb_keep_left"] == "" and row["gb_start_x_a0"] == "" for row in result)
    assert [int(row["half_open"]) for row in result] == [4, 4]
    assert current.manual_counts is sentinel
    np.testing.assert_array_equal(current.manual_vertices[2].grain_positions, [[2, 2], [2, 2]])


def test_counts_use_independent_translated_deformed_polygons_and_only_picked_layer():
    current = state(lattice="BCC")
    current.deformations = (np.diag([1.1, 0.9]), np.diag([0.95, 1.05]))
    current.translations = np.array([[3.25, -2.5], [-1.5, 7.25]])
    # BCC (100) layer 1 is offset by half a grid spacing in x and y.
    polygons = np.array([
        [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]],
        [[0.5, 0.5], [3.5, 0.5], [3.5, 2.5], [0.5, 2.5]],
    ])
    polygons = np.einsum("gij,gnj->gni", current.deformations, polygons) + current.translations[:, None]
    current.manual_vertices = cell_vertices(polygons, layer=1)
    result = rows(build_export_tables(current, {}), "counts.csv")
    assert len(result) == 2
    assert [int(row["layer"]) for row in result] == [1, 1]
    assert [int(row["half_open"]) for row in result] == [4, 6]
    np.testing.assert_allclose([float(row["area_a0_squared"]) for row in result], [4 * .99, 6 * .9975])


def test_nonparallelogram_half_open_counts_are_blank_not_zero():
    current = state()
    polygons = np.array([
        [[0, 0], [2, 0], [2, 2], [0, 2]],
        [[0, 0], [3, 0], [2, 2], [0, 2]],
    ])
    current.manual_vertices = cell_vertices(polygons)
    result = rows(build_export_tables(current, {}), "counts.csv")
    assert result[0]["half_open_available"] == "1"
    assert result[1]["half_open_available"] == "0"
    assert [result[1][key] for key in ("half_open", "half_open_edges", "half_open_corners")] == ["", "", ""]
    assert int(result[1]["closed"]) > 0


def test_incomplete_selections_export_headers_and_truthful_unstrained_tensors():
    current = state()
    current.selected_atoms = [SelectedAtom(np.zeros(2), 0, 0, np.zeros(3, dtype=int))]
    current.manual_vertices = [CellVertex(np.zeros(2), 0, "csl", np.zeros((2, 2)))]
    tables = build_export_tables(current, {})
    assert set(tables) == {"counts.csv", "vectors.csv", "strain.csv", "README.txt"}
    assert rows(tables, "counts.csv") == []
    assert rows(tables, "vectors.csv") == []
    assert len(tables["counts.csv"].splitlines()) == 1
    assert len(tables["vectors.csv"].splitlines()) == 1
    values = rows(tables, "strain.csv")
    for grain in ("G1", "G2"):
        for quantity in ("F", "R", "U"):
            np.testing.assert_array_equal(matrix(values, quantity, grain=grain), np.eye(2))
        np.testing.assert_array_equal(matrix(values, "E", grain=grain, frame="reference_analysis_xy"), np.zeros((2, 2)))
        np.testing.assert_array_equal(matrix(values, "E", "uvw", grain, "reference_cubic_uvw"), np.zeros((3, 3)))
    assert "header-only" in tables["README.txt"]


def test_same_grain_vector_preserves_axial_image_and_distinguishes_lattice_coordinates():
    current = state()
    stretch = np.array([[1.04, .015], [.015, .98]])
    current.deformations = (rotation(4) @ stretch, np.eye(2))
    current.translations = np.array([[.3, -.2], [0, 0]])
    first = current.translations[0]
    second = current.deformations[0] @ [2, 3] + first
    current.selected_atoms = [
        SelectedAtom(first, 0, 0, np.zeros(3, dtype=int)),
        SelectedAtom(second, 0, 0, np.array([2, 4, 6])),
    ]
    current.axial_repeat = 2
    current.display_rotation_deg = 90
    result = rows(build_export_tables(current, {}), "vectors.csv")
    assert len(result) == 4
    by_frame = {row["frame"]: row for row in result}
    components = lambda frame: np.array([float(by_frame[frame][f"component_{i}"]) for i in (1, 2, 3)])
    displacement = np.r_[second - first, 3]
    np.testing.assert_allclose(components("analysis_xyz"), displacement)
    np.testing.assert_allclose(components("display_xyz"), [-displacement[1], displacement[0], 3])
    np.testing.assert_allclose(components("deformed_cubic_lattice_uvw"), [3, 2, 3], atol=1e-12)
    np.testing.assert_allclose(components("current_polar_cubic_uvw"), np.r_[3, stretch @ [2, 3]], atol=1e-12)
    assert {row["grain"] for row in result} == {"", "G1"}
    assert all(int(row["axial_repeat"]) == 2 for row in result)
    assert all(float(row["length_a0"]) == np.linalg.norm(displacement) for row in result)
    assert all(float(row["length_angstrom"]) == 2.5 * np.linalg.norm(displacement) for row in result)


def test_cross_grain_vector_uses_both_actual_positions_and_phase_height():
    current = state(lattice="BCC", angle=30)
    stretches = (np.diag([1.03, .97]), np.array([[.98, .01], [.01, 1.02]]))
    polar_angles = (3, -2)
    current.deformations = tuple(rotation(a) @ u for a, u in zip(polar_angles, stretches))
    current.translations = np.array([[.2, -.4], [-.3, .7]])
    positions = [
        current.translations[0],
        current.deformations[1] @ rotation(-15) @ [.5, .5] + current.translations[1],
    ]
    current.selected_atoms = [
        SelectedAtom(positions[0], 0, 0, np.zeros(3, dtype=int)),
        SelectedAtom(positions[1], 1, 1, np.ones(3, dtype=int)),
    ]
    current.axial_repeat = -1
    result = rows(build_export_tables(current, {}), "vectors.csv")
    assert len(result) == 6
    displacement = np.r_[positions[1] - positions[0], -.5]
    cubic_to_analysis = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    for grain, angle in enumerate((15, -15)):
        ref = np.eye(3)
        ref[:2, :2] = rotation(angle)
        ref = ref @ cubic_to_analysis
        f = np.eye(3)
        f[:2, :2] = current.deformations[grain]
        r = np.eye(3)
        r[:2, :2] = rotation(polar_angles[grain])
        for frame, expected in (
            ("current_polar_cubic_uvw", (r @ ref).T @ displacement),
            ("deformed_cubic_lattice_uvw", np.linalg.solve(f @ ref, displacement)),
        ):
            row = next(row for row in result if row["grain"] == f"G{grain + 1}" and row["frame"] == frame)
            np.testing.assert_allclose([float(row[f"component_{i}"]) for i in (1, 2, 3)], expected, atol=1e-12)
            assert row["p1_layer"] == "0" and row["p2_layer"] == "1"


def test_polar_and_cubic_strain_export_use_current_f_not_cached_fit():
    current = state(angle=24)
    stretch = np.array([[1.04, .01], [.01, .98]])
    polar = rotation(3)
    current.deformations = (polar @ stretch, rotation(-2))
    current.translations = np.array([[.125, -.25], [.375, .5]])
    current.near_cell = object()  # Deliberately cannot be used as a tensor source.
    current.manual_strain_fit = object()
    result = rows(build_export_tables(current, {}), "strain.csv")
    np.testing.assert_allclose(matrix(result, "R"), polar, atol=1e-12)
    np.testing.assert_allclose(matrix(result, "U"), stretch, atol=1e-12)
    expected_e = (stretch @ stretch - np.eye(2)) / 2
    np.testing.assert_allclose(matrix(result, "E", frame="reference_analysis_xy"), expected_e, atol=1e-12)
    # For [100], planar coordinates are cubic y,z, with grain reference rotation +12°.
    expected_cubic = np.zeros((3, 3))
    expected_cubic[1:, 1:] = rotation(12).T @ expected_e @ rotation(12)
    np.testing.assert_allclose(matrix(result, "E", "uvw", frame="reference_cubic_uvw"), expected_cubic, atol=1e-12)
    principal = [strain_value(result, "principal_engineering_strain", str(i)) for i in (1, 2)]
    np.testing.assert_allclose(principal, np.linalg.eigvalsh(stretch)[::-1] - 1, atol=1e-12)
    assert strain_value(result, "polar_rotation", unit="degree") == pytest.approx(3)
    assert strain_value(result, "reference_misorientation", grain="both", unit="degree") == 24
    assert strain_value(result, "polar_misorientation", grain="both", unit="degree") == pytest.approx(29)
    assert strain_value(result, "translation", "x", unit="a0") == .125
    assert strain_value(result, "translation", "x", unit="angstrom") == .3125
    np.testing.assert_allclose(matrix(result, "E", grain="G2", frame="reference_analysis_xy"), 0, atol=1e-12)


@pytest.mark.parametrize("invalid", [np.diag([-1., 1.]), np.zeros((2, 2)), np.array([[1., np.nan], [0., 1.]])])
def test_invalid_deformation_never_exports_partial_or_nonfinite_results(invalid):
    current = state()
    current.deformations = (invalid, np.eye(2))
    with pytest.raises(ValueError):
        build_export_tables(current, {})


def test_invalid_complete_cell_does_not_become_an_empty_table():
    current = state()
    current.manual_vertices = [CellVertex(np.zeros(2), 0, "csl") for _ in range(4)]
    with pytest.raises(ValueError, match="Cell atom endpoints"):
        build_export_tables(current, {})
