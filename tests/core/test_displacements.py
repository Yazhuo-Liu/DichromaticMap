"""Physical atom translations preserve identities, crops and count semantics."""

from itertools import product

import numpy as np
import pytest

from dichromatic_map.cells import count_cell_atoms
from dichromatic_map.affine import affine_points
from dichromatic_map.compute import generate_grain_worker
from dichromatic_map.crystal import (
    GeometryLimitError, get_geometry, projected_columns, rotation_matrix_2d,
)
from dichromatic_map.displacements import (
    MAX_DISPLACEMENT_OPERATIONS, apply_displacements,
    displacement_source_rectangles, normalize_displacements,
)


def _identity_map(grain):
    return {(int(layer), *indices): position for layer, indices, position
            in zip(grain.layers, grain.half_indices, grain.positions)}


@pytest.mark.parametrize("lattice,axis", [("SC", "100"), ("FCC", "112"), ("BCC", "110")])
def test_whole_grain_operation_matches_existing_uniform_translation(lattice, axis):
    delta = [.234, -.678]
    options = dict(width=6, height=5, rotation_deg=27, lattice=lattice, axis=axis,
                   deformation=np.array([[1.04, .03], [.02, .98]]))
    expected = projected_columns(translation=delta, **options)
    actual = projected_columns(displacements=[{"delta": delta}], **options)
    np.testing.assert_array_equal(actual.half_indices, expected.half_indices)
    np.testing.assert_array_equal(actual.layers, expected.layers)
    np.testing.assert_allclose(actual.positions, expected.positions, atol=2e-15)


def test_region_moves_selected_identities_and_preserves_distinct_overlapping_atoms():
    # The x=0 column moves on top of x=1. Both physical atoms remain present.
    operation = {"bounds": [-.1, .1, -1.1, 1.1], "delta": [1, 0]}
    grain = projected_columns(4, 4, 0, lattice="SC", axis="100", displacements=[operation])
    assert len(grain.positions) == 25
    assert len(_identity_map(grain)) == 25
    assert np.count_nonzero(np.all(grain.positions == [1, 0], axis=1)) == 2
    for identity, position in _identity_map(grain).items():
        original = np.array(identity[2:]) / 2
        expected = original + [1, 0] if original[0] == 0 and abs(original[1]) <= 1 else original
        np.testing.assert_array_equal(position, expected)


def test_long_distance_region_generates_incoming_atoms_without_empty_space_enumeration():
    operation = {"bounds": [999999.9, 1000000.1, -.1, .1], "delta": [-1000000, 0]}
    grain = projected_columns(2, 2, 0, lattice="SC", axis="100", displacements=[operation])
    assert len(grain.positions) == 10
    assert np.count_nonzero(np.all(grain.positions == [0, 0], axis=1)) == 2
    np.testing.assert_array_equal(_identity_map(grain)[(0, 0, 2000000, 0)], [0, 0])


def test_region_shift_removes_old_positions_even_when_destination_is_outside_crop():
    grain = projected_columns(4, 4, 0, lattice="SC", axis="100", displacements=[
        {"bounds": [-.1, .1, -2.1, 2.1], "delta": [1000000, 0]}
    ])
    assert len(grain.positions) == 20
    assert not np.any(grain.positions[:, 0] == 0)


@pytest.mark.parametrize("lattice", ["FCC", "BCC", "SC"])
def test_sequential_regions_with_strain_rotation_and_layer_filter_match_large_crop_oracle(lattice):
    options = dict(rotation_deg=23.7, lattice=lattice, axis="112",
                   deformation=np.array([[1.03, .015], [.02, .97]]),
                   translation=np.array([.2, -.3]), layers=[0, 1])
    operations = [
        {"bounds": [-1.3, 2.7, -2.1, 1.3], "delta": [3.2, -.2]},
        {"delta": [-.4, .7]},
        {"bounds": [2, 4, -.5, 2.5], "delta": [-2.1, 1.2]},
    ]
    reference = projected_columns(24, 24, **options)
    positions = reference.positions.copy()
    # Independent, point-by-point operation oracle over a generously padded
    # view checks that sequential selection uses current physical positions.
    for index, point in enumerate(positions):
        for operation in operations:
            bounds = operation.get("bounds")
            if bounds is None or (bounds[0] <= point[0] <= bounds[1]
                                  and bounds[2] <= point[1] <= bounds[3]):
                point += operation["delta"]
        positions[index] = point
    expected = {(int(layer), *indices): point for layer, indices, point
                in zip(reference.layers, reference.half_indices, positions)
                if -3 <= point[0] <= 3 and -2.5 <= point[1] <= 2.5}
    actual = projected_columns(6, 5, displacements=operations, **options)
    assert len(actual.positions) == len(expected)
    assert np.all(np.diff(actual.layers) >= 0)
    assert actual.layer_count == reference.layer_count
    assert _identity_map(actual).keys() == expected.keys()
    for identity, point in _identity_map(actual).items():
        np.testing.assert_allclose(point, expected[identity], atol=1e-14)


def test_selection_is_closed_and_later_whole_shift_carries_selected_atoms():
    points = np.array([[-1., 0.], [0., 0.], [1., 0.], [2., 0.]])
    operations = [
        {"bounds": [-1, 1, -1, 1], "delta": [0, .5]},
        {"delta": [.25, -.25]},
    ]
    actual = apply_displacements(points, operations)
    np.testing.assert_array_equal(actual, [[-.75, .25], [.25, .25], [1.25, .25], [2.25, -.25]])
    np.testing.assert_array_equal(points, [[-1, 0], [0, 0], [1, 0], [2, 0]])


def test_closed_edge_membership_agrees_with_independent_saved_atom_reconstruction(tmp_path):
    # Reference-index reconstruction and bulk projection differ by two ULPs
    # here. A selected rectangle edge must not move the atom in only one path.
    deformation = np.array([[1.03, .015], [.02, .97]])
    translation = np.array([.2, -.3])
    options = dict(rotation_deg=23.7, deformation=deformation,
                   translation=translation, axis="112")
    indices = np.array([-4, 0, 2])
    original = projected_columns(6, 5, **options)
    point = original.positions[np.all(original.half_indices == indices, axis=1)][0]
    operation = {"bounds": [point[0] - 1, point[0], point[1] - .1, point[1] + .1],
                 "delta": [.1, .3]}
    reference = rotation_matrix_2d(23.7) @ (indices @ get_geometry("FCC", "112").frame[:, :2] / 2)
    reconstructed = affine_points(reference, deformation, translation)
    checked = apply_displacements(reconstructed.reshape(1, 2), [operation])[0]
    for width, height in [(6, 5), (8, 7), (2, 2)]:
        generated = projected_columns(width, height, center=tuple(point),
                                      displacements=[operation], **options)
        actual = generated.positions[np.all(generated.half_indices == indices, axis=1)][0]
        np.testing.assert_allclose(actual, point + operation["delta"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual, checked, rtol=0, atol=1e-14)

    # Exercise the independent physical-identity validation performed by both
    # archive creation and reload, not just the bulk projection path.
    from dichromatic_map.session import load_session, save_session
    from dichromatic_map.state import PatternParameters, PatternState, SelectedAtom
    state = PatternState(PatternParameters(axis="112", angle_deg=47.4))
    state.deformations = (deformation, np.eye(2))
    state.translations = np.array([translation, [0, 0]])
    state.grain_displacements = [[operation], []]
    atom_index = np.flatnonzero(np.all(generated.half_indices == indices, axis=1))[0]
    state.selected_atoms = [SelectedAtom(actual, 0, int(generated.layers[atom_index]), indices)]
    settings = dict(region_states=(True,) * 4, manual_visible=False,
                    show_common_cell=False, local_distance=.05, strain_percent=2.,
                    search_index=12, manual_strain_percent=2., manual_rotation_deg=1.)
    path = tmp_path / "translated-boundary-atom.dmap"
    save_session(path, state, settings, (-3, 3, -2.5, 2.5))
    restored = load_session(path).state
    np.testing.assert_array_equal(restored.selected_atoms[0].position, actual)
    assert restored.grain_displacements == state.grain_displacements


def test_inverse_regions_include_all_source_atoms_for_ordered_selections():
    points = np.array(list(product(np.arange(-4, 4.01, .25), repeat=2)))
    operations = [
        {"bounds": [-3, 1, -2, 2], "delta": [1.75, -.5]},
        {"bounds": [-2, 3, -.5, 1], "delta": [-2, 1]},
        {"delta": [.25, -.25]},
    ]
    low, high = np.array([-1., -1.]), np.array([1., 1.])
    sources = displacement_source_rectangles(low, high, operations)
    positions = apply_displacements(points, operations)
    enters_crop = np.all((positions >= low) & (positions <= high), axis=1)
    covered = np.zeros(len(points), dtype=bool)
    for lower, upper in sources:
        covered |= np.all((points >= lower) & (points <= upper), axis=1)
    assert np.all(covered[enters_crop])


def test_counts_follow_actual_local_positions_and_leave_other_grain_unchanged():
    polygon = [[-.5, -.5], [1.5, -.5], [1.5, 1.5], [-.5, 1.5]]
    counts = count_cell_atoms(polygon, 0, [np.eye(2), np.eye(2)], lattice="SC", axis="100",
        grain_displacements=[[
            {"bounds": [-.1, .1, -.1, 1.1], "delta": [4, 0]},
            {"bounds": [9.9, 10.1, -.1, .1], "delta": [-10, 0]},
        ], []])
    np.testing.assert_array_equal(counts.interior[:, 0], [3, 4])
    np.testing.assert_array_equal(counts.boundary[:, 0], [0, 0])
    np.testing.assert_array_equal(counts.half_open[:, 0], [3, 4])


def test_worker_supports_positional_translation_history():
    operations = [{"delta": [.25, .5], "bounds": [-.1, .1, -1.1, 1.1]}]
    _, actual = generate_grain_worker(4, 4, 0, (0, 0), None, "SC", "100", None, operations)
    expected = projected_columns(4, 4, 0, lattice="SC", axis="100", displacements=operations)
    np.testing.assert_array_equal(actual.positions, expected.positions)
    np.testing.assert_array_equal(actual.half_indices, expected.half_indices)


@pytest.mark.parametrize("operations", [
    {}, [None], [{}], [{"delta": [0]}], [{"delta": [0, np.inf]}],
    [{"delta": [0, 0], "bounds": [0, 0, 0, 1]}],
    [{"delta": [0, 0], "bounds": [0, 1, np.nan, 1]}],
    [{"delta": [0, 0], "bounds": [0, 1]}], [{"delta": [0, 0], "unknown": True}],
    [{"delta": [True, 0]}], [{"delta": ["1", "0"]}], [{"delta": [1e9, 0]}],
    [{"delta": [0, 0]}] * (MAX_DISPLACEMENT_OPERATIONS + 1),
])
def test_invalid_histories_fail_without_partially_applying_operations(operations):
    with pytest.raises(ValueError):
        normalize_displacements(operations)


def test_translations_cannot_bypass_enumeration_or_coordinate_precision_limits():
    with pytest.raises(GeometryLimitError, match="candidate columns"):
        projected_columns(1000, 1000, 0, displacements=[{"delta": [10000, 0]}])
    with pytest.raises(GeometryLimitError, match="precision"):
        projected_columns(2, 2, 0, lattice="SC", axis="100", center=[1e17, 0],
                          displacements=[{"delta": [1e8, 0]}])
