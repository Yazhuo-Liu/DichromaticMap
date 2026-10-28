"""Absolute translation controls retain stable atoms and legacy histories."""

import json
import zipfile

import numpy as np
import pytest

from dichromatic_map.displacements import (
    MAX_DISPLACEMENT_OPERATIONS, apply_displacements,
    set_region_displacement, set_whole_grain_displacement,
    whole_grain_displacement,
)


def test_whole_targets_replace_previous_offset_and_zero_restores_origin():
    points = np.array([[0., 0.], [1., 2.]])
    first = set_whole_grain_displacement([], [1, .5])
    second = set_whole_grain_displacement(first, [2, -.5])
    np.testing.assert_array_equal(apply_displacements(points, second), points + [2, -.5])
    assert whole_grain_displacement(second) == [2, -.5]
    repeated = set_whole_grain_displacement(second, [2, -.5])
    assert repeated == second
    assert repeated is not second and repeated[0] is not second[0]
    restored = set_whole_grain_displacement(second, [0, 0])
    np.testing.assert_array_equal(apply_displacements(points, restored), points)
    assert first == [{"bounds": None, "delta": [1, .5]}]


def test_region_absolute_edit_does_not_capture_atoms_at_its_destination():
    points = np.array([[0., 0.], [1., 0.], [2., 0.]])
    bounds = [-.1, .1, -.1, .1]
    first = set_region_displacement([], [1, 0], bounds)
    # The moved x=0 atom now coincides with the unchanged x=1 atom.
    np.testing.assert_array_equal(apply_displacements(points, first), [[1, 0], [1, 0], [2, 0]])
    # Rebuild from the selection snapshot, never reselect the destination.
    second = set_region_displacement([], [2, 0], bounds)
    np.testing.assert_array_equal(apply_displacements(points, second), [[2, 0], [1, 0], [2, 0]])
    assert set_region_displacement([], [0, 0], bounds) == []
    # Keeping the complete previous history is sufficient to undo this edit.
    np.testing.assert_array_equal(apply_displacements(points, first), [[1, 0], [1, 0], [2, 0]])


def test_new_selection_can_overlap_an_earlier_selection_without_rewriting_it():
    points = np.array([[0., 0.], [1., 0.], [2., 0.]])
    first = set_region_displacement([], [1, 0], [-.1, .1, -.1, .1])
    # A deliberately new selection includes both atoms now at x=1.
    second = set_region_displacement(first, [0, 2], [.9, 1.1, -.1, .1])
    np.testing.assert_array_equal(apply_displacements(points, second), [[1, 2], [1, 2], [2, 0]])
    reset_region = set_region_displacement(first, [0, 0], [.9, 1.1, -.1, .1])
    assert reset_region == first
    reset_region[0]["delta"][0] = 99
    assert first[0]["delta"] == [1, 0]


def test_edited_distant_region_still_generates_only_its_original_atom():
    from dichromatic_map.crystal import projected_columns
    bounds = [999999.9, 1000000.1, -.1, .1]
    for target, destination in [([-1000000, 0], [0, 0]),
                                ([-1000001, 0], [-1, 0])]:
        history = set_region_displacement([], target, bounds)
        grain = projected_columns(4, 4, 0, lattice="SC", axis="100", displacements=history)
        moved = np.all(grain.half_indices == [0, 2000000, 0], axis=1)
        np.testing.assert_array_equal(grain.positions[moved], [destination])
        # The original center atom remains at zero after both absolute edits.
        center = np.all(grain.half_indices == [0, 0, 0], axis=1)
        np.testing.assert_array_equal(grain.positions[center], [[0, 0]])
        assert len(grain.positions) == 26


def test_whole_target_carries_regions_without_changing_their_selected_identities():
    points = np.array([[0., 0.], [1., 0.], [2., 0.]])
    baseline = set_whole_grain_displacement([], [2, .5])
    region = set_region_displacement(baseline, [-1, 1], [1.9, 2.1, .4, .6])
    whole = set_whole_grain_displacement(region, [3, -.5])
    assert whole_grain_displacement(whole) == [3, -.5]
    np.testing.assert_array_equal(apply_displacements(points, whole), [[2, .5], [4, -.5], [5, -.5]])
    reset_whole = set_whole_grain_displacement(whole, [0, 0])
    np.testing.assert_array_equal(apply_displacements(points, reset_whole), [[-1, 1], [1, 0], [2, 0]])


def test_unchanged_whole_target_at_operation_limit_does_not_consume_history():
    history = [{"delta": [.25, 0], "bounds": None}] * MAX_DISPLACEMENT_OPERATIONS
    assert set_whole_grain_displacement(history, [16, 0]) == history
    with pytest.raises(ValueError, match="At most"):
        set_whole_grain_displacement(history, [17, 0])


@pytest.mark.parametrize("target", [[True, 0], [float("nan"), 0], [1e9, 0], [0], ["1", 0]])
def test_absolute_controls_reject_invalid_targets_without_modifying_history(target):
    history = [{"delta": [1., 0.], "bounds": None}]
    with pytest.raises(ValueError):
        set_whole_grain_displacement(history, target)
    with pytest.raises(ValueError):
        set_region_displacement(history, target, [-1, 1, -1, 1])
    assert history == [{"delta": [1., 0.], "bounds": None}]


@pytest.mark.parametrize("bounds", [None, [0, 0, -1, 1], [0, 1, float("inf"), 1]])
def test_zero_region_target_still_validates_selection(bounds):
    with pytest.raises(ValueError):
        set_region_displacement([], [0, 0], bounds)


def test_absolute_edits_roundtrip_as_ordinary_v5_incremental_session(tmp_path):
    from dichromatic_map.session import load_session, save_session
    from dichromatic_map.state import PatternParameters, PatternState
    state = PatternState(PatternParameters(lattice="SC", axis="100", angle_deg=0))
    # This is a history written by the previous release, with additive deltas.
    legacy = [{"delta": [1, 0], "bounds": None},
              {"delta": [0, .5], "bounds": [.9, 1.1, -.1, .1]},
              {"delta": [2, 0], "bounds": None}]
    state.grain_displacements = [set_whole_grain_displacement(legacy, [4, 0]), []]
    settings = dict(region_states=(True,) * 4, manual_visible=False,
                    show_common_cell=False, local_distance=.05, strain_percent=2.,
                    search_index=12, manual_strain_percent=2., manual_rotation_deg=1.)
    path = tmp_path / "absolute-offset.dmap"
    save_session(path, state, settings, (-5, 5, -5, 5))
    with zipfile.ZipFile(path) as archive:
        saved = json.loads(archive.read("session.json"))
    assert saved["schema_version"] == 5
    restored = load_session(path).state
    assert restored.grain_displacements == state.grain_displacements
    assert restored.grain_displacements[0][:3] == legacy
    np.testing.assert_array_equal(
        apply_displacements([[0, 0], [1, 0]], restored.grain_displacements[0]),
        [[4, .5], [5, 0]],
    )
