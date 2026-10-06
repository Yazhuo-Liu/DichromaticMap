"""Partial axial generation and mutable, order-preserving layer indices."""

import pickle
import numpy as np
import pytest
from dichromatic_map import crystal, cells, matching


@pytest.mark.parametrize("lattice", ("FCC", "BCC", "SC"))
@pytest.mark.parametrize("axis", ("100", "112", "2 3 5"))
def test_selected_layers_equal_full_generation_in_reference_order(lattice, axis):
    geometry = crystal.get_geometry(lattice, axis)
    requested = [geometry.layer_count - 1, 0, geometry.layer_count - 1]
    # A cropped/translated strained view also represents matching's halo
    # rectangles. Selecting phases must change no edge or half-index choices.
    options = dict(center=(13.5, -8.25),
                   deformation=np.array([[1.02, 0.015], [0.015, 0.97]]),
                   lattice=lattice, axis=axis, translation=np.array([0.3, -0.4]))
    full = crystal.projected_columns(6.4, 5.2, 23.4, **options)
    partial = crystal.projected_columns(6.4, 5.2, 23.4, layers=requested, **options)
    keep = np.isin(full.layers, requested)
    for name in ("positions", "layers", "half_indices"):
        np.testing.assert_array_equal(getattr(partial, name), getattr(full, name)[keep])
    assert partial.layer_count == full.layer_count == geometry.layer_count
    for layer, selection in partial.layer_selections().items():
        assert isinstance(selection, slice)
        np.testing.assert_array_equal(partial.layers[selection],
                                      np.full(len(partial.layers[selection]), layer))
        assert np.shares_memory(partial.positions[selection], partial.positions)


def test_empty_layer_selection_keeps_metadata_and_array_shapes():
    grain = crystal.projected_columns(2, 2, 17, axis="112", layers=[])
    assert grain.layer_count == 6
    assert grain.positions.shape == (0, 2)
    assert grain.layers.shape == (0,)
    assert grain.half_indices.shape == (0, 3)
    assert grain.half_indices.dtype.kind == "i"
    assert grain.layer_selections() == {}
    assert len(matching.same_layer_coincidence_sites(grain, grain, 1e-6)) == 6
    assert len(matching.local_near_pairs(grain, grain).layers) == 0


@pytest.mark.parametrize("requested", ([-1], [6], [1.5], [np.nan], ["1"], 1))
def test_invalid_selected_layers_raise_clear_error(requested):
    with pytest.raises(ValueError, match="valid axial layer indices"):
        crystal.projected_columns(2, 2, 0, axis="112", layers=requested)


def test_layer_indices_preserve_external_order_and_follow_mutation():
    positions = np.arange(12, dtype=float).reshape(6, 2)
    layers = np.array([4, 1, 4, 0, 1, 4], dtype=np.int16)
    grain = crystal.ProjectedGrain(positions, layers, np.arange(18).reshape(6, 3), 6)
    for mapping in (grain.layer_selections(), grain.layer_selections()):
        assert set(mapping) == {0, 1, 4}
        for layer, selection in mapping.items():
            np.testing.assert_array_equal(grain.positions[selection],
                                          positions[layers == layer])
    # The mapping returned to callers cannot replace the cached selections.
    mapping.clear()
    assert set(grain.layer_selections()) == {0, 1, 4}
    layers[:] = [1, 1, 4, 4, 4, 0]
    mapping = grain.layer_selections()
    assert all(isinstance(selection, slice) for selection in mapping.values())
    for layer, selection in mapping.items():
        np.testing.assert_array_equal(positions[selection], positions[layers == layer])
    positions[2] = [-3, -4]
    np.testing.assert_array_equal(grain.positions[mapping[4]][0], [-3, -4])
    # Compute coordinators pass grains through spawn process serialization.
    restored = pickle.loads(pickle.dumps(grain))
    for layer, selection in restored.layer_selections().items():
        np.testing.assert_array_equal(restored.positions[selection],
                                      positions[layers == layer])


def test_matching_keeps_noncontiguous_layers_and_external_input_order():
    positions = np.array([[3, 0], [1, 0], [4, 0], [0, 0], [2, 0]], dtype=float)
    layers = np.array([4, 1, 4, 0, 1], dtype=np.int16)
    indices = np.zeros((5, 3), dtype=int)
    first = crystal.ProjectedGrain(positions, layers, indices, 6)
    same = crystal.ProjectedGrain(positions[::-1].copy(), layers[::-1].copy(), indices, 6)
    exact = matching.same_layer_coincidence_sites(first, same, 1e-6)
    assert len(exact) == 6
    for layer, sites in enumerate(exact):
        np.testing.assert_array_equal(sites, positions[layers == layer])
    second = crystal.ProjectedGrain(same.positions + [0.0625, 0], same.layers, indices, 6)
    pairs = matching.local_near_pairs(first, second, 0.125, layers=[4, 2, 1])
    np.testing.assert_array_equal(pairs.layers, [4, 4, 1, 1])
    np.testing.assert_array_equal(pairs.first, positions[[0, 2, 1, 4]])
    np.testing.assert_array_equal(pairs.second, pairs.first + [0.0625, 0])
    # Mutating an external layer array after a first match must be respected.
    first.layers[0] = 2
    exact = matching.same_layer_coincidence_sites(first, same, 1e-6)
    assert len(exact[2]) == 0
    np.testing.assert_array_equal(exact[4], [[4, 0]])


@pytest.mark.parametrize("lattice", ("FCC", "BCC", "SC"))
def test_selected_cell_count_generates_only_one_phase(monkeypatch, lattice):
    polygon = np.array([[-3, -2], [4, -2], [4, 3], [-3, 3]], dtype=float)
    deformations = (np.array([[1.02, 0.01], [0.01, 0.99]]), np.eye(2))
    translations = np.array([[0.23, -0.31], [-0.13, 0.29]])
    boundary = np.array([[0, -5], [0, 5]])
    options = dict(lattice=lattice, axis="112", translations=translations,
                   boundary_points=boundary, region_states=(True, False, True, True))
    all_layers = cells.count_cell_atoms(polygon, 21.3, deformations, **options)
    calls = []
    original = cells.projected_columns

    def generate(*args, **kwargs):
        grain = original(*args, **kwargs)
        calls.append((kwargs.get("layers"), set(grain.layers)))
        return grain

    monkeypatch.setattr(cells, "projected_columns", generate)
    selected = cells.count_cell_atoms(polygon, 21.3, deformations, layer=4, **options)
    assert calls == [((4,), {4}), ((4,), {4})]
    for name in ("interior", "boundary", "half_open", "half_open_edges",
                 "half_open_corners"):
        expected = np.zeros_like(getattr(all_layers, name))
        expected[:, 4] = getattr(all_layers, name)[:, 4]
        np.testing.assert_array_equal(getattr(selected, name), expected)
    np.testing.assert_array_equal(selected.areas, all_layers.areas)
