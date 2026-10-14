"""A model-space translation must not enlarge a projected crop."""

from itertools import product
from math import ceil, floor
import numpy as np
import pytest
from dichromatic_map import crystal


@pytest.mark.parametrize("center", (
    (1e12 + .2, 0), (0, -1e12 - .2), (1e12 + .2, -1e12 - .2),
    (1e12 + 1.0001220703125, 0),
))
def test_large_sc_crops_equal_an_independent_integer_grid_oracle(center):
    center = np.asarray(center)
    width, height = 2., 2.
    expected = set(product(range(ceil(center[0] - 1), floor(center[0] + 1) + 1),
                           range(ceil(center[1] - 1), floor(center[1] + 1) + 1)))
    grain = crystal.projected_columns(width, height, 0, center=center,
                                      lattice="SC", axis="100")
    assert set(map(tuple, grain.positions)) == expected
    assert len(grain.positions) == len(expected)
    assert np.all(np.abs(grain.positions - center) <= [width / 2, height / 2])
    # SC [100] has one phase, with screen x/y equal to reference y/z.
    np.testing.assert_array_equal(grain.half_indices[:, 0], 0)
    np.testing.assert_array_equal(grain.half_indices[:, 1:] / 2, grain.positions)


@pytest.mark.parametrize("lattice", ("FCC", "BCC", "SC"))
@pytest.mark.parametrize("axis", ("100", "112"))
def test_large_uniform_translation_keeps_layer_ids_and_crop_membership(lattice, axis):
    center = np.array([.231, -.347])
    translation = np.array([1e12, -1e12])
    deformation = np.array([[1.03, .015], [.015, .98]])
    layers = [0, crystal.get_geometry(lattice, axis).layer_count - 1]
    options = dict(width=5.2, height=4.6, rotation_deg=23.4,
                   deformation=deformation, lattice=lattice, axis=axis, layers=layers)
    reference = crystal.projected_columns(center=center, **options)
    translated = crystal.projected_columns(center=center + translation,
                                           translation=translation, **options)
    np.testing.assert_array_equal(translated.half_indices, reference.half_indices)
    np.testing.assert_array_equal(translated.layers, reference.layers)
    assert translated.layer_count == reference.layer_count
    np.testing.assert_allclose(translated.positions - translation, reference.positions,
                               atol=2 * np.spacing(1e12), rtol=0)
    assert np.all(np.abs(translated.positions - (center + translation)) <= [2.6, 2.3])


def test_unresolvable_atom_coordinates_fail_before_candidate_allocation(monkeypatch):
    crystal.get_geometry("SC", "100")

    def allocate(*args, **kwargs):
        pytest.fail("A precision failure reached dense candidate allocation")

    monkeypatch.setattr(crystal.np, "empty", allocate)
    with pytest.raises(crystal.GeometryLimitError, match="precision"):
        crystal.projected_columns(2, 2, 0, center=(1e17, 0), lattice="SC", axis="100")


def test_precision_guard_respects_anisotropic_coordinate_scales():
    grain = crystal.projected_columns(2, 2e-12, 0, center=(1e10, 0),
        deformation=np.diag([1., 1e-12]), lattice="SC", axis="100")
    assert len(grain.positions) == 9
    assert len(set(map(tuple, grain.positions))) == 9


@pytest.mark.parametrize("center,expected_mesh_dtype", (
    ((.2, -.3), np.int32),
    ((1e12 + .2, 0), np.int64),
    ((0, -1e12 - .2), np.int64),
    # The mesh fits int32, but its reference half-indices do not.
    ((1.5e9 + .2, 0), np.int64),
    ((0, -1.5e9 - .2), np.int64),
))
def test_32_bit_mesh_storage_preserves_sc_coordinates_and_reference_ids(
    monkeypatch, center, expected_mesh_dtype,
):
    reference = crystal.projected_columns(2, 2, 0, center=center, lattice="SC", axis="100")
    monkeypatch.setattr(crystal, "_MESH_INDEX_DTYPE", np.dtype(np.int32))
    allocated_dtypes = []
    original_empty = np.empty

    def allocate(shape, *args, **kwargs):
        if len(shape) == 3 and shape[-1] == 2:
            allocated_dtypes.append(np.dtype(kwargs["dtype"]))
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(crystal.np, "empty", allocate)
    generated = crystal.projected_columns(2, 2, 0, center=center, lattice="SC", axis="100")
    np.testing.assert_array_equal(generated.positions, reference.positions)
    np.testing.assert_array_equal(generated.layers, reference.layers)
    np.testing.assert_array_equal(generated.half_indices, reference.half_indices)
    assert allocated_dtypes == [np.dtype(expected_mesh_dtype)]
    expected = set(product(range(ceil(center[0] - 1), floor(center[0] + 1) + 1),
                           range(ceil(center[1] - 1), floor(center[1] + 1) + 1)))
    assert set(map(tuple, generated.positions)) == expected
    np.testing.assert_array_equal(generated.half_indices[:, 1:] / 2, generated.positions)


def test_32_bit_mesh_storage_keeps_noncontinuous_phase_order_and_ids(monkeypatch):
    geometry = crystal.get_geometry("FCC", "112")
    options = dict(width=5.2, height=4.6, rotation_deg=23.4,
                   center=(1e12 + .2, -1e12 - .3), lattice="FCC", axis="112",
                   layers=[geometry.layer_count - 1, 0, geometry.layer_count - 1])
    reference = crystal.projected_columns(**options)
    monkeypatch.setattr(crystal, "_MESH_INDEX_DTYPE", np.dtype(np.int32))
    generated = crystal.projected_columns(**options)
    np.testing.assert_array_equal(generated.positions, reference.positions)
    np.testing.assert_array_equal(generated.layers, reference.layers)
    np.testing.assert_array_equal(generated.half_indices, reference.half_indices)
    assert generated.layer_count == geometry.layer_count
    assert np.all(np.diff(generated.layers) >= 0)
    assert set(generated.layers) == {0, geometry.layer_count - 1}


@pytest.mark.parametrize("options,error", (
    ({"width": 120, "height": 90, "axis": "2 11 1"}, "candidate columns"),
    ({"width": 2, "height": 2, "center": (1e17, 0),
      "lattice": "SC", "axis": "100"}, "precision"),
    ({"width": 4, "height": 4, "center": (float(2**63), 0),
      "lattice": "SC", "axis": "100"}, "coordinate range"),
))
def test_32_bit_storage_preserves_preallocation_limits(monkeypatch, options, error):
    crystal.get_geometry(options.get("lattice", "FCC"), options["axis"])
    monkeypatch.setattr(crystal, "_MESH_INDEX_DTYPE", np.dtype(np.int32))

    def allocate(*args, **kwargs):
        pytest.fail("A rejected view reached dense candidate allocation")

    monkeypatch.setattr(crystal.np, "empty", allocate)
    with pytest.raises(crystal.GeometryLimitError, match=error):
        crystal.projected_columns(rotation_deg=0, **options)
