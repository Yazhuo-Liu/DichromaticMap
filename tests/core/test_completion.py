"""Integer lattice and strain checks for symmetry-assisted cell completion."""

import numpy as np
import pytest

from dichromatic_map.cells import bases
from dichromatic_map.completion import cell_completion_candidates
from dichromatic_map.crystal import get_geometry, rotation_matrix_2d
from .helpers import fcc22_diamond_pairs


def fcc21_edge():
    geometry = get_geometry("FCC", "110")
    ends = np.array([
        rotation_matrix_2d(sign * 10.5) @ (np.array(vector) @ geometry.frame[:, :2] / 2)
        for sign, vector in zip((1, -1), ([21, -21, -42], [9, -9, -50]))
    ])
    return np.stack((np.zeros((2, 2)), ends), axis=1)


def test_fcc21_grain_exchange_completes_corner_beyond_local_cutoff():
    edge = fcc21_edge()
    before = edge.copy()
    candidates = cell_completion_candidates(edge, 21)
    assert len(candidates) == 2
    candidate = next(c for c in candidates if "0.000°" in c.description
                     and "90.000°" not in c.description)
    np.testing.assert_array_equal(edge, before)
    np.testing.assert_array_equal(candidate.vertices[:, :2], edge)
    assert candidate.source == "symmetry"
    assert candidate.atoms == (714, 714)
    np.testing.assert_allclose(candidate.areas, np.sqrt(0.5) * 714)
    distances = np.linalg.norm(candidate.vertices[0] - candidate.vertices[1], axis=1)
    assert distances[1] < 0.1 < distances[2]
    np.testing.assert_allclose(distances[1], distances[3], atol=1e-12)
    mirror = np.diag([1., -1.])
    np.testing.assert_allclose(candidate.vertices[0, 3], mirror @ edge[1, 1], atol=1e-12)
    np.testing.assert_allclose(candidate.vertices[1, 3], mirror @ edge[0, 1], atol=1e-12)
    assert candidate.fit is not None
    assert 100 * candidate.fit.cell.max_strain == pytest.approx(0.198524093639, abs=1e-9)
    np.testing.assert_allclose(candidate.fit.rotations_deg, [0.06493001, -0.06493001], atol=1e-8)
    np.testing.assert_allclose(candidate.fit.vertices[0], candidate.fit.vertices[1], atol=1e-12)


@pytest.mark.parametrize("percent,rotation", [(0.1, 1.0), (2.0, 0.01), (2.0, 0.0)])
def test_completion_respects_strain_and_rotation_constraints(percent, rotation):
    candidates = cell_completion_candidates(fcc21_edge(), 21, percent=percent, max_rotation_deg=rotation)
    assert candidates
    assert all(c.fit is None and "above" in c.error for c in candidates)


@pytest.mark.parametrize("lattice,axis", [("FCC", "110"), ("BCC", "100"), ("SC", "111")])
def test_symmetry_candidates_preserve_every_layer_and_integer_translations(lattice, axis):
    geometry = get_geometry(lattice, axis)
    grain_bases = bases(0, lattice, axis)
    for layer in range(geometry.layer_count):
        offset = geometry.layer_offsets_half_indices[layer] @ geometry.frame[:, :2] / 2
        start = offset + grain_bases[0] @ [4, -3]
        edge = np.array([[start, start + grain_bases[0] @ [2, 1]]] * 2)
        candidates = cell_completion_candidates(
            edge, 0, lattice=lattice, axis=axis, layer=layer, max_rotation_deg=0,
        )
        assert candidates
        for candidate in candidates:
            for polygon, basis in zip(candidate.vertices, grain_bases):
                coefficients = np.linalg.solve(basis, (polygon - offset).T).T
                np.testing.assert_allclose(coefficients, np.rint(coefficients), atol=1e-10)
            assert candidate.fit is not None, candidate.error
            assert candidate.fit.cell.max_strain < 1e-12
            np.testing.assert_array_equal(candidate.vertices[:, :2], edge)


def test_generic_axis_without_independent_symmetry_falls_back_to_three_vertices():
    basis = bases(0, "SC", "1 2 3")[0]
    points = np.array([[0, 0], [1, 0], [1, 1]]) @ basis.T
    vertices = np.array([points, points])
    assert cell_completion_candidates(vertices[:, :2], 0, lattice="SC", axis="1 2 3") == []
    candidates = cell_completion_candidates(vertices, 0, lattice="SC", axis="1 2 3")
    assert len(candidates) == 1
    assert candidates[0].fit is not None
    np.testing.assert_allclose(candidates[0].vertices[:, 3], [basis[:, 1]] * 2, atol=1e-12)


def test_parallelogram_closure_preserves_shifted_paired_atoms_and_nonzero_layer():
    _, polygons = fcc22_diamond_pairs()
    for grain, basis in enumerate(bases(22)):
        polygons[grain] += basis @ [8, -5]
    candidates = cell_completion_candidates(polygons[:, :3], 22, layer=1)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "closure"
    assert candidate.atoms == (40, 40)
    np.testing.assert_allclose(candidate.vertices, polygons, atol=1e-12)
    np.testing.assert_array_equal(candidate.vertices[:, :3], polygons[:, :3])
    assert candidate.fit is not None
    np.testing.assert_allclose(candidate.fit.vertices[0], candidate.fit.vertices[1], atol=1e-12)


def test_degenerate_closure_and_non_atom_vertices_are_rejected():
    basis = bases(0, "FCC", "110")[0]
    collinear = np.array([np.array([[0, 0], [1, 0], [2, 0]]) @ basis.T] * 2)
    assert cell_completion_candidates(collinear, 0) == []
    collinear[0, 0, 0] += 0.123
    with pytest.raises(ValueError, match="original atoms"):
        cell_completion_candidates(collinear, 0)


def test_symmetry_is_checked_against_both_grains_not_only_the_first():
    # Generic [100] misorientation retains common quarter-turns, but the
    # first grain's own mirror is not a simultaneous mirror of both grains.
    grain_bases = bases(17, "SC", "100")
    edge = np.array([[[0, 0], basis @ [2, 1]] for basis in grain_bases])
    candidates = cell_completion_candidates(edge, 17, lattice="SC", axis="100")
    descriptions = [c.description for c in candidates]
    assert any("Rotation" in label for label in descriptions)
    assert all("exchange G1/G2" in label for label in descriptions if "Mirror" in label)
