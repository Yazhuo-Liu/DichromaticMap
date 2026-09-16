"""Raw high-angle CSL geometry and fitting remain independent of UI reduction."""

import numpy as np
import pytest

from dichromatic_map import cells, crystal, matching, strain
from .helpers import solve


@pytest.mark.parametrize("lattice", ("FCC", "BCC", "SC"))
@pytest.mark.parametrize("axis", ("112", "1 -1 3"))
def test_high_angle_exact_cell_against_three_dimensional_rotation(lattice, axis):
    direction = np.array(crystal.parse_axis(axis), dtype=float)
    angle = np.degrees(2 * np.arctan2(np.linalg.norm(direction), 1))
    assert 90 < angle < 180
    cell = matching.exact_csl_cell(angle, lattice=lattice, axis=axis)
    assert cell is not None
    geometry = crystal.get_geometry(lattice, axis)
    first, second = cells.bases(angle, lattice, axis)
    np.testing.assert_allclose(first @ cell.m1, second @ cell.m2, atol=1e-10)

    # Independent Rodrigues rotation in cubic coordinates verifies that both
    # common translations remain legal three-dimensional lattice vectors.
    unit = direction / np.linalg.norm(direction)
    x, y, z = unit
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    radians = np.deg2rad(angle)
    rotation = (np.cos(radians) * np.eye(3)
                + (1 - np.cos(radians)) * np.outer(unit, unit)
                + np.sin(radians) * cross)
    first_indices = geometry.basis_half_indices @ cell.m1
    second_indices = geometry.basis_half_indices @ cell.m2
    np.testing.assert_allclose(rotation @ first_indices, second_indices, atol=1e-10)
    np.testing.assert_array_equal(direction @ first_indices, [0, 0])
    np.testing.assert_array_equal(direction @ second_indices, [0, 0])
    assert abs(cells.determinant(cell.m1)) == (7 if axis == "112" else 3)


@pytest.mark.parametrize("lattice", ("FCC", "BCC", "SC"))
def test_half_turn_has_primitive_common_plane_and_valid_selected_fit(lattice):
    axis = "112"
    cell = matching.exact_csl_cell(180, lattice=lattice, axis=axis)
    assert cell is not None
    assert abs(cells.determinant(cell.m1)) == 1
    assert abs(cells.determinant(cell.m2)) == 1
    np.testing.assert_array_equal(cell.m2, -cell.m1)
    first, second = cells.bases(180, lattice, axis)
    np.testing.assert_allclose(first @ cell.m1, second @ cell.m2, atol=1e-12)

    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    polygon = uv @ cell.cell.T
    fitted = strain.strain_selected_cell(
        np.stack((polygon, polygon)), 180, lattice=lattice, axis=axis
    )
    np.testing.assert_allclose(fitted.vertices[0], fitted.vertices[1], atol=1e-12)
    assert fitted.cell.max_strain < 1e-12


@pytest.mark.parametrize("lattice", ("FCC", "BCC", "SC"))
def test_selected_high_angle_cell_fits_all_four_atomic_corners(lattice):
    axis = "112"
    exact_angle = np.degrees(2 * np.arctan2(np.sqrt(6), 1))
    reference = matching.exact_csl_cell(exact_angle, lattice=lattice, axis=axis)
    angle = exact_angle + 0.2
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    vertices = np.stack([
        uv @ matrix.T @ basis.T
        for matrix, basis in zip((reference.m1, reference.m2),
                                 cells.bases(angle, lattice, axis))
    ])
    fitted = strain.strain_selected_cell(vertices, angle, lattice=lattice, axis=axis)
    assert fitted.residual < 1e-10
    assert fitted.cell.max_strain < 0.02
    np.testing.assert_allclose(fitted.vertices[0], fitted.vertices[1], atol=1e-10)
    for basis, matrix, deformation in zip(
        cells.bases(angle, lattice, axis), (fitted.cell.m1, fitted.cell.m2),
        (fitted.cell.f1, fitted.cell.f2),
    ):
        np.testing.assert_allclose(deformation @ basis @ matrix,
                                   fitted.cell.cell, atol=1e-10)


def test_high_angle_strain_search_and_raw_complementary_angles_remain_valid():
    angle = np.degrees(2 * np.arctan2(np.sqrt(2), 1))
    candidates = solve("FCC", "110", angle, extent=2)
    assert candidates
    assert candidates[0].max_strain < 1e-10
    first, second = cells.bases(angle, "FCC", "110")
    np.testing.assert_allclose(first @ candidates[0].m1,
                               second @ candidates[0].m2, atol=1e-10)
    endpoint = solve("SC", "112", 180, extent=2)
    assert endpoint and endpoint[0].max_strain < 1e-10

    # The raw solver still supports angles outside the viewer's reduced
    # <100> range; these rotations are useful to numerical API callers.
    for angle in (np.degrees(2 * np.arctan2(1, 2)), 90, 180):
        assert matching.exact_csl_cell(angle, lattice="SC", axis="100") is not None


def test_near_half_turn_recognition_is_bounded_and_invalid_angles_rejected():
    # A finite, nearly singular tangent must not allocate huge integer cells.
    for angle in (179.999999, np.nextafter(180.0, 0.0)):
        assert matching.exact_csl_cell(angle, axis="112") is None
    high_order = np.degrees(2 * np.arctan2(np.sqrt(6) * 129, 1))
    assert matching.exact_csl_cell(high_order, axis="112") is None
    assert matching.exact_csl_cell(high_order, max_denominator=129, axis="112") is not None
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    for angle in (-0.01, 180.01, np.nan, np.inf):
        assert matching.exact_csl_cell(angle, axis="112") is None
        with pytest.raises(ValueError, match="Reference angle"):
            strain.strain_selected_cell(np.stack((polygon, polygon)), angle,
                                        lattice="SC", axis="100")
