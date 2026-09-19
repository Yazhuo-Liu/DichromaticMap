"""Integer direction labels and polar-transported orthogonal reference axes."""

from math import gcd

import numpy as np
import pytest

from dichromatic_map import crystal
from .helpers import rotation


@pytest.mark.parametrize(("axis", "expected"), (
    ("100", [[0, 1, 0], [0, 0, 1]]),
    ("110", [[-1, 1, 0], [0, 0, 1]]),
    ("111", [[-1, 1, 0], [-1, -1, 2]]),
    ("112", [[-1, 1, 0], [-1, -1, 1]]),
    ("1 -1 3", [[1, 1, 0], [-3, 3, 2]]),
    ("2 3 5", [[-3, 2, 0], [-10, -15, 13]]),
    ("0 0 1", [[1, 0, 0], [0, 1, 0]]),
    ("0 0 -1", [[-1, 0, 0], [0, 1, 0]]),
    ("-2 2 -6", [[-1, -1, 0], [-3, 3, 2]]),
))
def test_reference_directions_are_exact_reduced_and_right_handed(axis, expected):
    directions = crystal.in_plane_reference_directions(axis)
    np.testing.assert_array_equal(directions, expected)
    tilt = np.array(crystal.parse_axis(axis))
    np.testing.assert_array_equal(directions @ tilt, [0, 0])
    assert directions[0] @ directions[1] == 0
    assert np.cross(*directions) @ tilt > 0
    assert all(gcd(*(int(component) for component in row)) == 1 for row in directions)
    assert directions.dtype.kind == "i"
    assert not directions.flags.writeable
    with pytest.raises(ValueError):
        directions[0, 0] = 99


@pytest.mark.parametrize("lattice", crystal.SUPPORTED_LATTICES)
@pytest.mark.parametrize("axis", ("100", "110", "111", "112", "1 -1 3", "0 0 -1"))
def test_reference_directions_and_axes_match_existing_crystal_frame(lattice, axis):
    geometry = crystal.get_geometry(lattice, axis)
    directions = crystal.in_plane_reference_directions(axis)
    unit_directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    np.testing.assert_allclose(unit_directions, geometry.frame[:, :2].T, atol=1e-12)

    # A rotated symmetric stretch has a known polar factor. Check the full
    # current cubic frame rather than applying the same 2D helper twice.
    f = rotation(0.8) @ np.array([[1.04, 0.013], [0.013, 0.97]])
    for grain_rotation in (-26.5, 26.5):
        vectors = crystal.crystal_vector_coordinates(
            [1, 2, 3], grain_rotation, f, lattice=lattice, axis=axis
        )
        expected = unit_directions @ vectors.current_frame.T
        axes = crystal.in_plane_reference_axes(grain_rotation, f)
        np.testing.assert_allclose(axes, expected[:, :2], atol=1e-12)
        np.testing.assert_allclose(expected[:, 2], 0, atol=1e-12)
        np.testing.assert_allclose(axes @ axes.T, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(axes), 1, atol=1e-12)


@pytest.mark.parametrize("grain_rotation", (-35.0, 0.0, 28.0))
def test_reference_axes_depend_on_polar_rotation_not_symmetric_stretch(grain_rotation):
    np.testing.assert_allclose(crystal.in_plane_reference_axes(grain_rotation),
                               rotation(grain_rotation).T, atol=1e-12)
    for polar_angle in (-1.3, 0.0, 2.1):
        expected = rotation(grain_rotation + polar_angle).T
        for stretch in (np.eye(2), np.diag([0.9, 1.1]),
                        np.array([[1.05, 0.09], [0.09, 0.96]])):
            axes = crystal.in_plane_reference_axes(
                grain_rotation, rotation(polar_angle) @ stretch
            )
            np.testing.assert_allclose(axes, expected, atol=1e-12)
            assert not axes.flags.writeable


def test_simple_shear_preserves_indicator_orthogonality_and_includes_polar_rotation():
    shear = 0.2
    deformation = np.array([[1, shear], [0, 1]])
    axes = crystal.in_plane_reference_axes(0, deformation)
    # For a positive simple xy shear, polar rotation is clockwise by atan(k/2).
    angle = -np.degrees(np.arctan(shear / 2))
    np.testing.assert_allclose(axes, rotation(angle).T, atol=1e-12)
    np.testing.assert_allclose(axes @ axes.T, np.eye(2), atol=1e-12)
    actual = deformation.T / np.linalg.norm(deformation.T, axis=1)[:, None]
    assert actual[0] @ actual[1] > 0.1


@pytest.mark.parametrize("deformation", (
    np.zeros((2, 2)), np.diag([-1, 1]), np.diag([1e-13, 1]),
    [[1, 0], [0, np.nan]], [[1, 0], [0, np.inf]], np.eye(3),
))
def test_reference_axes_reject_invalid_deformation(deformation):
    with pytest.raises(ValueError):
        crystal.in_plane_reference_axes(0, deformation)


@pytest.mark.parametrize("angle", (np.nan, np.inf, -np.inf))
def test_reference_axes_reject_nonfinite_rotation(angle):
    with pytest.raises(ValueError):
        crystal.in_plane_reference_axes(angle)


@pytest.mark.parametrize("axis", ("0 0 0", "1 2", "1.5 0 1", "65 1 0"))
def test_reference_directions_reuse_axis_validation(axis):
    with pytest.raises(ValueError):
        crystal.in_plane_reference_directions(axis)
