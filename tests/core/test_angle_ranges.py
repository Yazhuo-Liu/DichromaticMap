"""Fixed-axis cubic rotation periods, reduced intervals and CSL menus."""

from dataclasses import FrozenInstanceError
from itertools import permutations, product

import numpy as np
import pytest

from dichromatic_map import crystal


@pytest.mark.parametrize("lattice", crystal.SUPPORTED_LATTICES)
@pytest.mark.parametrize(("axis", "order", "period", "maximum"), (
    ("100", 4, 90, 45),
    ("110", 2, 180, 90),
    ("111", 3, 120, 60),
    ("112", 1, 360, 180),
    ("1 -1 3", 1, 360, 180),
    ("2 3 5", 1, 360, 180),
))
def test_cubic_fixed_axis_ranges(lattice, axis, order, period, maximum):
    result = crystal.misorientation_range(axis, lattice)
    assert result.symmetry_order == order
    assert result.period_deg == period
    assert result.maximum_deg == maximum


@pytest.mark.parametrize("axis", ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 3, 5)))
def test_sign_permutation_and_integer_scale_leave_range_unchanged(axis):
    expected = crystal.misorientation_range(axis)
    for permuted in set(permutations(axis)):
        for signs in product((-1, 1), repeat=3):
            transformed = tuple(6 * sign * value for sign, value in zip(signs, permuted))
            assert crystal.misorientation_range(transformed) == expected


@pytest.mark.parametrize("axis", ("100", "110", "111"))
def test_full_period_is_a_crystal_symmetry_but_half_period_is_not(axis):
    # Rodrigues' formula checks the physical distinction between the rotation
    # period and the interval obtained after identifying the two grain orders.
    direction = np.asarray(crystal.parse_axis(axis), dtype=float)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    interval = crystal.misorientation_range(axis)
    for angle, is_symmetry in ((interval.period_deg, True), (interval.maximum_deg, False)):
        radians = np.deg2rad(angle)
        rotation = (np.cos(radians) * np.eye(3)
                    + (1 - np.cos(radians)) * np.outer(direction, direction)
                    + np.sin(radians) * cross)
        # A cubic proper rotation maps Cartesian cube edges to cube edges.
        integral = np.allclose(rotation, np.rint(rotation), atol=1e-12)
        assert integral == is_symmetry


def test_unknown_lattice_uses_explicit_generic_fallback():
    assert crystal.misorientation_range("111", "hexagonal") == crystal.AngleRange(180, None, None)
    assert crystal.misorientation_range("100", " sc ") == crystal.AngleRange(45, 90, 4)
    with pytest.raises(FrozenInstanceError):
        crystal.misorientation_range().maximum_deg = 1


@pytest.mark.parametrize("axis", ("0 0 0", "1 2", "1.5 0 1", "65 1 0"))
@pytest.mark.parametrize("lattice", ("FCC", "unknown"))
def test_invalid_axis_is_not_silently_treated_as_unknown_symmetry(axis, lattice):
    with pytest.raises(ValueError):
        crystal.misorientation_range(axis, lattice)


@pytest.mark.parametrize("axis", ("100", "110", "111", "112", "1 -1 3", "2 3 5",
                                  "0 0 -1", "0 -1 1", "-1 1 -1"))
def test_csl_presets_respect_fixed_axis_interval(axis):
    presets = crystal.csl_presets(axis)
    maximum = crystal.misorientation_range(axis).maximum_deg
    assert presets
    assert all(0 < preset.angle_deg <= maximum + 1e-10 for preset in presets)
    angles = [preset.angle_deg for preset in presets]
    assert angles == sorted(angles)
    assert len(set(angles)) == len(angles)
    assert all(preset.axis == crystal.axis_key(axis) for preset in presets)


def test_110_menu_retains_existing_order_and_sigma_variants():
    assert [(p.sigma, p.m, p.n, p.variant) for p in crystal.csl_presets("110")] == [
        (33, 8, 1, "a"), (19, 6, 1, ""), (27, 5, 1, ""), (9, 4, 1, ""),
        (11, 3, 1, ""), (33, 5, 2, "c"), (3, 2, 1, ""), (17, 3, 2, ""),
    ]


@pytest.mark.parametrize(("canonical", "equivalent"), (("100", "0 -3 0"), ("110", "-2 0 2")))
def test_equivalent_axes_share_curated_presets(canonical, equivalent):
    expected = [(p.sigma, p.angle_deg, p.variant) for p in crystal.csl_presets(canonical)]
    assert [(p.sigma, p.angle_deg, p.variant) for p in crystal.csl_presets(equivalent)] == expected


def test_100_menu_removes_complementary_representations_of_same_csl():
    presets = crystal.csl_presets("100")
    assert [p.sigma for p in presets] == [25, 13, 17, 5]
    np.testing.assert_allclose([p.angle_deg for p in presets],
                               [16.26020470831196, 22.619864948040426,
                                28.072486935852954, 36.86989764584402])


def test_111_menu_includes_sigma3_at_60_degrees():
    preset = crystal.matching_csl_preset(60, axis="111")
    assert preset is not None
    assert preset.sigma == 3
    assert (preset.m, preset.n) == (3, 1)


@pytest.mark.parametrize(("axis", "sigma"), (("112", 3), ("1 -1 3", 11), ("2 3 5", 19)))
def test_generic_axis_menu_includes_exact_half_turn(axis, sigma):
    preset = crystal.matching_csl_preset(180, axis=axis)
    assert preset is not None
    assert preset.sigma == sigma
    assert (preset.m, preset.n) == (0, 1)


def test_public_rotation_angle_calculation_keeps_complementary_angles_available():
    assert crystal.csl_angle_deg(2, 1, "100") == pytest.approx(53.13010235415598)
    assert crystal.csl_angle_deg(0, 1, "111") == 180
