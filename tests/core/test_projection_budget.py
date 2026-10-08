"""Reject full-view resource budgets before any dense candidate allocation."""

import numpy as np
import pytest
from dichromatic_map import crystal


@pytest.mark.parametrize("axis", ("2 11 1", "1 8 7"))
def test_all_selected_phases_are_budgeted_before_allocating(monkeypatch, axis):
    crystal.get_geometry("FCC", axis)
    allocations = []
    original = np.empty

    def allocate(*args, **kwargs):
        allocations.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(crystal.np, "empty", allocate)
    with pytest.raises(crystal.GeometryLimitError, match="candidate columns"):
        crystal.projected_columns(120, 90, 11, axis=axis)
    assert allocations == []


def test_budget_multiplication_cannot_wrap_to_negative(monkeypatch):
    crystal.get_geometry("SC", "100")

    def allocate(*args, **kwargs):
        pytest.fail("Overflowed budget reached a dense allocation")

    monkeypatch.setattr(crystal.np, "empty", allocate)
    with pytest.raises(crystal.GeometryLimitError, match="candidate columns"):
        crystal.projected_columns(2**32 - 4, 2**32 - 4, 0, lattice="SC", axis="100")


@pytest.mark.parametrize("options", (
    {"width": 1.6e308, "height": 4, "center": (1.6e308, 0)},
    {"width": 4, "height": 4, "center": (float(2**63), 0)},
    {"width": 4, "height": 4, "center": (float(2**62), 0)},
    {"width": 4, "height": 4, "deformation": np.diag([1e-320, 1])},
))
def test_coordinate_overflow_is_rejected_before_integer_conversion(monkeypatch, options):
    crystal.get_geometry("SC", "100")

    def allocate(*args, **kwargs):
        pytest.fail("Out-of-range coordinates reached a dense allocation")

    monkeypatch.setattr(crystal.np, "empty", allocate)
    with np.errstate(all="raise"), pytest.raises(crystal.GeometryLimitError, match="coordinate range"):
        crystal.projected_columns(rotation_deg=0, lattice="SC", axis="100", **options)


def test_unselected_phases_do_not_contribute_to_the_budget():
    # The complete 126-layer grain exceeds the limit, while two real phases
    # fit. Their IDs and full axial period must survive the preflight pass.
    grain = crystal.projected_columns(120, 90, 11, axis="2 11 1", layers=[125, 3])
    assert grain.layer_count == 126
    assert set(grain.layers) == {3, 125}
    for layer in (3, 125):
        single = crystal.projected_columns(120, 90, 11, axis="2 11 1", layers=[layer])
        mask = grain.layers == layer
        np.testing.assert_array_equal(grain.positions[mask], single.positions)
        np.testing.assert_array_equal(grain.half_indices[mask], single.half_indices)
