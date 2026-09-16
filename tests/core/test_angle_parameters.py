"""Viewer launch parameters obey the same fixed-axis ranges as the controls."""

import numpy as np
import pytest
from dichromatic_map.state import PatternParameters, PatternState


@pytest.mark.parametrize("axis,maximum", [
    ("100", 45), ("110", 90), ("111", 60), ("112", 180),
    ("0 -2 0", 45), ("-2 2 2", 60), ("1 -1 3", 180),
])
def test_launch_range_endpoints_and_default_presets(axis, maximum):
    for angle in (0, maximum, None):
        state = PatternState(PatternParameters(axis=axis, angle_deg=angle))
        assert state.angle_range.maximum_deg == maximum
        assert 0 <= state.angle_deg <= maximum
        assert all(0 < preset.angle_deg <= maximum + 1e-10 for preset in state.presets)
        if angle is not None:
            assert state.angle_deg == pytest.approx(angle)
    with pytest.raises(ValueError, match=f"between 0 and {maximum}"):
        PatternParameters(axis=axis, angle_deg=maximum + 0.01).validate()


@pytest.mark.parametrize("angle", [-0.01, np.nan, np.inf, -np.inf])
def test_invalid_launch_angle_is_rejected_instead_of_wrapped(angle):
    with pytest.raises(ValueError, match="between 0 and 60"):
        PatternParameters(axis="111", angle_deg=angle).validate()
