"""Axis changes keep numerical state, CSL choices and angle widgets in agreement."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytestmark = pytest.mark.gui


def assert_range(window, maximum):
    state, controls = window.state, window.controls
    assert state.angle_range.maximum_deg == maximum
    assert controls.angle_spin.minimum() == 0
    assert controls.angle_spin.maximum() == maximum
    assert controls.angle_slider.minimum() == 0
    assert controls.angle_slider.maximum() == maximum * 100
    assert f"0–{maximum}°" in controls.angle_range_label.text()
    assert 0 <= state.angle_deg <= maximum
    assert controls.angle_spin.value() == pytest.approx(state.angle_deg, abs=0.0051)
    assert controls.angle_slider.value() == round(state.angle_deg * 100)
    assert controls.preset_combo.count() == len(state.presets) + 1
    assert all(0 < preset.angle_deg <= maximum + 1e-10 for preset in state.presets)


@pytest.mark.parametrize("lattice,axis,maximum", [
    ("FCC", "100", 45), ("BCC", "110", 90), ("SC", "111", 60),
    ("FCC", "112", 180), ("BCC", "0 -2 0", 45), ("SC", "2 -2 -2", 60),
])
def test_axis_specific_range_at_launch_and_both_angle_controls(gui, lattice, axis, maximum):
    window = gui.window(lattice=lattice, axis=axis, angle_deg=maximum)
    assert_range(window, maximum)
    window.controls.angle_slider.setValue(maximum * 50)
    gui.settle(window)
    assert window.state.angle_deg == pytest.approx(maximum / 2, abs=0.006)
    assert_range(window, maximum)
    window.controls.angle_spin.setValue(maximum + 10)
    gui.settle(window)
    assert window.state.angle_deg == pytest.approx(maximum)
    assert_range(window, maximum)


def test_axis_switches_cancel_pending_angles_and_refresh_presets(gui):
    window = gui.window(lattice="FCC", axis="112", angle_deg=135)
    controls = window.controls
    for axis, maximum in (("111", 60), ("100", 45), ("110", 90), ("112", 180)):
        # Queue an angle from the previous domain without waiting for its timer.
        controls.angle_spin.setValue(controls.angle_spin.maximum())
        controls.axis_combo.setCurrentIndex(controls.axis_combo.findData(axis))
        gui.settle(window)
        assert window.state.geometry.axis == axis
        assert_range(window, maximum)
        controls.preset_combo.setCurrentIndex(controls.preset_combo.count() - 1)
        gui.settle(window)
        assert window.state.angle_deg == pytest.approx(window.state.presets[-1].angle_deg)
        assert_range(window, maximum)
    assert window.state.angle_deg == 180
    assert window.common_cell is not None
    for lattice in ("BCC", "SC"):
        controls.structure_combo.setCurrentIndex(controls.structure_combo.findData(lattice))
        gui.settle(window)
        assert window.state.geometry.lattice == lattice
        assert window.state.angle_deg == 180
        assert_range(window, 180)


def test_custom_axis_alias_and_invalid_axis_keep_correct_range(gui):
    window = gui.window(lattice="SC", axis="112", angle_deg=130)
    controls = window.controls
    controls.axis_combo.setCurrentIndex(controls.axis_combo.findData(None))
    for text, maximum in (("-2 2 2", 60), ("0 -2 0", 45), ("1 -1 3", 180)):
        controls.custom_axis_edit.setText(text)
        controls.custom_axis_apply.click()
        gui.settle(window)
        assert_range(window, maximum)
    before = (window.state.geometry, window.state.angle_deg, controls.angle_range_label.text())
    controls.custom_axis_edit.setText("0 0 0")
    controls.custom_axis_apply.click()
    gui.settle(window)
    assert controls.axis_error.isVisible()
    assert window.state.geometry is before[0]
    assert window.state.angle_deg == before[1]
    assert controls.angle_range_label.text() == before[2]
    assert_range(window, 180)
