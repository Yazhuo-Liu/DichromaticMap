"""Desktop workflows for two grains with different lattices, axes and a0."""

import numpy as np
import pytest

from dichromatic_map.crystal import bicrystal

pytestmark = pytest.mark.gui

FE = dict(lattice="FCC", axis="110", lattice2="BCC", axis2="111", lattice_ratio=2.87 / 3.59)


def test_grain_two_controls_start_as_a_copy_of_grain_one(gui):
    window = gui.window()
    controls = window.controls
    assert not controls.heterophase_check.isChecked()
    assert controls.structure2_combo.isHidden() and controls.lattice_ratio_spin.isHidden()
    before = window.state
    positions = before.grains[1].positions.copy()
    controls.heterophase_check.setChecked(True)
    gui.settle(window)
    # Checking only reveals G2's controls; the bicrystal and its analyses stay.
    assert window.state is before and window.state.crystal.homophase
    np.testing.assert_array_equal(window.state.grains[1].positions, positions)
    assert not controls.structure2_combo.isHidden() and not controls.lattice_ratio_spin.isHidden()
    assert controls.structure2_combo.currentData() == "FCC"
    assert controls.axis2_combo.currentData() == "110"
    assert controls.lattice_ratio_spin.value() == 1.0

    controls.structure2_combo.setCurrentIndex(controls.structure2_combo.findData("BCC"))
    gui.settle(window)
    crystal = window.state.crystal
    assert crystal.lattices == ("FCC", "BCC") and crystal.axes == ("110", "110")
    assert crystal.scale == 1.0 and window.state.parameters.lattice2 == "BCC"
    assert controls.preset_label.text() == "Orientation relationship"
    controls.lattice_ratio_spin.setValue(0.7994)
    gui.settle(window)
    assert window.state.crystal.scale == 0.7994
    assert window.state.parameters.lattice_ratio == 0.7994
    assert len(window.state.grains[1].positions) > len(window.state.grains[0].positions)

    controls.axis2_combo.setCurrentIndex(controls.axis2_combo.findData("111"))
    gui.settle(window)
    assert window.state.crystal.layer_counts == (2, 3)
    assert [len(checks) for checks in controls.grain_layer_checks] == [2, 3]
    assert controls.preset_combo.itemText(1).startswith("KS")
    assert controls.preset_combo.currentIndex() == 1, "the KS relationship is the default angle"
    assert window.state.angle_range.maximum_deg == 30
    assert set(np.unique(window.state.grains[1].layers)) == {0, 1, 2}

    controls.heterophase_check.setChecked(False)
    gui.settle(window)
    assert window.state.crystal.homophase
    assert window.state.parameters.lattice2 is None
    assert window.state.parameters.lattice_ratio == 1.0
    assert controls.preset_label.text() == "CSL preset"


def test_projected_matching_and_layer_filters_use_each_grains_layers(gui):
    window = gui.window(**FE)
    controls = window.controls
    assert window.state.crystal == bicrystal(("FCC", "BCC"), ("110", "111"), 2.87 / 3.59)
    assert controls.heterophase_check.isChecked()
    assert controls.lattice_ratio_spin.value() == pytest.approx(0.799443)
    # Only the shared origin coincides for an incommensurate a0 ratio.
    assert sum(len(points) for points in window.state.coincident_points) == 1
    controls.near_button.click()
    gui.settle(window)
    pairs = window.state.local_pairs
    assert len(pairs.first) and set(np.unique(pairs.second_layers)) == {0, 1, 2}
    visible = window._local_pair_mask().sum()
    controls.grain_layer_checks[1][2].click()  # Hide BCC layer C only.
    gui.settle(window)
    mask = window._local_pair_mask()
    assert 0 < mask.sum() < visible
    assert not np.any(pairs.second_layers[mask] == 2)
    assert np.all(np.isin(pairs.layers[mask], [0, 1]))


def test_two_grain_manual_cell_counts_and_session_roundtrip(gui, tmp_path):
    window = gui.window(**FE)
    controls = window.controls
    controls.near_button.click()
    gui.settle(window)
    pairs = window.state.local_pairs
    # Four pairs of one (G1, G2) layer family around a near-coincident cell.
    family = np.flatnonzero((pairs.layers == 0) & (pairs.second_layers == 0))
    assert len(family) >= 4
    candidates = window._common_site_candidates()
    chosen = [candidate for candidate in candidates if candidate.layers == (0, 0)]
    origin = min(chosen, key=lambda vertex: np.linalg.norm(vertex.position))
    others = sorted(chosen, key=lambda vertex: np.linalg.norm(vertex.position - origin.position))
    a = next(v for v in others[1:] if np.linalg.norm(v.position - origin.position) > 1)
    b = next(v for v in others[1:] if abs(np.cross(a.position - origin.position,
                                                   v.position - origin.position)) > 1)
    fourth_target = a.position + b.position - origin.position
    c = min(chosen, key=lambda vertex: np.linalg.norm(vertex.position - fourth_target))
    vertices = [origin, a, c, b]
    if np.cross(a.position - origin.position, b.position - origin.position) < 0:
        vertices = [origin, b, c, a]
    window.state.manual_vertices = [window._resolve_cell_vertex(vertex) for vertex in vertices]
    window._queue_manual_count()
    gui.settle(window)
    counts = window.state.manual_counts
    assert counts is not None and counts.interior.shape == (2, 3)
    text = controls.manual_info.toPlainText()
    assert "G1 ○/G2 ○ layers" in text

    output = tmp_path / "fcc-bcc.dmap"
    window.save_session(output)
    restored = gui.window()
    restored.load_session(output)
    gui.settle(restored)
    assert restored.state.crystal == window.state.crystal
    assert restored.controls.heterophase_check.isChecked()
    assert restored.controls.structure2_combo.currentData() == "BCC"
    assert restored.controls.axis2_combo.currentData() == "111"
    assert restored.controls.lattice_ratio_spin.value() == pytest.approx(0.799443)
    assert restored.state.parameters.lattice_ratio == 2.87 / 3.59
    assert [vertex.layers for vertex in restored.state.manual_vertices] == [(0, 0)] * 4
    np.testing.assert_array_equal(restored.state.manual_counts.interior, counts.interior)
