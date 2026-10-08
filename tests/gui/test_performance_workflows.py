"""Keep interaction responsive without publishing stale numerical work."""

import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from dichromatic_map import crystal
from dichromatic_map.ui import window as window_module

pytestmark = pytest.mark.gui


def test_angle_previews_run_in_background_and_coalesce_to_latest(gui, monkeypatch):
    window = gui.window(workers=1)
    release = threading.Event()
    started = threading.Event()
    responsive = threading.Event()
    main_thread = threading.get_ident()
    generated = []
    original_worker = window_module.generate_grain_worker
    original_scatter = window.plot._set_scatter

    def blocked_worker(*args):
        generated.append((args[2], threading.get_ident()))
        started.set()
        assert release.wait(10)
        return original_worker(*args)

    def scatter(*args, **kwargs):
        assert threading.get_ident() == main_thread
        return original_scatter(*args, **kwargs)

    monkeypatch.setattr(window_module, "generate_grain_worker", blocked_worker)
    monkeypatch.setattr(window.plot, "_set_scatter", scatter)
    try:
        window.state.pending_angle = 22
        window._apply_pending_angle()
        assert started.wait(10)
        gui.QtCore.QTimer.singleShot(0, responsive.set)
        gui.app.processEvents()
        assert responsive.is_set()  # Qt runs while generation is blocked.
        for angle in (23, 24):
            window.state.pending_angle = angle
            window._apply_pending_angle()
        assert window.compute.parallel_stage == "waiting"
        window._finish_angle_update()
        release.set()
        gui.settle(window)
        assert window.state.angle_deg == 24
        assert window.state.grain_signature == window._geometry_signature()
        assert all(thread != main_thread for _angle, thread in generated)
        assert not any(abs(angle) == 11.5 for angle, _thread in generated)
        for grain, sign in zip(window.state.grains, (1, -1)):
            planar = (0.5 * grain.half_indices) @ window.state.geometry.frame[:, :2]
            expected = planar @ crystal.rotation_matrix_2d(sign * 12).T
            np.testing.assert_allclose(grain.positions, expected, atol=1e-12)
    finally:
        release.set()


def test_exact_overlay_completion_preserves_existing_grain_scatters(gui, monkeypatch):
    window = gui.window(workers=1)
    gui.select_layer(window, 0)
    items = [item for grain in window.plot.grain_layer_items for item in grain]
    points = [np.column_stack(item.getData()).copy() for item in items]

    def rebuild(*_args, **_kwargs):
        pytest.fail("Exact CSL completion must not rebuild unchanged grain scatters")

    for item in items:
        monkeypatch.setattr(item, "setData", rebuild)
    window._start_parallel_coincidences()
    gui.settle(window)
    assert not window.state.csl_updating
    assert len(window.plot.coincidence_items[0].getData()[0]) > 0
    for item, expected in zip(items, points):
        np.testing.assert_array_equal(np.column_stack(item.getData()), expected)


def test_matching_warmup_is_requested_only_after_first_result_is_published(gui, monkeypatch):
    window = gui.window(workers=4)
    calls = []

    def warmup(point_count, layer_count):
        assert window.compute.parallel_stage is None
        assert not window.state.csl_updating
        assert len(window.state.coincident_points[0]) > 0
        calls.append((point_count, layer_count))

    monkeypatch.setattr(window.compute, "warm_matching_pool", warmup)
    window._start_parallel_coincidences()
    assert calls == []
    gui.settle(window)
    assert calls == [(
        sum(len(grain.positions) for grain in window.state.grains),
        window.state.geometry.layer_count,
    )]


def test_navigation_coalesces_counts_and_updates_final_view(gui, monkeypatch):
    window = gui.window(workers=1)
    original = window._refresh_visible_counts
    refreshes = []

    def refresh():
        refreshes.append(window.plot._view_range())
        return original()

    monkeypatch.setattr(window, "_refresh_visible_counts", refresh)
    x0, x1, y0, y1 = window.plot._view_range()
    for index in range(1, 21):
        window.plot.view_box.setRange(
            xRange=(x0 + index * 0.01, x1 + index * 0.01),
            yRange=(y0, y1), padding=0,
        )
    assert refreshes == []
    gui.settle(window)
    assert len(refreshes) == 1
    assert refreshes[0] == window.plot._view_range()
    x0, x1, y0, y1 = refreshes[0]
    for grain, items in enumerate(window.plot.grain_layer_items):
        expected = 0
        for item in items:
            x, y = item.getData()
            expected += np.count_nonzero((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
        assert window.state.visible_atom_counts[grain] == expected


def test_single_layer_visibility_updates_only_affected_grain_and_csl(gui, monkeypatch):
    window = gui.window(lattice="SC", axis="2 3 5")
    plot = window.plot
    affected = {plot.grain_layer_items[0][1], plot.coincidence_items[1]}
    originals = {
        item: np.column_stack(item.getData()).copy()
        for group in (*plot.grain_layer_items, plot.coincidence_items)
        for item in group if item not in affected
    }
    calls = []
    scatter = plot._set_scatter

    def update(item, points):
        assert item in affected
        calls.append(item)
        return scatter(item, points)

    monkeypatch.setattr(plot, "_set_scatter", update)
    masks = [mask.copy() for mask in window.state.visible_atom_masks]
    window.controls.grain_layer_checks[0][1].click()
    assert set(calls) == affected and len(calls) == 2
    assert not np.any(window.state.visible_atom_masks[0][window.state.grains[0].layers == 1])
    np.testing.assert_array_equal(window.state.visible_atom_masks[1], masks[1])
    for item, points in originals.items():
        np.testing.assert_array_equal(np.column_stack(item.getData()), points)
    window.controls.grain_layer_checks[0][1].click()
    np.testing.assert_array_equal(window.state.visible_atom_masks[0], masks[0])


def test_display_rotation_coalesces_to_latest_and_preserves_physical_grains(gui, monkeypatch):
    window = gui.window()
    updates = []
    original = window._apply_display_rotation
    grains = window.state.grains

    def apply(angle):
        updates.append(angle)
        return original(angle)

    monkeypatch.setattr(window, "_apply_display_rotation", apply)
    for angle in (10, 20, 30, 37):
        window.controls.rotation_spin.setValue(angle)
    assert updates == []
    gui.settle(window)
    assert updates == [37]
    assert window.state.display_rotation_deg == 37
    assert all(actual is expected for actual, expected in zip(window.state.grains, grains))
    for grain, items in zip(grains, window.plot.grain_layer_items):
        for layer, item in enumerate(items):
            np.testing.assert_allclose(
                np.column_stack(item.getData()),
                window.plot._to_view(grain.positions[grain.layers == layer]), atol=1e-12,
            )


def test_navigation_within_existing_buffer_reuses_scatter_geometry(gui, monkeypatch):
    window = gui.window()
    grain_signature = window.state.grain_signature

    def rebuild(*_args, **_kwargs):
        pytest.fail("Navigation within the buffered region must reuse scatter data")

    for group in (*window.plot.grain_layer_items, window.plot.coincidence_items):
        for item in group:
            monkeypatch.setattr(item, "setData", rebuild)
    window.plot.view_box.translateBy(x=0.1, y=0.1)
    gui.settle(window)
    assert window.state.grain_signature == grain_signature
