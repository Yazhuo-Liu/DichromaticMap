"""Offscreen GUI fixtures without pytest-qt or dependencies on core tests."""

import os
import sys
import time
import traceback

import pytest

# Set before importing either GUI dependency, including during collection.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture(scope="session")
def qt_app():
    from dichromatic_map.ui.controls import create_application

    application = create_application()
    application.setQuitOnLastWindowClosed(False)
    yield application
    application.processEvents()


class GuiHarness:
    def __init__(self, application, errors):
        from PySide6 import QtCore, QtTest

        self.app = application
        self.errors = errors
        self.windows = []
        self.QtCore = QtCore
        self.QtTest = QtTest.QTest

    def wait_for(self, window, predicate, timeout=30, *, diagnostics=None):
        deadline = time.monotonic() + timeout
        while True:
            self.app.processEvents(self.QtCore.QEventLoop.AllEvents, 10)
            assert not self.errors, "\n".join(self.errors)
            if predicate():
                return
            if time.monotonic() >= deadline:
                pytest.fail(
                    "GUI timed out:\n"
                    + window.controls.status_label.text()
                    + "\n"
                    + window.controls.manual_info.toPlainText()
                    + "\n"
                    + window.controls.near_info.toPlainText()
                    + ("\n" + diagnostics() if diagnostics is not None else "")
                )
            time.sleep(0.005)

    def settle(self, window, timeout=30):
        def finished():
            state, compute = window.state, window.compute
            return (
                compute.parallel_stage is None
                and not state.csl_updating
                and not state.local_updating
                and compute.manual_count_future is None
                and compute.manual_count_pending is None
                and (compute.near_search is None or not compute.near_search.busy)
                and not any(
                    timer.isActive()
                    for timer in (
                        window.angle_preview_timer,
                        window.coincidence_timer,
                        window.view_refresh_timer,
                        window.near_debounce_timer,
                    )
                )
            )

        self.wait_for(window, finished, timeout)
        assert window.state.render_error is None
        assert window.state.local_error is None
        assert window.state.manual_count_error is None
        if window.compute.near_search is not None:
            assert window.compute.near_search.error is None

    def window(self, workers=1, **parameters):
        from dichromatic_map.state import PatternParameters
        from dichromatic_map.ui.window import DichromaticPatternWindow

        window = DichromaticPatternWindow(
            PatternParameters(**parameters), worker_count=workers
        )
        self.windows.append(window)
        window.show()
        window.activateWindow()
        self.settle(window)
        return window

    def select_layer(self, window, layer, grains=(0, 1)):
        controls = window.controls
        controls.orientation_layers_tabs.setCurrentIndex(1)
        controls.no_layers_button.click()
        for grain in grains:
            controls.grain_layer_checks[grain][layer].click()
        self.app.processEvents()

    def click_plot(self, window, point):
        """Deliver a real viewport click through the scene's picking route."""
        plot = window.plot
        viewport = plot.plot_widget.viewport()
        pixel = None

        def target_in_viewport():
            nonlocal pixel
            coordinates = plot._to_view(point)
            scene_point = plot.view_box.mapViewToScene(
                self.QtCore.QPointF(float(coordinates[0]), float(coordinates[1]))
            )
            pixel = plot.plot_widget.mapFromScene(scene_point)
            return (
                plot.view_box.sceneBoundingRect().contains(scene_point)
                and viewport.rect().contains(pixel)
            )

        # Idle calculation workers do not imply that Qt's queued layout and
        # viewport updates have finished. Map the target after processing them;
        # never pan the view or clamp an out-of-view target to make a click pass.
        self.wait_for(
            window, target_in_viewport, timeout=5,
            diagnostics=lambda: (
                f"Point outside viewport after waiting: {point}; pixel={pixel}; "
                f"viewport={viewport.rect()}; viewRange={plot.view_box.viewRange()}; "
                f"scene={plot.view_box.sceneBoundingRect()}; "
                f"viewportTransform={plot.plot_widget.viewportTransform()}"
            ),
        )
        self.QtTest.mouseClick(
            viewport, self.QtCore.Qt.LeftButton, self.QtCore.Qt.NoModifier, pixel
        )
        self.app.processEvents()
        assert not self.errors, "\n".join(self.errors)

    def key(self, window, key):
        self.QtTest.keyClick(window.plot.plot_widget.viewport(), key)
        self.app.processEvents()
        assert not self.errors, "\n".join(self.errors)

    def close(self):
        # The application intentionally uses nonblocking shutdown. Joining the
        # owned executors here prevents work from escaping a test's lifetime.
        for window in reversed(self.windows):
            compute = window.compute
            executors = {compute.executor, compute.local_thread_executor}
            if compute.near_search is not None and compute.near_search.owns_executor:
                executors.add(compute.near_search.executor)
            for timer in window.findChildren(self.QtCore.QTimer):
                timer.stop()
            window.close()
            for executor in executors - {None}:
                executor.shutdown(wait=True, cancel_futures=True)
            assert compute.executor is None
            assert compute.local_thread_executor is None
            window.deleteLater()
        self.app.processEvents()
        self.QtCore.QCoreApplication.sendPostedEvents(
            None, self.QtCore.QEvent.DeferredDelete
        )


@pytest.fixture
def gui(qt_app):
    errors = []
    old_hook = sys.excepthook
    sys.excepthook = lambda *error: errors.append(
        "".join(traceback.format_exception(*error))
    )
    harness = GuiHarness(qt_app, errors)
    try:
        yield harness
    finally:
        try:
            harness.close()
        finally:
            sys.excepthook = old_hook
        assert not errors, "Unhandled Qt callback exception:\n" + "\n".join(errors)
