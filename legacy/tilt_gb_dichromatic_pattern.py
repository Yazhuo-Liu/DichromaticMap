#!/usr/bin/env python3
"""Interactive FCC <110> tilt grain-boundary dichromatic pattern.

DEPRECATED: retained as the legacy FCC [110] Matplotlib viewer. New crystal,
axis and a0-coordinate features are maintained only in the Qt viewer
``dichromatic_map.ui.window`` (launch with ``python main.py``).

This file is intentionally standalone: it imports no code from GBClaw.  It
projects FCC atomic columns along [110] onto the [-110]-[001] plane and rotates
the two grains by +/- half of the requested misorientation.

Interaction
-----------
1. The default mode is idle.  Drag freely in any direction anywhere in the
   pattern to pan; dragging never cancels a partially selected pair.
2. Press "Pick GB" and click B1/B2.  The line through them becomes the trial
   boundary; its sides are Left and Right relative to B1 -> B2.
3. Press "Vector" and click P1/P2 to draw and label their lattice vector.
4. Use the four checkboxes to show/hide either side of either grain, or use the
   preset buttons for a complete dichromatic pattern and the two bicrystals.
5. Use "Field size" or the mouse wheel to regenerate a smaller/larger field.
   The vector pair reports a lattice vector.  "P2 axial image" chooses
   which periodic copy of P2 is used along [110]; adjacent values differ by
   (a/2)[110], which is invisible in this projection.
6. "Near-CSL" (off by default) opens strain limits and periodic-cell results.
   Its shared solver lives in ``dichromatic_map.strain`` and ``dichromatic_map.matching``.
7. "Show cell" / "Fit cell" also display the unstrained exact CSL periodic cell.

Blue/orange identifies the grain, while circles/diamonds identify the A/B
stacking layers along [110].  Gold circles/diamonds highlight coincidences only
when both the projected coordinate and the stacking layer match.  Common
low-Sigma [110] CSL rotations can be selected directly from the radio buttons.

Only NumPy and Matplotlib are required.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from dichromatic_map.compute import NearSearch
from dichromatic_map.strain import (
    DEFAULT_STRAIN_PERCENT,
    DEFAULT_SEARCH_INDEX,
    tensor_readout,
)
from dichromatic_map.ui import NEAR_COLOR
from dichromatic_map.matching import exact_csl_cell


SIDE_TOLERANCE = 1.0e-9
COINCIDENCE_TOLERANCE_FACTOR = 1.0e-6
VIEW_SCALE_MIN = 0.1
VIEW_SCALE_MAX = 5.0
VIEW_SCALE_STOPS = np.array((0.1, 0.5, 1.0, 2.0, 3.0, 5.0))


def csl_angle_deg(m: int, n: int) -> float:
    """Return the exact cubic [110] CSL angle for quaternion (m, n, n, 0)."""

    return float(np.degrees(2.0 * np.arctan2(np.sqrt(2.0) * n, m)))


@dataclass(frozen=True)
class CSLPreset:
    """A low-Sigma [110] rotation represented by its integer quaternion."""

    sigma: int
    m: int
    n: int

    @property
    def angle_deg(self) -> float:
        return csl_angle_deg(self.m, self.n)

    @property
    def label(self) -> str:
        return f"Σ{self.sigma}   {self.angle_deg:.2f}°"


# Common low-Sigma members in the 0-90 degree [110] fundamental interval.
# The two Sigma-33 rotations are distinct CSL variants.
CSL_PRESETS = (
    CSLPreset(33, 8, 1),
    CSLPreset(19, 6, 1),
    CSLPreset(27, 5, 1),
    CSLPreset(9, 4, 1),
    CSLPreset(11, 3, 1),
    CSLPreset(33, 5, 2),
    CSLPreset(3, 2, 1),
    CSLPreset(17, 3, 2),
)
DEFAULT_ANGLE_DEG = CSL_PRESETS[3].angle_deg


def matching_csl_preset(
    angle_deg: float, tolerance_deg: float = 0.006
) -> CSLPreset | None:
    """Return a displayed CSL preset when *angle_deg* is on its rounded tick."""

    return next(
        (
            preset
            for preset in CSL_PRESETS
            if abs(preset.angle_deg - angle_deg) <= tolerance_deg
        ),
        None,
    )


@dataclass(frozen=True)
class ProjectedGrain:
    """Projected atomic columns and their alternating [110] layer labels."""

    positions: np.ndarray
    layers: np.ndarray  # 0 = layer A, 1 = layer B
    half_indices: np.ndarray  # representative FCC site: (a/2) * [i, j, k]


@dataclass(frozen=True)
class SelectedAtom:
    """An interactively selected projected atomic column."""

    position: np.ndarray
    grain_index: int
    layer: int
    half_indices: np.ndarray


@dataclass(frozen=True)
class PatternParameters:
    """Geometry and drawing parameters for the projected pattern."""

    angle_deg: float = DEFAULT_ANGLE_DEG
    lattice_constant: float = 3.52
    width: float = 40.0
    height: float = 30.0
    marker_size: float = 32.0
    view_scale: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.angle_deg <= 90.0:
            raise ValueError("angle_deg must be between 0 and 90 degrees")
        if self.lattice_constant <= 0.0:
            raise ValueError("lattice_constant must be positive")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("width and height must be positive")
        if self.marker_size <= 0.0:
            raise ValueError("marker_size must be positive")
        if not VIEW_SCALE_MIN <= self.view_scale <= VIEW_SCALE_MAX:
            raise ValueError(
                f"view_scale must be between {VIEW_SCALE_MIN} and {VIEW_SCALE_MAX}"
            )


def rotation_matrix_2d(angle_deg: float) -> np.ndarray:
    """Return the counter-clockwise 2-D rotation matrix for *angle_deg*."""

    angle = np.deg2rad(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def fcc_110_projected_columns(
    lattice_constant: float,
    width: float,
    height: float,
    rotation_deg: float,
    center: tuple[float, float] = (0.0, 0.0),
    deformation: np.ndarray | None = None,
) -> ProjectedGrain:
    """Generate FCC atomic-column positions viewed along [110].

    In units of ``a/2``, FCC lattice sites have integer coordinates whose sum
    is even.  Projection along [110] gives the screen coordinates

        x || [-110] = a (j - i) / (2 sqrt(2))
        y || [001]  = a k / 2.

    Therefore the projected columns are a centered rectangular lattice with
    integer indices ``m, n`` constrained by ``m + n`` even.  Columns with both
    indices even belong to the A layer; columns with both indices odd belong to
    the B layer half a [110] repeat later.  Constructing the columns directly
    avoids arbitrary slab-thickness and duplicate-removal choices in a finite
    3-D projection.
    """

    spacing_x = lattice_constant / (2.0 * np.sqrt(2.0))
    spacing_y = lattice_constant / 2.0
    center_x, center_y = center
    half_width, half_height = 0.5 * width, 0.5 * height
    screen_corners = np.array(
        [
            [center_x - half_width, center_y - half_height],
            [center_x - half_width, center_y + half_height],
            [center_x + half_width, center_y - half_height],
            [center_x + half_width, center_y + half_height],
        ]
    )
    rotation = rotation_matrix_2d(rotation_deg)
    if deformation is not None:
        rotation = deformation @ rotation
    unrotated_corners = screen_corners @ np.linalg.inv(rotation).T
    m_min = int(np.floor(np.min(unrotated_corners[:, 0]) / spacing_x)) - 3
    m_max = int(np.ceil(np.max(unrotated_corners[:, 0]) / spacing_x)) + 3
    n_min = int(np.floor(np.min(unrotated_corners[:, 1]) / spacing_y)) - 3
    n_max = int(np.ceil(np.max(unrotated_corners[:, 1]) / spacing_y)) + 3

    m_values = np.arange(m_min, m_max + 1)
    n_values = np.arange(n_min, n_max + 1)
    m_grid, n_grid = np.meshgrid(m_values, n_values, indexing="xy")
    parity_mask = (m_grid + n_grid) % 2 == 0
    selected_m = m_grid[parity_mask]
    selected_n = n_grid[parity_mask]
    layer_labels = (selected_m % 2).astype(np.int8)
    points = np.column_stack((selected_m * spacing_x, selected_n * spacing_y))
    # Choose t = i + j as 0 for A and 1 for B.  Other atoms in the same
    # projected column differ only by complete [110] repeats.
    axial_layer_index = layer_labels.astype(int)
    half_indices = np.column_stack(
        (
            (axial_layer_index - selected_m) // 2,
            (axial_layer_index + selected_m) // 2,
            selected_n,
        )
    ).astype(int)

    rotated = points @ rotation.T
    crop_mask = (
        (rotated[:, 0] >= center_x - half_width)
        & (rotated[:, 0] <= center_x + half_width)
        & (rotated[:, 1] >= center_y - half_height)
        & (rotated[:, 1] <= center_y + half_height)
    )
    return ProjectedGrain(
        rotated[crop_mask],
        layer_labels[crop_mask],
        half_indices[crop_mask],
    )


def selected_region_mask(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    keep_left: bool,
    keep_right: bool,
) -> np.ndarray:
    """Return the points retained by the two side switches."""

    direction = second - first
    signed_cross_product = direction[0] * (points[:, 1] - first[1]) - direction[1] * (
        points[:, 0] - first[0]
    )
    keep = np.zeros(len(points), dtype=bool)
    if keep_left:
        keep |= signed_cross_product >= -SIDE_TOLERANCE
    if keep_right:
        keep |= signed_cross_product <= SIDE_TOLERANCE
    return keep


def points_on_selected_regions(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    keep_left: bool,
    keep_right: bool,
) -> np.ndarray:
    """Filter points using the directed line ``first -> second``.

    Left and right are relative to the direction from ``first`` to ``second``.
    Points on the boundary belong to both sides, so boundary atoms stay visible
    when either adjacent side is enabled.
    """

    return points[selected_region_mask(points, first, second, keep_left, keep_right)]


def same_layer_coincidence_sites(
    grain_1: ProjectedGrain,
    grain_2: ProjectedGrain,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find projected CSL sites, rejecting apparent A-B column overlaps.

    A projected match is a genuine coincidence only when its two columns have
    the same position *and* the same phase in the ABAB stacking along [110].
    The returned arrays contain A-A and B-B sites, respectively.
    """

    if tolerance <= 0.0:
        raise ValueError("coincidence tolerance must be positive")

    sites_by_layer: list[np.ndarray] = []
    tolerance_squared = tolerance * tolerance
    for layer in (0, 1):
        first = grain_1.positions[grain_1.layers == layer]
        second = grain_2.positions[grain_2.layers == layer]
        if len(first) == 0 or len(second) == 0:
            sites_by_layer.append(np.empty((0, 2)))
            continue

        # Hash into tolerance-sized cells.  Rather than constructing an O(N^2)
        # distance matrix, this stays practical when the field of view is large.
        second_bins: dict[tuple[int, int], list[int]] = {}
        second_keys = np.floor(second / tolerance).astype(np.int64)
        for index, key in enumerate(second_keys):
            second_bins.setdefault((int(key[0]), int(key[1])), []).append(index)

        matches: list[np.ndarray] = []
        first_keys = np.floor(first / tolerance).astype(np.int64)
        for point, key in zip(first, first_keys, strict=True):
            match = None
            for delta_x in (-1, 0, 1):
                for delta_y in (-1, 0, 1):
                    candidates = second_bins.get(
                        (int(key[0]) + delta_x, int(key[1]) + delta_y), ()
                    )
                    for index in candidates:
                        if np.sum((point - second[index]) ** 2) <= tolerance_squared:
                            match = 0.5 * (point + second[index])
                            break
                    if match is not None:
                        break
                if match is not None:
                    break
            if match is not None:
                matches.append(match)

        sites_by_layer.append(
            np.asarray(matches).reshape((-1, 2)) if matches else np.empty((0, 2))
        )

    return sites_by_layer[0], sites_by_layer[1]


class DichromaticPatternApp:
    """Matplotlib user interface for the interactive dichromatic pattern."""

    CHECK_LABELS = (
        "Grain 1 on Left side",
        "Grain 1 on Right side",
        "Grain 2 on Left side",
        "Grain 2 on Right side",
    )

    def __init__(self, parameters: PatternParameters, worker_count: int = 1) -> None:
        parameters.validate()
        self.parameters = parameters
        self.worker_count = max(1, worker_count)
        self.near_enabled = False
        self.near_search = None
        self.near_solutions = []
        self.near_cell = None
        self.near_cell_index = 0
        self.near_figure = None
        self.deformations = (np.eye(2), np.eye(2))
        initial_preset = matching_csl_preset(parameters.angle_deg)
        self.angle_deg = (
            initial_preset.angle_deg
            if initial_preset is not None
            else parameters.angle_deg
        )
        self.view_scale = parameters.view_scale
        self.view_center = np.zeros(2, dtype=float)
        self.axial_repeat = 0
        self.grains: list[ProjectedGrain] = []
        self.coincident_points = (np.empty((0, 2)), np.empty((0, 2)))
        self.visible_atom_counts = [0, 0]
        self.visible_atom_masks: list[np.ndarray] = []
        self.visible_coincidence_counts = [0, 0]
        self.selected_points: list[np.ndarray] = []
        self.selected_atoms: list[SelectedAtom] = []
        self.interaction_mode = "idle"
        self._mouse_press_pixels: np.ndarray | None = None
        self._mouse_press_center = np.zeros(2, dtype=float)
        self._mouse_dragging = False
        self._suspend_check_callback = False
        self._suspend_preset_callback = False

        self.figure, self.axes = plt.subplots(figsize=(12.8, 8.0))
        self.figure.subplots_adjust(left=0.075, right=0.73, bottom=0.22, top=0.91)
        try:
            self.figure.canvas.manager.set_window_title(
                "FCC <110> Tilt GB Dichromatic Pattern"
            )
        except AttributeError:
            pass

        self._configure_pattern_axes()
        # Color identifies the grain; marker shape identifies the ABAB layer.
        self.grain_layer_artists = [
            [
                self.axes.scatter(
                    [],
                    [],
                    s=parameters.marker_size,
                    marker="o",
                    c="#1479d1",
                    edgecolors="#084c8d",
                    linewidths=0.55,
                    alpha=0.75,
                    label="G1 layer A  (+theta/2)",
                    zorder=3,
                ),
                self.axes.scatter(
                    [],
                    [],
                    s=parameters.marker_size * 1.05,
                    marker="D",
                    c="#1479d1",
                    edgecolors="#084c8d",
                    linewidths=0.55,
                    alpha=0.75,
                    label="G1 layer B  (+theta/2)",
                    zorder=3,
                ),
            ],
            [
                self.axes.scatter(
                    [],
                    [],
                    s=parameters.marker_size * 1.12,
                    marker="o",
                    facecolors="none",
                    edgecolors="#e2532d",
                    linewidths=1.25,
                    alpha=0.94,
                    label="G2 layer A  (-theta/2)",
                    zorder=4,
                ),
                self.axes.scatter(
                    [],
                    [],
                    s=parameters.marker_size * 1.18,
                    marker="D",
                    facecolors="none",
                    edgecolors="#e2532d",
                    linewidths=1.25,
                    alpha=0.94,
                    label="G2 layer B  (-theta/2)",
                    zorder=4,
                ),
            ],
        ]
        self.coincidence_artists = [
            self.axes.scatter(
                [],
                [],
                s=parameters.marker_size * 3.0,
                marker="o",
                facecolors="none",
                edgecolors="#f0b429",
                linewidths=2.0,
                label="CSL coincidence A-A",
                zorder=6,
            ),
            self.axes.scatter(
                [],
                [],
                s=parameters.marker_size * 3.15,
                marker="D",
                facecolors="none",
                edgecolors="#f0b429",
                linewidths=2.0,
                label="CSL coincidence B-B",
                zorder=6,
            ),
        ]
        self.selection_artist = self.axes.scatter(
            [],
            [],
            s=parameters.marker_size * 3.0,
            facecolors="none",
            edgecolors="#171717",
            linewidths=1.7,
            zorder=8,
        )
        self.selection_labels = [
            self.axes.text(
                0.0,
                0.0,
                label,
                fontsize=9,
                fontweight="bold",
                color="#171717",
                ha="left",
                va="bottom",
                visible=False,
                zorder=9,
            )
            for label in ("B1", "B2")
        ]
        self.vector_endpoint_artist = self.axes.scatter(
            [],
            [],
            s=parameters.marker_size * 3.2,
            facecolors="none",
            edgecolors="#8d3db0",
            linewidths=2.0,
            zorder=10,
        )
        self.vector_endpoint_labels = [
            self.axes.text(
                0.0,
                0.0,
                label,
                fontsize=9,
                fontweight="bold",
                color="#6f268f",
                ha="left",
                va="bottom",
                visible=False,
                zorder=11,
            )
            for label in ("P1", "P2")
        ]
        self.vector_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.2,
            color="#8d3db0",
            shrinkA=4,
            shrinkB=4,
            visible=False,
            zorder=9,
        )
        self.axes.add_patch(self.vector_arrow)
        self.vector_annotation = self.axes.text(
            0.0,
            0.0,
            "",
            fontsize=8.5,
            color="#5e1d78",
            ha="center",
            va="center",
            visible=False,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#8d3db0"},
            zorder=11,
        )
        (self.boundary_artist,) = self.axes.plot(
            [], [], color="#202020", linewidth=2.0, zorder=7, label="Trial GB"
        )
        self.left_side_label = self.axes.text(
            0.0,
            0.0,
            "L",
            fontsize=15,
            fontweight="bold",
            color="#207244",
            ha="center",
            va="center",
            visible=False,
            zorder=9,
        )
        self.right_side_label = self.axes.text(
            0.0,
            0.0,
            "R",
            fontsize=15,
            fontweight="bold",
            color="#8c3e9d",
            ha="center",
            va="center",
            visible=False,
            zorder=9,
        )

        self._create_controls()
        (self.near_cell_artist,) = self.axes.plot(
            [],
            [],
            "--",
            color=NEAR_COLOR,
            lw=2.2,
            zorder=7,
            label="Common periodic cell",
        )
        self.near_timer = self.figure.canvas.new_timer(interval=40)
        self.near_timer.add_callback(self._poll_near_search)
        self._regenerate_pattern()
        self.axes.legend(loc="upper right", framealpha=0.94, fontsize=8, ncol=2)

        self.figure.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.figure.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("close_event", self._close_near)

    def _configure_pattern_axes(self) -> None:
        self._set_view_limits()
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.set_xlabel(r"$[-110]$ direction ($\mathrm{\AA}$)")
        self.axes.set_ylabel(r"$[001]$ direction ($\mathrm{\AA}$)")
        self.axes.grid(True, color="#d8dde3", linewidth=0.55, alpha=0.65)
        self.axes.set_axisbelow(True)
        for spine in self.axes.spines.values():
            spine.set_visible(True)
            spine.set_color("#8f9aa6")
            spine.set_linewidth(0.9)
        self.axes.text(
            0.015,
            0.985,
            "view / tilt axis: [110]",
            transform=self.axes.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color="#40454a",
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#bbc2c9"},
            zorder=10,
        )

    def _view_dimensions(self) -> tuple[float, float]:
        return (
            self.parameters.width * self.view_scale,
            self.parameters.height * self.view_scale,
        )

    def _set_view_limits(self) -> None:
        width, height = self._view_dimensions()
        self.axes.set_xlim(
            self.view_center[0] - 0.5 * width,
            self.view_center[0] + 0.5 * width,
        )
        self.axes.set_ylim(
            self.view_center[1] - 0.5 * height,
            self.view_center[1] + 0.5 * height,
        )

    def _update_marker_sizes(self) -> None:
        size_factor = max(1.0, self.view_scale**0.8)
        base_size = self.parameters.marker_size / size_factor
        grain_multipliers = ((1.0, 1.05), (1.12, 1.18))
        for artists, multipliers in zip(
            self.grain_layer_artists, grain_multipliers, strict=True
        ):
            for artist, multiplier in zip(artists, multipliers, strict=True):
                artist.set_sizes([max(3.0, base_size * multiplier)])
        for artist, multiplier in zip(
            self.coincidence_artists, (3.0, 3.15), strict=True
        ):
            artist.set_sizes([max(10.0, base_size * multiplier)])
        self.selection_artist.set_sizes([max(18.0, base_size * 3.0)])
        self.vector_endpoint_artist.set_sizes([max(20.0, base_size * 3.2)])

    def _create_controls(self) -> None:
        cell_axes = self.figure.add_axes((0.765, 0.966, 0.125, 0.029))
        self.cell_checks = CheckButtons(cell_axes, ["Show cell"], [True])
        self.cell_checks.labels[0].set_fontsize(8.3)
        self.cell_checks.on_clicked(self._update_common_cell)
        self.cell_fit_button = Button(
            self.figure.add_axes((0.895, 0.966, 0.085, 0.029)), "Fit cell"
        )
        self.cell_fit_button.label.set_fontsize(8.3)
        self.cell_fit_button.on_clicked(self._fit_near_cell)
        near_axes = self.figure.add_axes((0.765, 0.915, 0.215, 0.045))
        self.near_button = Button(near_axes, "Near-CSL: OFF")
        self.near_button.on_clicked(self._toggle_near)
        check_axes = self.figure.add_axes((0.765, 0.69, 0.215, 0.18))
        check_axes.set_title("Visible regions", fontsize=10, loc="left")
        self.region_checks = CheckButtons(
            check_axes, self.CHECK_LABELS, (True, True, True, True)
        )
        self.region_checks.on_clicked(self._on_region_check)

        csl_axes = self.figure.add_axes((0.765, 0.395, 0.215, 0.26))
        csl_axes.set_title("Low-Sigma [110] CSL presets", fontsize=10, loc="left")
        self.csl_labels = ["Custom"] + [preset.label for preset in CSL_PRESETS]
        active_preset = matching_csl_preset(self.angle_deg)
        active_index = (
            self.csl_labels.index(active_preset.label)
            if active_preset is not None
            else 0
        )
        self.csl_radio = RadioButtons(csl_axes, self.csl_labels, active=active_index)
        for label in self.csl_radio.labels:
            label.set_fontsize(8.2)
        self.csl_radio.on_clicked(self._on_csl_preset)

        new_line_axes = self.figure.add_axes((0.765, 0.315, 0.102, 0.052))
        self.new_line_button = Button(new_line_axes, "Pick GB [R]")
        self.new_line_button.on_clicked(self._start_new_boundary)

        vector_axes = self.figure.add_axes((0.878, 0.315, 0.102, 0.052))
        self.vector_button = Button(vector_axes, "Vector [V]")
        self.vector_button.on_clicked(self._start_vector_measurement)

        full_axes = self.figure.add_axes((0.765, 0.245, 0.102, 0.052))
        self.full_button = Button(full_axes, "Full [F]")
        self.full_button.on_clicked(
            lambda _event: self._set_region_states((True, True, True, True))
        )

        center_axes = self.figure.add_axes((0.878, 0.245, 0.102, 0.052))
        self.center_button = Button(center_axes, "Center [C]")
        self.center_button.on_clicked(self._reset_view_center)

        first_bicrystal_axes = self.figure.add_axes((0.765, 0.175, 0.102, 0.052))
        self.first_bicrystal_button = Button(first_bicrystal_axes, "G1:L/G2:R [1]")
        self.first_bicrystal_button.on_clicked(
            lambda _event: self._set_region_states((True, False, False, True))
        )

        second_bicrystal_axes = self.figure.add_axes((0.878, 0.175, 0.102, 0.052))
        self.second_bicrystal_button = Button(second_bicrystal_axes, "G1:R/G2:L [2]")
        self.second_bicrystal_button.on_clicked(
            lambda _event: self._set_region_states((False, True, True, False))
        )
        for button in (
            self.new_line_button,
            self.vector_button,
            self.full_button,
            self.center_button,
            self.first_bicrystal_button,
            self.second_bicrystal_button,
        ):
            button.label.set_fontsize(8.3)

        status_axes = self.figure.add_axes((0.765, 0.015, 0.215, 0.14))
        status_axes.axis("off")
        self.status_text = status_axes.text(
            0.0,
            1.0,
            "",
            transform=status_axes.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
            linespacing=1.2,
            wrap=True,
        )

        angle_axes = self.figure.add_axes((0.12, 0.12, 0.55, 0.028))
        self.angle_slider = Slider(
            angle_axes,
            "Misorientation",
            0.0,
            90.0,
            valinit=self.angle_deg,
            valstep=0.01,
            valfmt="%1.2f deg",
        )
        self.angle_slider.on_changed(self._on_angle_changed)

        view_axes = self.figure.add_axes((0.12, 0.075, 0.55, 0.028))
        self.view_slider = Slider(
            view_axes,
            "Field size",
            VIEW_SCALE_MIN,
            VIEW_SCALE_MAX,
            valinit=self.view_scale,
            valstep=0.1,
            valfmt="%1.2fx",
        )
        view_axes.set_xticks(VIEW_SCALE_STOPS)
        view_axes.set_xticklabels(
            [f"{scale:g}×" for scale in VIEW_SCALE_STOPS], fontsize=7
        )
        view_axes.tick_params(axis="x", length=3, pad=2, colors="#5e6975")
        self.view_slider.on_changed(self._on_view_scale_changed)

        axial_axes = self.figure.add_axes((0.12, 0.03, 0.55, 0.028))
        self.axial_slider = Slider(
            axial_axes,
            "P2 axial image",
            -4,
            4,
            valinit=self.axial_repeat,
            valstep=1,
            valfmt="%+1.0f",
        )
        self.axial_slider.on_changed(self._on_axial_repeat_changed)

    def _regenerate_pattern(self) -> None:
        half_angle = 0.5 * self.angle_deg
        width, height = self._view_dimensions()
        self._set_view_limits()
        self._update_marker_sizes()
        self.grains = [
            fcc_110_projected_columns(
                self.parameters.lattice_constant,
                width,
                height,
                +half_angle,
                center=tuple(self.view_center),
                deformation=self.deformations[0],
            ),
            fcc_110_projected_columns(
                self.parameters.lattice_constant,
                width,
                height,
                -half_angle,
                center=tuple(self.view_center),
                deformation=self.deformations[1],
            ),
        ]
        coincidence_tolerance = (
            COINCIDENCE_TOLERANCE_FACTOR * self.parameters.lattice_constant
        )
        self.coincident_points = same_layer_coincidence_sites(
            self.grains[0], self.grains[1], coincidence_tolerance
        )
        preset = matching_csl_preset(self.angle_deg)
        csl_text = f", {preset.label.split()[0]}" if preset is not None else ""
        if self.near_cell is not None:
            csl_text = f", strained near-CSL {100*self.near_cell.max_strain:.3f}%"
        self.axes.set_title(
            rf"FCC $\langle 110\rangle$ tilt dichromatic pattern"
            rf"   ($\theta={self.angle_deg:.2f}^\circ${csl_text})"
        )
        self._update_common_cell()
        self._update_visible_points()

    @property
    def common_cell(self):
        return (
            self.near_cell
            if self.near_cell is not None
            else exact_csl_cell(self.angle_deg)
        )

    def _update_common_cell(self, *_args):
        cell = self.common_cell
        self.cell_fit_button.set_active(cell is not None)
        self.cell_fit_button.label.set_color("black" if cell is not None else "#999999")
        if cell is None:
            self.near_cell_artist.set_data([], [])
        else:
            corners = (
                np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
                @ cell.cell.T
                * self.parameters.lattice_constant
            )
            self.near_cell_artist.set_data(corners[:, 0], corners[:, 1])
        self.near_cell_artist.set_visible(
            cell is not None and self.cell_checks.get_status()[0]
        )
        self.figure.canvas.draw_idle()

    def _clear_near_cell(self):
        if self.near_cell is not None:
            self._start_new_boundary()
            self._start_vector_measurement()
            self.interaction_mode = "idle"
        self.near_cell = None
        self.near_solutions = []
        self.deformations = (np.eye(2), np.eye(2))
        self._update_common_cell()

    def _create_near_controls(self):
        self.near_figure = plt.figure(
            "Near-CSL · strain and periodic cell", figsize=(8.5, 7.5)
        )
        fig = self.near_figure
        self.near_info = fig.text(
            0.05, 0.69, "", va="top", family="monospace", fontsize=9
        )
        self.strain_slider = Slider(
            fig.add_axes((0.25, 0.90, 0.62, 0.025)),
            "Max strain (%)",
            0.01,
            10.0,
            valinit=DEFAULT_STRAIN_PERCENT,
            valstep=0.01,
        )
        self.search_index_slider = Slider(
            fig.add_axes((0.25, 0.84, 0.62, 0.025)),
            "Search index",
            2,
            40,
            valinit=DEFAULT_SEARCH_INDEX,
            valstep=1,
        )
        self.near_next_button = Button(
            fig.add_axes((0.05, 0.75, 0.27, 0.05)), "Next common cell"
        )
        self.near_next_button.on_clicked(self._next_near_cell)
        self.near_fit_button = Button(
            fig.add_axes((0.37, 0.75, 0.27, 0.05)), "Fit common cell"
        )
        self.near_fit_button.on_clicked(self._fit_near_cell)
        self.strain_slider.on_changed(self._start_near_search)
        self.search_index_slider.on_changed(self._start_near_search)
        fig.canvas.mpl_connect("close_event", self._near_controls_closed)
        fig.show()

    def _near_controls_closed(self, _event):
        if self.near_enabled:
            self._toggle_near()
        self.near_figure = None

    def _toggle_near(self, _event=None):
        self.near_enabled = not self.near_enabled
        self.near_button.label.set_text(
            "Near-CSL: ON" if self.near_enabled else "Near-CSL: OFF"
        )
        if self.near_enabled:
            if self.near_figure is None:
                self._create_near_controls()
            self._start_near_search()
        else:
            self.near_timer.stop()
            if self.near_search is not None:
                self.near_search.cancel()
            self._clear_near_cell()
            self._regenerate_pattern()
            if self.near_figure is not None:
                self.near_info.set_text("Off. Original unstrained lattices restored.")
                self.near_figure.canvas.draw_idle()
        self.figure.canvas.draw_idle()

    def _start_near_search(self, _value=None):
        if not self.near_enabled:
            return
        self._clear_near_cell()
        self._regenerate_pattern()
        if self.near_search is None:
            self.near_search = NearSearch(self.worker_count)
        self.near_search.request(
            self.angle_deg, self.strain_slider.val, int(self.search_index_slider.val)
        )
        self.near_info.set_text(
            f"Searching actual strained periodic cells…\n{self.worker_count} workers"
        )
        self.near_figure.canvas.draw_idle()
        self.near_timer.start()

    def _poll_near_search(self):
        if not self.near_enabled:
            return
        result = self.near_search.poll()
        if self.near_search.error:
            self.near_timer.stop()
            self.near_info.set_text("Search failed: " + self.near_search.error)
        elif result is not None:
            self.near_timer.stop()
            self.near_solutions = result
            self.near_cell_index = 0
            if result:
                self._apply_near_cell()
            else:
                self.near_info.set_text(
                    "No cell found within this bounded search.\n"
                    "Increase strain limit or search index. Unstrained lattices displayed."
                )
        else:
            self.near_info.set_text(
                f"Searching strained periodic cells…\n"
                f"{self.near_search.completed}/{self.near_search.total} chunks complete"
            )
        self.near_figure.canvas.draw_idle()

    def _next_near_cell(self, _event=None):
        if self.near_enabled and self.near_solutions:
            self.near_cell_index = (self.near_cell_index + 1) % len(self.near_solutions)
            self._apply_near_cell()

    def _apply_near_cell(self):
        self._start_new_boundary()
        self._start_vector_measurement()
        self.interaction_mode = "idle"
        self.near_cell = self.near_solutions[self.near_cell_index]
        self.deformations = (self.near_cell.f1, self.near_cell.f2)
        self.near_info.set_text(
            f"Candidate {self.near_cell_index+1}/{len(self.near_solutions)}\n"
            + tensor_readout(self.near_cell, self.angle_deg)
        )
        self._regenerate_pattern()
        self.near_figure.canvas.draw_idle()

    def _close_near(self, _event=None):
        self.near_timer.stop()
        self.near_enabled = False
        if self.near_search is not None:
            self.near_search.close()
        if self.near_figure is not None:
            plt.close(self.near_figure)

    def _fit_near_cell(self, _event=None):
        """Fit whichever periodic cell is displayed (exact CSL or Near-CSL)."""
        cell = self.common_cell
        if cell is None:
            return
        corners = (
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            @ cell.cell.T
            * self.parameters.lattice_constant
        )
        self.view_center[:] = (corners.min(axis=0) + corners.max(axis=0)) / 2
        scale = 1.15 * np.max(
            np.ptp(corners, axis=0) / [self.parameters.width, self.parameters.height]
        )
        self.view_slider.set_val(float(np.clip(scale, VIEW_SCALE_MIN, VIEW_SCALE_MAX)))

    def _update_visible_points(self) -> None:
        states = self.region_checks.get_status()
        boundary_ready = len(self.selected_points) == 2
        self.visible_atom_masks = []
        for grain_index, grain in enumerate(self.grains):
            if boundary_ready:
                visible_mask = selected_region_mask(
                    grain.positions,
                    self.selected_points[0],
                    self.selected_points[1],
                    states[2 * grain_index],
                    states[2 * grain_index + 1],
                )
            else:
                # Before a line exists, the complete pattern remains available
                # for choosing either endpoint even if a preset was pressed.
                visible_mask = np.ones(len(grain.positions), dtype=bool)
            self.visible_atom_masks.append(visible_mask)
            self.visible_atom_counts[grain_index] = int(np.count_nonzero(visible_mask))
            for layer, artist in enumerate(self.grain_layer_artists[grain_index]):
                layer_mask = grain.layers == layer
                artist.set_offsets(grain.positions[visible_mask & layer_mask])

        for layer, (points, artist) in enumerate(
            zip(self.coincident_points, self.coincidence_artists, strict=True)
        ):
            if boundary_ready:
                visible_in_grain_1 = selected_region_mask(
                    points,
                    self.selected_points[0],
                    self.selected_points[1],
                    states[0],
                    states[1],
                )
                visible_in_grain_2 = selected_region_mask(
                    points,
                    self.selected_points[0],
                    self.selected_points[1],
                    states[2],
                    states[3],
                )
                visible_points = points[visible_in_grain_1 & visible_in_grain_2]
            else:
                visible_points = points
            artist.set_offsets(visible_points)
            self.visible_coincidence_counts[layer] = len(visible_points)
        self._update_status()
        self.figure.canvas.draw_idle()

    @staticmethod
    def _atom_name(atom: SelectedAtom) -> str:
        layer_name = "A" if atom.layer == 0 else "B"
        return f"G{atom.grain_index + 1}-{layer_name}"

    @staticmethod
    def _format_lattice_vector(half_indices: np.ndarray) -> str:
        nonzero = np.abs(half_indices[half_indices != 0])
        common_factor = int(np.gcd.reduce(nonzero)) if len(nonzero) else 1
        direction = half_indices // common_factor
        if common_factor == 1:
            scale = "a/2"
        elif common_factor == 2:
            scale = "a"
        elif common_factor % 2 == 0:
            scale = f"{common_factor // 2}a"
        else:
            scale = f"{common_factor}a/2"
        indices = " ".join(str(int(component)) for component in direction)
        return f"{scale}[{indices}]"

    def _selected_vector_readout(self) -> str:
        if len(self.selected_atoms) != 2:
            return ""

        first, second = self.selected_atoms
        projected = (
            second.position - first.position
        ) / self.parameters.lattice_constant
        projected_text = (
            "proj/a([-110],[001]) = " f"({projected[0]:.4f}, {projected[1]:.4f})"
        )

        if first.grain_index != second.grain_index:
            return f"Cross-grain: no unique [hkl]\n{projected_text}"

        delta_half_indices = second.half_indices - first.half_indices
        delta_half_indices = delta_half_indices + self.axial_repeat * np.array(
            [1, 1, 0]
        )
        vector = self._format_lattice_vector(delta_half_indices)
        magnitude_over_a = 0.5 * float(np.linalg.norm(delta_half_indices))
        strained = ""
        if self.near_cell is not None:
            delta = delta_half_indices
            actual = np.sqrt(
                np.dot(projected, projected)
                + ((delta[0] + delta[1]) / (2 * np.sqrt(2))) ** 2
            )
            strained = f"\nStrained |Delta r|/a = {actual:.4f} (reference indices)"
        return (
            f"Miller vector: {vector}\n"
            f"G{first.grain_index + 1}, axial {self.axial_repeat:+d}; "
            f"ref |Delta r|/a = {magnitude_over_a:.4f}" + strained
        )

    def _update_status(self, message: str | None = None) -> None:
        if message is None:
            if self.interaction_mode == "boundary":
                message = (
                    "GB mode: click B1."
                    if not self.selected_points
                    else "B1 selected; click B2."
                )
            elif self.interaction_mode == "vector":
                message = (
                    "Vector mode: click P1."
                    if not self.selected_atoms
                    else "P1 selected; click P2."
                )
            else:
                message = "Idle: choose Pick GB or Vector."
        atom_counts = " / ".join(str(count) for count in self.visible_atom_counts)
        coincidence_counts = " / ".join(
            str(count) for count in self.visible_coincidence_counts
        )
        self.status_text.set_text(
            f"{message}\nDrag freely to pan (selection kept).\n"
            f"Visible G1 / G2: {atom_counts}\n"
            f"CSL A-A / B-B: {coincidence_counts}; view {self.view_scale:.2f}x"
        )

    def _toolbar_mode_active(self) -> bool:
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        return bool(toolbar is not None and getattr(toolbar, "mode", ""))

    def _on_mouse_press(self, event) -> None:
        if (
            event.inaxes is not self.axes
            or event.button != MouseButton.LEFT
            or event.x is None
            or event.y is None
            or self._toolbar_mode_active()
        ):
            return
        self._mouse_press_pixels = np.array([event.x, event.y], dtype=float)
        self._mouse_press_center = self.view_center.copy()
        self._mouse_dragging = False

    def _on_mouse_motion(self, event) -> None:
        if self._mouse_press_pixels is None or event.x is None or event.y is None:
            return
        displacement = np.array([event.x, event.y]) - self._mouse_press_pixels
        if np.linalg.norm(displacement) <= 4.0 and not self._mouse_dragging:
            return

        self._mouse_dragging = True
        width, height = self._view_dimensions()
        data_per_pixel = np.array(
            [
                width / max(float(self.axes.bbox.width), 1.0),
                height / max(float(self.axes.bbox.height), 1.0),
            ]
        )
        self.view_center[:] = self._mouse_press_center - displacement * data_per_pixel
        self._set_view_limits()
        if len(self.selected_points) == 2:
            self._draw_boundary()
        if len(self.selected_atoms) == 2:
            self._draw_vector()
        self.figure.canvas.draw_idle()

    def _on_mouse_release(self, event) -> None:
        if self._mouse_press_pixels is None:
            return
        was_dragging = self._mouse_dragging
        self._mouse_press_pixels = None
        self._mouse_dragging = False
        if was_dragging:
            self._regenerate_pattern()
            if len(self.selected_points) == 2:
                self._draw_boundary()
            if len(self.selected_atoms) == 2:
                self._draw_vector()
            self.figure.canvas.draw_idle()
            return
        if (
            event.inaxes is self.axes
            and event.button == MouseButton.LEFT
            and event.x is not None
            and event.y is not None
        ):
            self._handle_atom_pick(event.x, event.y)

    def _handle_atom_pick(self, mouse_x: float, mouse_y: float) -> None:
        if self.interaction_mode == "idle":
            return
        atom, pixel_distance = self._nearest_atom(mouse_x, mouse_y)
        if atom is None or pixel_distance > 12.0:
            self._update_status("No atom nearby; current selection is preserved.")
            self.figure.canvas.draw_idle()
            return
        if self.interaction_mode == "boundary":
            self._pick_boundary_atom(atom)
        else:
            self._pick_vector_atom(atom)

    def _pick_boundary_atom(self, atom: SelectedAtom) -> None:
        if (
            self.selected_points
            and np.linalg.norm(atom.position - self.selected_points[0]) < 1e-8
        ):
            self._update_status("Choose a different second atom.")
            self.figure.canvas.draw_idle()
            return

        self.selected_points.append(atom.position.copy())
        self.selection_artist.set_offsets(np.asarray(self.selected_points))
        label = self.selection_labels[len(self.selected_points) - 1]
        label.set_text(
            f"B{len(self.selected_points)} G{atom.grain_index + 1}-"
            f"{'A' if atom.layer == 0 else 'B'}"
        )
        label.set_position(atom.position)
        label.set_visible(True)
        if len(self.selected_points) == 2:
            self.interaction_mode = "idle"
            self._draw_boundary()
            self._update_visible_points()
        else:
            self._update_status()
            self.figure.canvas.draw_idle()

    def _pick_vector_atom(self, atom: SelectedAtom) -> None:
        if (
            self.selected_atoms
            and np.linalg.norm(atom.position - self.selected_atoms[0].position) < 1e-8
        ):
            self._update_status("Choose a different second atom.")
            self.figure.canvas.draw_idle()
            return

        self.selected_atoms.append(atom)
        positions = np.asarray([item.position for item in self.selected_atoms])
        self.vector_endpoint_artist.set_offsets(positions)
        label = self.vector_endpoint_labels[len(self.selected_atoms) - 1]
        label.set_text(f"P{len(self.selected_atoms)} {self._atom_name(atom)}")
        label.set_position(atom.position)
        label.set_visible(True)
        if len(self.selected_atoms) == 2:
            self.interaction_mode = "idle"
            self._draw_vector()
        self._update_status()
        self.figure.canvas.draw_idle()

    def _nearest_atom(
        self, mouse_x_pixels: float, mouse_y_pixels: float
    ) -> tuple[SelectedAtom | None, float]:
        best_atom = None
        best_squared_distance = np.inf
        mouse = np.array([mouse_x_pixels, mouse_y_pixels])
        for grain_index, (grain, visible_mask) in enumerate(
            zip(self.grains, self.visible_atom_masks, strict=True)
        ):
            candidate_indices = np.flatnonzero(visible_mask)
            if not len(candidate_indices):
                continue
            display_points = self.axes.transData.transform(
                grain.positions[candidate_indices]
            )
            squared_distances = np.sum((display_points - mouse) ** 2, axis=1)
            local_index = int(np.argmin(squared_distances))
            if squared_distances[local_index] < best_squared_distance:
                index = int(candidate_indices[local_index])
                best_squared_distance = float(squared_distances[local_index])
                best_atom = SelectedAtom(
                    position=grain.positions[index].copy(),
                    grain_index=grain_index,
                    layer=int(grain.layers[index]),
                    half_indices=grain.half_indices[index].copy(),
                )
        return best_atom, float(np.sqrt(best_squared_distance))

    def _draw_boundary(self) -> None:
        first, second = self.selected_points
        intersections = self._line_box_intersections(first, second)
        if len(intersections) >= 2:
            self.boundary_artist.set_data(
                [intersections[0][0], intersections[1][0]],
                [intersections[0][1], intersections[1][1]],
            )

        direction = second - first
        normal = np.array([-direction[1], direction[0]])
        normal /= np.linalg.norm(normal)
        offset = 0.10 * min(self._view_dimensions())
        midpoint = 0.5 * (first + second)
        label_a = self._clamp_to_view(midpoint + normal * offset)
        label_b = self._clamp_to_view(midpoint - normal * offset)
        self.left_side_label.set_position(label_a)
        self.right_side_label.set_position(label_b)
        self.left_side_label.set_visible(True)
        self.right_side_label.set_visible(True)

    def _draw_vector(self) -> None:
        if len(self.selected_atoms) != 2:
            return
        first = self.selected_atoms[0].position
        second = self.selected_atoms[1].position
        self.vector_arrow.set_positions(first, second)
        self.vector_arrow.set_visible(True)

        direction = second - first
        normal = np.array([-direction[1], direction[0]])
        normal_length = np.linalg.norm(normal)
        if normal_length > 0.0:
            normal /= normal_length
        offset = 0.12 * min(self._view_dimensions())
        text_position = self._clamp_to_view(0.5 * (first + second) + normal * offset)
        self.vector_annotation.set_position(text_position)
        self.vector_annotation.set_text(self._selected_vector_readout())
        self.vector_annotation.set_visible(True)

    def _line_box_intersections(
        self, first: np.ndarray, second: np.ndarray
    ) -> list[np.ndarray]:
        """Return the two intersections of an infinite line and the plot box."""

        direction = second - first
        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        candidates: list[np.ndarray] = []

        if abs(direction[0]) > 1e-12:
            for x_value in (x_min, x_max):
                t_value = (x_value - first[0]) / direction[0]
                y_value = first[1] + t_value * direction[1]
                if y_min - 1e-9 <= y_value <= y_max + 1e-9:
                    candidates.append(np.array([x_value, y_value]))
        if abs(direction[1]) > 1e-12:
            for y_value in (y_min, y_max):
                t_value = (y_value - first[1]) / direction[1]
                x_value = first[0] + t_value * direction[0]
                if x_min - 1e-9 <= x_value <= x_max + 1e-9:
                    candidates.append(np.array([x_value, y_value]))

        unique: list[np.ndarray] = []
        for candidate in candidates:
            if not any(np.linalg.norm(candidate - old) < 1e-7 for old in unique):
                unique.append(candidate)
        return unique[:2]

    def _clamp_to_view(self, point: np.ndarray) -> np.ndarray:
        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        x_margin = 0.035 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)
        return np.array(
            [
                np.clip(point[0], x_min + x_margin, x_max - x_margin),
                np.clip(point[1], y_min + y_margin, y_max - y_margin),
            ]
        )

    def _start_new_boundary(self, _event=None) -> None:
        self.interaction_mode = "boundary"
        self.selected_points.clear()
        self.selection_artist.set_offsets(np.empty((0, 2)))
        for label in self.selection_labels:
            label.set_visible(False)
        self.boundary_artist.set_data([], [])
        self.left_side_label.set_visible(False)
        self.right_side_label.set_visible(False)
        self._update_visible_points()

    def _start_vector_measurement(self, _event=None) -> None:
        self.interaction_mode = "vector"
        self.selected_atoms.clear()
        self.vector_endpoint_artist.set_offsets(np.empty((0, 2)))
        for label in self.vector_endpoint_labels:
            label.set_visible(False)
        self.vector_arrow.set_visible(False)
        self.vector_annotation.set_visible(False)
        self._update_status()
        self.figure.canvas.draw_idle()

    def _reset_view_center(self, _event=None) -> None:
        self.view_center[:] = 0.0
        self._regenerate_pattern()
        if len(self.selected_points) == 2:
            self._draw_boundary()
        if len(self.selected_atoms) == 2:
            self._draw_vector()
        self.figure.canvas.draw_idle()

    def _on_region_check(self, _label: str) -> None:
        if not self._suspend_check_callback:
            self._update_visible_points()

    def _on_csl_preset(self, label: str) -> None:
        if self._suspend_preset_callback or label == "Custom":
            return
        preset = next(preset for preset in CSL_PRESETS if preset.label == label)
        self.angle_slider.set_val(preset.angle_deg)

    def _show_matching_csl_selection(self, preset: CSLPreset | None) -> None:
        desired_label = preset.label if preset is not None else "Custom"
        if self.csl_radio.value_selected == desired_label:
            return
        self._suspend_preset_callback = True
        try:
            self.csl_radio.set_active(self.csl_labels.index(desired_label))
        finally:
            self._suspend_preset_callback = False

    def _set_region_states(self, desired_states: tuple[bool, bool, bool, bool]) -> None:
        self._suspend_check_callback = True
        try:
            current_states = self.region_checks.get_status()
            for index, (current, desired) in enumerate(
                zip(current_states, desired_states, strict=True)
            ):
                if current != desired:
                    self.region_checks.set_active(index)
        finally:
            self._suspend_check_callback = False
        self._update_visible_points()

    def _on_angle_changed(self, angle_deg: float) -> None:
        if self.near_enabled:
            if self.near_search is not None:
                self.near_search.cancel()
            self._clear_near_cell()
        preset = matching_csl_preset(float(angle_deg))
        # A slider position displayed to 0.01 degrees is snapped internally to
        # the exact quaternion angle, preserving exact CSL coincidences.
        self.angle_deg = preset.angle_deg if preset is not None else float(angle_deg)
        self._show_matching_csl_selection(preset)
        # A changed lattice invalidates the exact atoms used by the old line.
        self.interaction_mode = "idle"
        self.selected_points.clear()
        self.selected_atoms.clear()
        self.selection_artist.set_offsets(np.empty((0, 2)))
        self.vector_endpoint_artist.set_offsets(np.empty((0, 2)))
        for label in self.selection_labels + self.vector_endpoint_labels:
            label.set_visible(False)
        self.boundary_artist.set_data([], [])
        self.left_side_label.set_visible(False)
        self.right_side_label.set_visible(False)
        self.vector_arrow.set_visible(False)
        self.vector_annotation.set_visible(False)
        self._regenerate_pattern()
        if self.near_enabled:
            self._start_near_search()

    def _on_view_scale_changed(self, view_scale: float) -> None:
        self.view_scale = float(view_scale)
        self._regenerate_pattern()
        if len(self.selected_points) == 2:
            self._draw_boundary()
        if len(self.selected_atoms) == 2:
            self._draw_vector()
        self.figure.canvas.draw_idle()

    def _on_axial_repeat_changed(self, axial_repeat: float) -> None:
        self.axial_repeat = int(round(axial_repeat))
        if len(self.selected_atoms) == 2:
            self._draw_vector()
        self._update_status()
        self.figure.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.axes or event.step == 0:
            return
        if self._toolbar_mode_active():
            return
        if event.step > 0:
            candidates = VIEW_SCALE_STOPS[VIEW_SCALE_STOPS < self.view_scale - 1.0e-9]
            proposed = candidates[-1] if len(candidates) else VIEW_SCALE_MIN
        else:
            candidates = VIEW_SCALE_STOPS[VIEW_SCALE_STOPS > self.view_scale + 1.0e-9]
            proposed = candidates[0] if len(candidates) else VIEW_SCALE_MAX
        self.view_slider.set_val(proposed)

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "r":
            self._start_new_boundary()
        elif key == "v":
            self._start_vector_measurement()
        elif key == "c":
            self._reset_view_center()
        elif key == "f":
            self._set_region_states((True, True, True, True))
        elif key == "1":
            self._set_region_states((True, False, False, True))
        elif key == "2":
            self._set_region_states((False, True, True, False))

    def show(self) -> None:
        plt.show()

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(output_path, dpi=180, bbox_inches="tight")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive standalone FCC <110> tilt-GB dichromatic pattern"
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=DEFAULT_ANGLE_DEG,
        help="misorientation angle in degrees (default: Sigma 9, 38.94)",
    )
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=3.52,
        help="FCC lattice constant in angstrom (default: %(default)s)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=40.0,
        help="plot width in angstrom (default: %(default)s)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=30.0,
        help="plot height in angstrom (default: %(default)s)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=32.0,
        help="atom marker area in points squared (default: %(default)s)",
    )
    parser.add_argument(
        "--view-scale",
        type=float,
        default=1.0,
        help="initial field-size multiplier, 0.1 to 5 (default: %(default)s)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        metavar="IMAGE",
        help="save the initial full-pattern view and exit (PNG, PDF, SVG, ...)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Near-CSL background workers (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    multiprocessing.freeze_support()
    arguments = parse_arguments()
    parameters = PatternParameters(
        angle_deg=arguments.angle,
        lattice_constant=arguments.lattice_constant,
        width=arguments.width,
        height=arguments.height,
        marker_size=arguments.marker_size,
        view_scale=arguments.view_scale,
    )
    app = DichromaticPatternApp(parameters, worker_count=arguments.workers)
    if arguments.save is not None:
        app.save(arguments.save)
        print(f"Saved dichromatic pattern to {arguments.save}")
    else:
        app.show()


if __name__ == "__main__":
    main()
