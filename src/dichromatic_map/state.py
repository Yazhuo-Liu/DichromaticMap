"""Qt-free session data and launch parameters.

This is the authoritative owner of physical geometry, display filtering,
selections and numerical results. Worker handles live in ComputeSession.
Changing display state must not change physical coordinates or cell counts.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .crystal import (
    angle_presets, bicrystal, csl_presets, matching_angle_preset, misorientation_range,
    ProjectedGrain,
)
from .matching import LocalPairs
from .appearance import DEFAULT_GRAIN_COLORS, default_layer_size_scales, default_layer_symbols

BUFFER_FACTOR = 2.0
VIEW_SCALE_MIN = 0.1
VIEW_SCALE_MAX = 5.0
VIEW_SCALE_STOPS = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0)
CSL_PRESETS = csl_presets("110")
DEFAULT_ANGLE_DEG = CSL_PRESETS[3].angle_deg


def default_angle_deg(axis: str) -> float:
    presets = csl_presets(axis)
    sigma = {"110": 9, "100": 5}.get(axis)
    if sigma is not None:
        return min(p.angle_deg for p in presets if p.sigma == sigma)
    return (
        min(presets, key=lambda p: (p.sigma, p.angle_deg)).angle_deg if presets else 0.0
    )


def default_crystal_angle_deg(crystal) -> float:
    """Homophase CSL default; otherwise the first orientation relationship, or 0°."""
    crystal = bicrystal(crystal)
    if crystal.homophase:
        return default_angle_deg(crystal.axes[0])
    presets = angle_presets(crystal)
    return presets[0].angle_deg if presets else 0.0


@dataclass(frozen=True)
class SelectedAtom:
    position: np.ndarray
    grain_index: int
    layer: int
    half_indices: np.ndarray


@dataclass(frozen=True)
class CellVertex:
    position: np.ndarray  # unrotated model coordinates
    layer: int  # G1 layer; also G2's for a homophase bicrystal
    source: str  # CSL, local near pair, symmetry, or parallelogram closure
    grain_positions: np.ndarray | None = None  # (G1, G2) actual endpoints
    second_layer: int | None = None  # G2 layer of two different grains

    @property
    def layers(self) -> tuple[int, int]:
        """The picked atom's layer in each grain."""
        return (self.layer, self.layer if self.second_layer is None else self.second_layer)


@dataclass(frozen=True)
class PatternParameters:
    angle_deg: float | None = None
    lattice_constant: float = 3.52
    width: float = 12.0  # units of a0, not Angstrom
    height: float = 9.0
    marker_size: float = 32.0
    view_scale: float = 1.0
    lattice: str = "FCC"
    axis: str = "110"
    # Grain 2; None repeats grain 1. Lengths remain in grain 1's a0, so
    # grain 2's lattice constant enters only as the ratio a0(G2)/a0(G1).
    lattice2: str | None = None
    axis2: str | None = None
    lattice_ratio: float = 1.0

    @property
    def crystal(self):
        return bicrystal(
            (self.lattice, self.lattice if self.lattice2 is None else self.lattice2),
            (self.axis, self.axis if self.axis2 is None else self.axis2),
            self.lattice_ratio,
        )

    def validate(self) -> None:
        if not np.isfinite(self.lattice_constant) or self.lattice_constant <= 0.0:
            raise ValueError("lattice_constant must be finite and positive")
        crystal = self.crystal
        limit = misorientation_range(crystal.axis, crystal.lattice, scale=crystal.scale).maximum_deg
        if self.angle_deg is not None and not 0.0 <= self.angle_deg <= limit:
            geometry = crystal.geometries[0]
            name = (f"{geometry.lattice} {geometry.axis_label}" if crystal.homophase
                    else crystal.label)
            raise ValueError(
                f"angle_deg must be between 0 and {limit:g} degrees for {name}"
            )
        if (not np.isfinite(self.width) or not np.isfinite(self.height)
                or self.width <= 0.0 or self.height <= 0.0):
            raise ValueError("width and height must be finite and positive")
        if not np.isfinite(self.marker_size) or self.marker_size <= 0.0:
            raise ValueError("marker_size must be finite and positive")
        if not VIEW_SCALE_MIN <= self.view_scale <= VIEW_SCALE_MAX:
            raise ValueError(
                f"view_scale must be between {VIEW_SCALE_MIN} and {VIEW_SCALE_MAX}"
            )


class PatternState:
    """A viewer's structure, view, selections and cached analysis results."""

    def __init__(self, parameters: PatternParameters):
        parameters.validate()
        self.parameters = parameters
        self.apply_crystal(parameters.crystal)
        initial_angle = parameters.angle_deg
        if initial_angle is None:
            initial_angle = default_crystal_angle_deg(self.crystal)
        initial_preset = matching_angle_preset(initial_angle, lattice=self.crystal)
        self.angle_deg = (
            initial_preset.angle_deg if initial_preset is not None else initial_angle
        )
        self.interaction_mode = "idle"
        self.display_rotation_deg = 0.0
        self.show_reference_axes = True
        self.grain_colors = list(DEFAULT_GRAIN_COLORS)
        self.layer_symbols = default_layer_symbols(self.layer_count)
        self.layer_size_scales = default_layer_size_scales(self.layer_count)
        self.manual_vertices = []
        self.manual_counts = None
        self.manual_count_error = None
        self.manual_count_key = None
        self.manual_local_cutoff = None
        self.manual_strain_fit = None
        self.manual_unstrained_vertices = None
        self.manual_unstrained_cutoff = None
        self.axial_repeat = 0
        self.grains: list[ProjectedGrain] = []
        self.grain_signature = None
        self.visible_atom_masks: list[np.ndarray] = []
        self.coincident_points, self.coincident_second_layers = self.empty_coincidences()
        self.visible_atom_counts = [0, 0]
        self.visible_coincidence_counts = [0] * self.layer_count
        # Filter every grain/phase independently. The legacy visible_layers
        # intersection is still used for overlays that require both grains.
        self.selected_layer = -1
        self.visible_grain_layers = [set(range(count)) for count in self.crystal.layer_counts]
        self.visible_layers = set.intersection(*self.visible_grain_layers)
        self.selected_points: list[np.ndarray] = []
        self.selected_atoms: list[SelectedAtom] = []
        self.buffer_bounds: tuple[float, float, float, float] | None = None
        self.pending_angle = self.angle_deg
        self.angle_update_active = False
        self.csl_updating = False
        self.render_error = None
        self.near_enabled = False
        self.near_method = "local"
        self.local_pairs = LocalPairs.empty()
        self.local_updating = False
        self.local_error = None
        self.near_solutions = []
        self.near_cell = None
        self.deformations = (np.eye(2), np.eye(2))
        self.translations = np.zeros((2, 2))
        # Ordered manual shifts after the affine deformation/translation.
        # Each grain has its own whole-grain or rectangular-region history.
        self.grain_displacements = [[], []]

    def apply_crystal(self, crystal) -> None:
        """Replace both grains' structure fields; callers reset dependent state."""
        self.crystal = bicrystal(crystal)
        self.geometries = self.crystal.geometries
        # Grain 1 geometry, retained for its frame, labels and homophase layers.
        self.geometry = self.geometries[0]
        self.angle_range = misorientation_range(
            self.crystal.axis, self.crystal.lattice, scale=self.crystal.scale,
        )
        self.presets = angle_presets(self.crystal)

    @property
    def layer_count(self) -> int:
        """Shared layer-ID range of display preferences and coincidence groups."""
        return self.crystal.layer_count

    def empty_coincidences(self):
        """Per-G1-layer coincident sites and their G2 layers, all empty."""
        return (
            tuple(np.empty((0, 2)) for _ in range(self.layer_count)),
            tuple(np.empty(0, dtype=np.int16) for _ in range(self.layer_count)),
        )
