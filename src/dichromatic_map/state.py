"""Qt-free session data and launch parameters.

This is the authoritative owner of physical geometry, display filtering,
selections and numerical results. Worker handles live in ComputeSession.
Changing display state must not change physical coordinates or cell counts.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .crystal import (
    get_geometry, csl_presets, matching_csl_preset, misorientation_range, ProjectedGrain,
)
from .matching import LocalPairs
from .appearance import DEFAULT_GRAIN_COLORS, default_layer_symbols

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


@dataclass(frozen=True)
class SelectedAtom:
    position: np.ndarray
    grain_index: int
    layer: int
    half_indices: np.ndarray


@dataclass(frozen=True)
class CellVertex:
    position: np.ndarray  # unrotated model coordinates
    layer: int
    source: str  # exact CSL or local near-pair midpoint
    grain_positions: np.ndarray | None = None  # (G1, G2) actual endpoints


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

    def validate(self) -> None:
        geometry = get_geometry(self.lattice, self.axis)
        limit = misorientation_range(geometry.axis, geometry.lattice).maximum_deg
        if self.angle_deg is not None and not 0.0 <= self.angle_deg <= limit:
            raise ValueError(
                f"angle_deg must be between 0 and {limit:g} degrees "
                f"for {geometry.lattice} {geometry.axis_label}"
            )
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


class PatternState:
    """A viewer's structure, view, selections and cached analysis results."""

    def __init__(self, parameters: PatternParameters):
        parameters.validate()
        self.parameters = parameters
        self.geometry = get_geometry(parameters.lattice, parameters.axis)
        self.angle_range = misorientation_range(self.geometry.axis, self.geometry.lattice)
        self.presets = csl_presets(self.geometry.axis)
        initial_angle = parameters.angle_deg
        if initial_angle is None:
            initial_angle = default_angle_deg(self.geometry.axis)
        initial_preset = matching_csl_preset(initial_angle, axis=self.geometry.axis)
        self.angle_deg = (
            initial_preset.angle_deg if initial_preset is not None else initial_angle
        )
        self.interaction_mode = "idle"
        self.display_rotation_deg = 0.0
        self.show_reference_axes = True
        self.grain_colors = list(DEFAULT_GRAIN_COLORS)
        self.layer_symbols = default_layer_symbols(self.geometry.layer_count)
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
        self.coincident_points = tuple(
            np.empty((0, 2)) for _ in range(self.geometry.layer_count)
        )
        self.visible_atom_counts = [0, 0]
        self.visible_coincidence_counts = [0] * self.geometry.layer_count
        # Filter every grain/phase independently. The legacy visible_layers
        # intersection is still used for overlays that require both grains.
        self.selected_layer = -1
        all_layers = set(range(self.geometry.layer_count))
        self.visible_grain_layers = [set(all_layers), set(all_layers)]
        self.visible_layers = set(all_layers)
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
