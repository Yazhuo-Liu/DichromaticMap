"""Cubic geometry, axial phases, projections and crystal vector coordinates.

Lengths use a0; half_indices use a0/2. Importing this module needs only NumPy.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from math import gcd, lcm
import re
import numpy as np

MAX_LAYERS = 256
MAX_PROJECTED_COLUMNS = 250_000


class GeometryLimitError(ValueError):
    """A requested lattice/view exceeds a documented interactive resource limit."""


def parse_axis(axis):
    """Normalize integer [h k l]; custom multi-digit/signed axes use separators."""
    if isinstance(axis, str):
        clean = axis.strip().strip("[]<>()⟨⟩").strip()
        if re.fullmatch(r"\d{3}", clean):
            values = tuple(int(x) for x in clean)
        else:
            tokens = re.split(r"[\s,;]+", clean)
            if len(tokens) != 3 or not all(
                re.fullmatch(r"[+-]?\d+", x) for x in tokens
            ):
                raise ValueError("Enter three integer indices, e.g. 1 1 2 or 1 -1 3")
            values = tuple(int(x) for x in tokens)
    else:
        try:
            values = tuple(axis)
            if len(values) != 3 or any(int(x) != x for x in values):
                raise ValueError
            values = tuple(int(x) for x in values)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Tilt axis must contain three integer indices") from None
    factor = gcd(*values)
    if not factor:
        raise ValueError("The zero vector is not a tilt axis")
    values = tuple(x // factor for x in values)
    if max(map(abs, values)) > 64:
        raise GeometryLimitError("Reduced axis indices must be between -64 and 64")
    return values


def axis_key(axis):
    values = parse_axis(axis)
    if values in ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2)):
        return "".join(map(str, values))
    return " ".join(map(str, values))


def layer_name(index):
    return chr(65 + index) if index < 26 else f"L{index+1}"


def extended_gcd(a, b):
    """g >= 0 and s*a+t*b=g, using integer arithmetic."""
    old_r, r, old_s, s, old_t, t = int(a), int(b), 1, 0, 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    sign = 1 if old_r >= 0 else -1
    return sign * old_r, sign * old_s, sign * old_t


def plane_integer_basis(row):
    """Unimodular U with row@U=(gcd(row),0,0)."""
    a, b, c = map(int, row)
    g, s, t = extended_gcd(a, b)
    first = (
        np.array([[s, -b // g, 0], [t, a // g, 0], [0, 0, 1]], dtype=np.int64)
        if g
        else np.eye(3, dtype=np.int64)
    )
    final_g, s, t = extended_gcd(g, c)
    second = np.array(
        [[s, 0, -c // final_g], [0, 1, 0], [t, 0, g // final_g]], dtype=np.int64
    )
    return final_g, first @ second


def congruence_kernel(matrix, denominator):
    """Basis of {v in Z^2: matrix@v == 0 modulo denominator}."""
    kernel = np.eye(2, dtype=object)
    for row in np.asarray(matrix, dtype=object):
        a, b = map(int, row @ kernel)
        g, s, t = extended_gcd(a, b)
        if not g:
            continue
        u = np.array([[s, -b // g], [t, a // g]], dtype=object)
        u[:, 0] *= denominator // gcd(denominator, g)
        kernel = kernel @ u
    return np.asarray(kernel, dtype=np.int64)


@dataclass(frozen=True)
class CrystalGeometry:
    lattice: str
    axis: str
    frame: np.ndarray
    basis_half_indices: np.ndarray
    layer_offsets_half_indices: np.ndarray
    axial_repeat_half_indices: np.ndarray
    planar_basis: np.ndarray
    axial_period: float
    axis_norm_squared: int
    x_label: str
    y_label: str
    axis_label: str
    axis_indices: np.ndarray
    layer_spacing: float

    @property
    def layer_count(self):
        return len(self.layer_offsets_half_indices)


def _readonly(values, dtype=float) -> np.ndarray:
    array = np.array(values, dtype=dtype)
    array.setflags(write=False)
    return array


def get_geometry(lattice: str = "FCC", axis: str = "110") -> CrystalGeometry:
    """Compute the primitive plane and every axial phase of a cubic lattice."""

    lattice = str(lattice).strip().upper()
    axis = axis_key(axis)
    return _geometry(lattice, axis)


@lru_cache(maxsize=64)
def _geometry(lattice: str, axis: str) -> CrystalGeometry:
    if lattice not in ("FCC", "BCC"):
        raise ValueError("lattice must be FCC or BCC")
    direction = np.array(parse_axis(axis), dtype=np.int64)
    axis_norm_squared = int(direction @ direction)
    unit = direction / np.sqrt(axis_norm_squared)
    reference = [0, 0, 1] if np.any(direction[:2]) else [0, 1, 0]
    ex = np.cross(reference, unit)
    ex /= np.linalg.norm(ex)
    frame = _readonly(np.column_stack((ex, np.cross(unit, ex), unit)))
    # Primitive cubic Bravais translations in units a0/2.
    primitive = (
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64)
        if lattice == "FCC"
        else np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=np.int64)
    )
    step, unimodular = plane_integer_basis(direction @ primitive)
    axial_factor = (
        (1 if int(direction.sum()) % 2 == 0 else 2)
        if lattice == "FCC"
        else (1 if np.all(direction % 2 == direction[0] % 2) else 2)
    )
    axial = _readonly(direction * axial_factor, int)
    count = axial_factor * axis_norm_squared // step
    if count > MAX_LAYERS:
        raise GeometryLimitError(
            f"This axis requires {count} layers; interactive limit is {MAX_LAYERS}. Use a lower-index axis."
        )
    basis = primitive @ unimodular[:, 1:]
    for _ in range(64):
        if np.dot(basis[:, 1], basis[:, 1]) < np.dot(basis[:, 0], basis[:, 0]):
            basis = basis[:, ::-1].copy()
        q = int(
            np.rint(np.dot(basis[:, 0], basis[:, 1]) / np.dot(basis[:, 0], basis[:, 0]))
        )
        if not q:
            break
        basis[:, 1] -= q * basis[:, 0]
    if np.linalg.det(frame[:, :2].T @ basis) < 0:
        basis[:, 1] *= -1
    planar_basis = _readonly(frame[:, :2].T @ basis / 2.0)
    offsets = np.arange(count)[:, None] * (primitive @ unimodular[:, 0])
    in_plane = offsets @ frame[:, :2] / 2
    integer_shifts = np.floor(in_plane @ np.linalg.inv(planar_basis).T + 1e-10).astype(
        np.int64
    )
    offsets = _readonly(offsets - integer_shifts @ basis.T, int)
    basis = _readonly(basis, int)
    axial_period = float(np.linalg.norm(axial) / 2.0)
    return CrystalGeometry(
        lattice,
        axis,
        frame,
        basis,
        offsets,
        axial,
        planar_basis,
        axial_period,
        axis_norm_squared,
        "x",
        "y",
        "[" + " ".join(map(str, direction)) + "]",
        _readonly(direction, int),
        step / (2 * np.sqrt(axis_norm_squared)),
    )


@dataclass(frozen=True)
class ProjectedGrain:
    positions: np.ndarray
    layers: np.ndarray
    half_indices: np.ndarray
    layer_count: int = 2


@dataclass(frozen=True)
class CrystalVectorCoordinates:
    """Two representations of the SAME current spatial vector, in units a0.

    Current components use orthonormal cubic axes transported by the polar
    rotation. Lattice coordinates use the actual (possibly sheared/stretched)
    conventional basis; for two atoms in one grain these retain reference
    lattice indices. Noninteger components are not integer Miller indices.
    """

    current: np.ndarray
    lattice: np.ndarray
    current_frame: np.ndarray
    lattice_basis: np.ndarray
    strained: bool


def crystal_vector_coordinates(
    displacement, rotation_deg, deformation=None, lattice="FCC", axis="110"
):
    """Express an actual analysis-space (x,y,z)/a0 displacement in one grain.

    Q maps reference cubic axes to analysis axes. The full current basis is
    B=F3 Q, with polar-transported orthonormal axes O=polar(F3) Q. Therefore
    current=O.T @ d and lattice=solve(B,d). The input already includes BOTH
    endpoints' deformations and translations; do not deform d a second time.
    This concerns direct-lattice directions [uvw], not reciprocal plane (hkl).
    """
    displacement = np.asarray(displacement, dtype=float)
    if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
        raise ValueError("Displacement must contain three finite components / a0")
    if not np.isfinite(rotation_deg):
        raise ValueError("Grain rotation must be finite")
    f = np.eye(2) if deformation is None else np.asarray(deformation, dtype=float)
    if f.shape != (2, 2) or not np.all(np.isfinite(f)):
        raise ValueError("Deformation must be a finite 2-by-2 matrix")
    f3 = np.eye(3)
    f3[:2, :2] = f
    left, stretches, right = np.linalg.svd(f3)
    if np.linalg.det(f3) <= 0 or np.min(stretches) <= 1e-12:
        raise ValueError("Deformation must be nonsingular and orientation-preserving")
    angle = np.deg2rad(rotation_deg)
    r = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    reference_frame = r @ get_geometry(lattice, axis).frame.T
    current_frame = left @ right @ reference_frame
    basis = f3 @ reference_frame
    return CrystalVectorCoordinates(
        current_frame.T @ displacement,
        np.linalg.solve(basis, displacement),
        current_frame,
        basis,
        bool(np.max(np.abs(stretches - 1)) > 1e-10),
    )


def format_direction_components(components, unit="a₀"):
    """Keep simple rational vectors exact; never invent high-index directions.

    Preserve magnitude, e.g. a0/2[1 1 2]. Use a rational only if ALL components
    match within 1e-10, with a common denominator <=48 and indices <=256.
    Otherwise print approximate real components, not rounded Miller integers.
    unit='' denotes coefficients in the current deformed conventional basis.
    """
    components = np.asarray(components, dtype=float)
    if components.shape != (3,) or not np.all(np.isfinite(components)):
        raise ValueError("Direction must contain three finite components")
    fractions = [Fraction(float(x)).limit_denominator(48) for x in components]
    denominator = lcm(*(x.denominator for x in fractions))
    if denominator <= 48 and all(
        abs(float(x) - value) <= 1e-10 for x, value in zip(fractions, components)
    ):
        integers = [x.numerator * (denominator // x.denominator) for x in fractions]
        common = gcd(*integers) or 1
        direction = [x // common for x in integers]
        if max(map(abs, direction)) <= 256:
            factor = Fraction(common, denominator)
            scale = (str(factor.numerator) if factor.numerator != 1 else "") + unit
            if factor.denominator != 1:
                scale = (scale or "1") + f"/{factor.denominator}"
            indices = " ".join(map(str, direction))
            return f"{scale}[{indices}]"
    values = " ".join("0" if abs(x) < 1e-12 else f"{x:.6g}" for x in components)
    return f"≈ {unit}[{values}]"


def projected_columns(
    width: float,
    height: float,
    rotation_deg: float,
    center: tuple[float, float] = (0.0, 0.0),
    deformation: np.ndarray | None = None,
    lattice: str = "FCC",
    axis: str = "110",
    translation: np.ndarray | None = None,
) -> ProjectedGrain:
    """Generate only the columns intersecting an a0-scaled screen rectangle.

    A primitive planar mesh is inverse-cropped separately for each stacking
    phase.  Work scales with the projected area, not a three-dimensional volume.
    ``deformation`` acts in screen coordinates after the grain rotation.
    ``half_indices`` remain reference-crystal indices even after deformation.
    ``translation`` is a uniform post-deformation shift in analysis x,y / a0.
    """

    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("width and height must be finite and positive")
    if not np.isfinite(rotation_deg):
        raise ValueError("rotation_deg must be finite")
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain two finite coordinates")
    translation = (
        np.zeros(2) if translation is None else np.asarray(translation, dtype=float)
    )
    if translation.shape != (2,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain two finite coordinates")

    geometry = get_geometry(lattice, axis)
    radians = np.deg2rad(rotation_deg)
    cosine, sine = np.cos(radians), np.sin(radians)
    transform = np.array([[cosine, -sine], [sine, cosine]])
    if deformation is not None:
        deformation = np.asarray(deformation, dtype=float)
        if deformation.shape != (2, 2) or not np.all(np.isfinite(deformation)):
            raise ValueError("deformation must be a finite 2-by-2 matrix")
        transform = deformation @ transform
    screen_basis = transform @ geometry.planar_basis
    inverse_basis = np.linalg.inv(screen_basis)
    half_size = np.array([width, height], dtype=float) / 2.0
    low, high = center - half_size, center + half_size
    corners = center + np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * half_size

    all_positions = []
    all_layers = []
    all_indices = []
    candidate_count = 0
    # Include points on the viewport edge despite floating-point rotation noise.
    crop_epsilon = 1.0e-10 * max(1.0, float(np.max(np.abs(corners))))
    for layer, offset in enumerate(geometry.layer_offsets_half_indices):
        screen_offset = transform @ (offset @ geometry.frame[:, :2] / 2.0) + translation
        integer_corners = (corners - screen_offset) @ inverse_basis.T
        minima = np.floor(integer_corners.min(axis=0)).astype(int) - 1
        maxima = np.ceil(integer_corners.max(axis=0)).astype(int) + 1
        candidate_count += int(np.prod(maxima - minima + 1))
        if candidate_count > MAX_PROJECTED_COLUMNS:
            raise GeometryLimitError(
                f"View requires over {MAX_PROJECTED_COLUMNS:,} candidate columns per grain. Zoom in or reduce --width/--height."
            )
        # Fill the final coordinate array directly, preserving meshgrid's
        # row-major order without allocating two intermediate dense meshes.
        nx, ny = maxima - minima + 1
        grid = np.empty((ny, nx, 2), dtype=int)
        grid[:, :, 0] = np.arange(minima[0], maxima[0] + 1)
        grid[:, :, 1] = np.arange(minima[1], maxima[1] + 1)[:, None]
        coordinates = grid.reshape(-1, 2)
        positions = coordinates @ screen_basis.T + screen_offset
        keep = (
            (positions[:, 0] >= low[0] - crop_epsilon)
            & (positions[:, 0] <= high[0] + crop_epsilon)
            & (positions[:, 1] >= low[1] - crop_epsilon)
            & (positions[:, 1] <= high[1] + crop_epsilon)
        )
        coordinates = coordinates[keep]
        all_positions.append(positions[keep])
        all_layers.append(np.full(len(coordinates), layer, dtype=np.int16))
        all_indices.append(coordinates @ geometry.basis_half_indices.T + offset)

    return ProjectedGrain(
        np.concatenate(all_positions, axis=0),
        np.concatenate(all_layers, axis=0),
        np.concatenate(all_indices, axis=0),
        geometry.layer_count,
    )


def csl_angle_deg(m: int, n: int, axis: str = "110") -> float:
    """Cubic-axis rotation of quaternion (m, n * axis) in degrees."""

    direction = parse_axis(axis)
    norm_squared = sum(x * x for x in direction)
    return float(np.degrees(2.0 * np.arctan2(np.sqrt(norm_squared) * n, m)))


@dataclass(frozen=True)
class CSLPreset:
    sigma: int
    m: int
    n: int
    variant: str = ""
    axis: str = "110"

    @property
    def angle_deg(self) -> float:
        return csl_angle_deg(self.m, self.n, self.axis)

    @property
    def name(self) -> str:
        return f"Σ{self.sigma}{self.variant}"

    @property
    def label(self) -> str:
        return f"{self.name}  ·  {self.angle_deg:.2f}°"


def csl_presets(axis: str = "110") -> tuple[CSLPreset, ...]:
    return _csl_presets(axis_key(axis))


@lru_cache(maxsize=64)
def _csl_presets(axis: str) -> tuple[CSLPreset, ...]:
    """Selected cubic CSL rotations; complementary angles remain selectable.

    Sigma is the odd part of the primitive integer quaternion norm.  These
    cubic rotation indices apply to both FCC and BCC; the actual planar common
    cell is calculated using the selected lattice's own primitive basis.
    """

    axis = axis_key(axis)
    if axis == "110":
        members = (
            (33, 8, 1, "a"),
            (19, 6, 1, ""),
            (27, 5, 1, ""),
            (9, 4, 1, ""),
            (11, 3, 1, ""),
            (33, 5, 2, "c"),
            (3, 2, 1, ""),
            (17, 3, 2, ""),
        )
    elif axis == "100":
        # No invented a/b/c suffixes: use the angle to distinguish complements.
        members = (
            (25, 7, 1, ""),
            (13, 5, 1, ""),
            (17, 4, 1, ""),
            (5, 3, 1, ""),
            (5, 2, 1, ""),
            (17, 5, 3, ""),
            (13, 3, 2, ""),
            (25, 4, 3, ""),
        )
    else:
        # Generic primitive integer quaternions. Bounded low-Sigma menu only;
        # exact cell recognition is independent of this displayed menu.
        norm_squared = sum(x * x for x in parse_axis(axis))
        options = []
        for m in range(1, 33):
            for n in range(1, 17):
                if gcd(m, n) != 1:
                    continue
                angle = csl_angle_deg(m, n, axis)
                if not 0 < angle <= 90 + 1e-10:
                    continue
                sigma = m * m + norm_squared * n * n
                while sigma % 2 == 0:
                    sigma //= 2
                options.append(CSLPreset(sigma, m, n, axis=axis))
        selected = sorted(options, key=lambda p: (p.sigma, p.angle_deg))[:12]
        return tuple(sorted(selected, key=lambda p: p.angle_deg))
    return tuple(CSLPreset(*member, axis=axis) for member in members)


def matching_csl_preset(
    angle_deg: float, tolerance_deg: float = 0.006, axis: str = "110"
) -> CSLPreset | None:
    return next(
        (
            preset
            for preset in csl_presets(axis)
            if abs(preset.angle_deg - angle_deg) <= tolerance_deg
        ),
        None,
    )


def rotation_matrix_2d(angle_deg: float) -> np.ndarray:
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
    """Legacy physical-unit adapter; the Qt viewer uses projected_columns in a0."""
    grain = projected_columns(
        width / lattice_constant,
        height / lattice_constant,
        rotation_deg,
        center=tuple(np.asarray(center) / lattice_constant),
        deformation=deformation,
        lattice="FCC",
        axis="110",
    )
    return ProjectedGrain(
        grain.positions * lattice_constant, grain.layers, grain.half_indices
    )
