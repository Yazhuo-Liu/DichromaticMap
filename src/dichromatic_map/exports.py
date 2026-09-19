"""Numerical CSV tables for a saved session, independent of Qt and render caches."""

from __future__ import annotations

import csv
import io
from types import SimpleNamespace

import numpy as np

from .cells import count_cell_atoms
from .crystal import crystal_vector_coordinates, rotation_matrix_2d
from .state import PatternState
from .strain import strain_tensors


COUNT_COLUMNS = (
    "grain", "layer", "interior", "boundary", "closed", "half_open",
    "half_open_edges", "half_open_corners", "half_open_available",
    "area_a0_squared", "area_angstrom_squared", "gb_side_filter_applied",
    "gb_keep_left", "gb_keep_right", "gb_start_x_a0", "gb_start_y_a0",
    "gb_end_x_a0", "gb_end_y_a0",
)
VECTOR_COLUMNS = (
    "vector", "p1_grain", "p1_layer", "p2_grain", "p2_layer", "axial_repeat",
    "grain", "frame", "component_unit", "component_1", "component_2",
    "component_3", "length_a0", "length_angstrom",
)
STRAIN_COLUMNS = ("grain", "quantity", "component", "value", "unit", "frame")


def _finite_array(values, shape, name):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return array


def _csv_text(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    raise ValueError("Cannot export a non-finite numerical result")
                value = format(float(value), ".17g")
            values.append(value)
        writer.writerow(values)
    return stream.getvalue()


def _count_rows(state, settings, a0):
    if len(state.manual_vertices) != 4:
        return []
    layers = {vertex.layer for vertex in state.manual_vertices}
    if len(layers) != 1:
        raise ValueError("The selected cell must belong to one axial layer")
    polygons = np.stack([
        _finite_array(vertex.grain_positions, (2, 2), "Cell atom endpoints")
        for vertex in state.manual_vertices
    ], axis=1)
    filtered = bool(settings.get("manual_visible", False))
    boundary = None
    if filtered and len(state.selected_points) == 2:
        boundary = _finite_array(state.selected_points, (2, 2), "GB endpoints")
    region_states = tuple(settings.get("region_states", (True,) * 4))
    if len(region_states) != 4:
        raise ValueError("Provide four GB side visibility states")
    layer = state.manual_vertices[0].layer
    counts = count_cell_atoms(
        polygons, state.angle_deg, state.deformations,
        state.geometry.lattice, state.geometry.axis,
        boundary_points=boundary,
        region_states=region_states if filtered else (True,) * 4,
        layer=layer, translations=state.translations,
    )
    rows = []
    for grain in (0, 1):
        available = bool(counts.half_open_available[grain])
        interior = int(counts.interior[grain, layer])
        boundary_count = int(counts.boundary[grain, layer])
        half_counts = [
            int(values[grain, layer]) if available else ""
            for values in (
                counts.half_open, counts.half_open_edges, counts.half_open_corners,
            )
        ]
        rows.append((
            f"G{grain + 1}", layer, interior, boundary_count,
            interior + boundary_count, *half_counts, int(available),
            counts.areas[grain], counts.areas[grain] * a0 * a0,
            int(filtered and boundary is not None),
            *(tuple(int(value) for value in region_states[2 * grain:2 * grain + 2])
              if boundary is not None else ("", "")),
            *(tuple(boundary.ravel()) if boundary is not None else ("",) * 4),
        ))
    return rows


def _vector_rows(state, a0):
    if len(state.selected_atoms) != 2:
        return []
    first, second = state.selected_atoms
    for atom in (first, second):
        _finite_array(atom.position, (2,), "Selected atom position")
        _finite_array(atom.half_indices, (3,), "Selected atom half-indices")
        if atom.grain_index not in (0, 1):
            raise ValueError("Selected atom grain must be G1 or G2")
    axial_indices = (
        second.half_indices - first.half_indices
        + state.axial_repeat * state.geometry.axial_repeat_half_indices
    )
    dz = 0.5 * axial_indices @ state.geometry.frame[:, 2]
    displacement = np.r_[second.position - first.position, dz]
    view_displacement = np.r_[
        rotation_matrix_2d(state.display_rotation_deg) @ displacement[:2], dz,
    ]
    metadata = (
        "P1_to_P2", f"G{first.grain_index + 1}", first.layer,
        f"G{second.grain_index + 1}", second.layer, state.axial_repeat,
    )
    length = float(np.linalg.norm(displacement))
    representations = [
        ("", "analysis_xyz", "a0", displacement),
        ("", "display_xyz", "a0", view_displacement),
    ]
    grains = (
        (0, 1) if first.grain_index != second.grain_index else (first.grain_index,)
    )
    for grain in grains:
        result = crystal_vector_coordinates(
            displacement, (1 if grain == 0 else -1) * state.angle_deg / 2,
            state.deformations[grain], state.geometry.lattice, state.geometry.axis,
        )
        representations.extend((
            (f"G{grain + 1}", "current_polar_cubic_uvw", "a0", result.current),
            (f"G{grain + 1}", "deformed_cubic_lattice_uvw", "1", result.lattice),
        ))
    return [
        (*metadata, grain, frame, unit, *components, length, length * a0)
        for grain, frame, unit, components in representations
    ]


def _strain_rows(state, a0):
    rows = []
    rotations = []
    # Use the same 3D tensor conversion as strain readouts, but always use
    # current F rather than a possibly stale fitted/search-result cell.
    cell = SimpleNamespace(
        f1=state.deformations[0], f2=state.deformations[1],
        lattice=state.geometry.lattice, axis=state.geometry.axis,
    )
    cubic_tensors = strain_tensors(cell, state.angle_deg)
    for grain, deformation in enumerate(state.deformations):
        name = f"G{grain + 1}"
        left, stretches, right = np.linalg.svd(deformation)
        rotation = left @ right
        stretch = right.T @ np.diag(stretches) @ right
        polar_angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
        rotations.append(polar_angle)
        tensors = (
            ("F", deformation, "xy", "analysis_xy"),
            ("R", rotation, "xy", "analysis_xy"),
            ("U", stretch, "xy", "reference_analysis_xy"),
            ("E", (deformation.T @ deformation - np.eye(2)) / 2,
             "xy", "reference_analysis_xy"),
            ("E", cubic_tensors[grain], "uvw", "reference_cubic_uvw"),
        )
        for quantity, tensor, axes, frame in tensors:
            for i, axis_i in enumerate(axes):
                for j, axis_j in enumerate(axes):
                    rows.append((name, quantity, axis_i + axis_j, tensor[i, j], "1", frame))
        for index, value in enumerate(stretches, start=1):
            rows.extend((
                (name, "principal_stretch", str(index), value, "1", "in_plane_principal"),
                (name, "principal_engineering_strain", str(index), value - 1,
                 "1", "in_plane_principal"),
            ))
        reference_angle = (1 if grain == 0 else -1) * state.angle_deg / 2
        rows.extend((
            (name, "polar_rotation", "", polar_angle, "degree", "analysis_xy"),
            (name, "reference_rotation", "", reference_angle, "degree", "analysis_xy"),
            (name, "current_polar_rotation", "", reference_angle + polar_angle,
             "degree", "analysis_xy"),
        ))
        for axis, value in zip("xy", state.translations[grain]):
            rows.extend((
                (name, "translation", axis, value, "a0", "analysis_xy"),
                (name, "translation", axis, value * a0, "angstrom", "analysis_xy"),
            ))
    rows.extend((
        ("both", "reference_misorientation", "", state.angle_deg,
         "degree", "reference_grains"),
        ("both", "polar_misorientation", "", state.angle_deg + rotations[0] - rotations[1],
         "degree", "current_polar_grains"),
        ("both", "display_rotation", "", state.display_rotation_deg,
         "degree", "display_xy"),
        ("both", "lattice_constant", "", a0, "angstrom", "reference_cubic"),
    ))
    return rows


_README = """DichromaticMap numerical export
==============================
The .dmap session file is a standard ZIP archive. Open it with an archive
manager, or unzip it to access session.json and these UTF-8 CSV tables.
Import the CSV files using Excel's UTF-8 text/CSV import or Python csv/pandas.
Floating-point numbers retain 17 significant digits. They are numerical
results, not rounded display labels. A blank value means not applicable.

session.json is the authoritative saved state used by Import session.
CSV files are numerical snapshots; changing them does not change restoration.
The state records lattice, tilt axis, lattice constant, reference angle,
deformations, translations, selections, and display/analysis settings.
a0 denotes the saved lattice constant in angstrom (angstrom = 1e-10 m).
All x/y analysis coordinates exclude the display-only rotation. Layers are
zero-based axial phase indices, not chemical species.

counts.csv
One row per grain, only for the layer in the complete four-vertex selection.
Counts are recomputed over each grain's actual physical polygon, independently
of the viewport, generated display buffers, and display-layer checkboxes.
interior excludes the boundary; closed = interior + boundary. Half-open means
0 <= u,v < 1 in the parallelogram basis from C1 to C2 and C1 to C4.
half_open_edges/corners count retained unique boundary representatives.
For non-parallelograms half-open fields are blank and half_open_available=0.
A parallelogram alone does not prove crystal periodicity. Areas are the full
polygon areas, even when GB side filtering reduces the counts.
gb_side_filter_applied=1 means the manual GB filter is enabled and two GB
endpoints exist; the saved four region checkboxes then select the retained
sides. The effective keep-left/right flags and directed GB endpoints are also
recorded in the table; these fields are blank when no GB filter applies.
Left/right are defined by the signed cross product from GB start to end.
This does not apply viewport or display-layer filtering.
An incomplete/absent cell yields a header-only table, not invented zero counts.

vectors.csv
Every row describes the same actual P1-to-P2 3D displacement. P2 includes the
signed axial_repeat image, added to its half-indices before the phase-height
difference is computed. The in-plane displacement uses actual atom positions,
including both grains' F and translations; reference indices are not used to
subtract cross-grain x/y positions. analysis_xyz and display_xyz give x,y,z/a0;
the latter rotates x/y only by the display angle. current_polar_cubic_uvw gives
components/a0 in orthonormal cubic axes carried by that grain's polar rotation.
deformed_cubic_lattice_uvw gives dimensionless coefficients in its deformed
conventional cubic basis. These can be nonintegers, especially for cross-grain
vectors; they are not rounded Miller indices. Crystal-frame rows use the
selected grain for same-grain vectors and both grains for cross-grain vectors.
length_a0 and length_angstrom are physical lengths and remain the same across
representations; the Euclidean norm of deformed-lattice coefficients is not
generally a physical length. An incomplete vector yields a header-only table.

strain.csv
Long format: grain, quantity, component, value, unit, frame. Unit 1 means
dimensionless (not percent). F and polar R act in analysis x/y; F = R U, with U
the right stretch. E = (F.T F - I)/2 is Green-Lagrange strain in the reference
analysis frame. The 3D E tensor is transformed to each original grain's cubic
[u,v,w] frame; axial stretch is fixed to one and axial shear to zero in this
in-plane model. principal_stretch is ordered largest first; principal engineering
strain is stretch minus one, not an eigenvalue of E. Translations follow
x_current = F x_reference + t, in analysis coordinates.
reference_misorientation is the saved reference theta. polar_misorientation =
theta + polar_rotation(G1) - polar_rotation(G2) is signed and not symmetry-folded.
Display rotation is separate and has no effect on strain or counts.
Default sessions with no deformation or translation report identity F/R/U and
zero E/strain/translation. Pure rigid rotations have zero strain but nonidentity
F/R; a translation alone also leaves strain zero.
"""


def build_export_tables(state: PatternState, settings: dict) -> dict[str, str]:
    """Return count, vector and strain CSV snapshots without mutating state.

    Invalid numerical results raise ValueError rather than being omitted or
    silently replaced with zeros. Selections that are not yet complete produce
    header-only count/vector tables.
    """
    a0 = float(state.parameters.lattice_constant)
    if not np.isfinite(a0) or a0 <= 0:
        raise ValueError("Lattice constant must be finite and positive")
    for value in (state.angle_deg, state.display_rotation_deg):
        if not np.isfinite(value):
            raise ValueError("Angles must be finite")
    deformations = _finite_array(state.deformations, (2, 2, 2), "Deformations")
    for deformation in deformations:
        if np.linalg.det(deformation) <= 0 or np.linalg.svd(deformation, compute_uv=False).min() <= 1e-12:
            raise ValueError("Deformations must be nonsingular and orientation-preserving")
    _finite_array(state.translations, (2, 2), "Translations")
    return {
        "counts.csv": _csv_text(COUNT_COLUMNS, _count_rows(state, settings, a0)),
        "vectors.csv": _csv_text(VECTOR_COLUMNS, _vector_rows(state, a0)),
        "strain.csv": _csv_text(STRAIN_COLUMNS, _strain_rows(state, a0)),
        "README.txt": _README,
    }
