"""Portable, versioned sessions with numerical CSV tables; no Qt or pickle.

A .dmap file is a ZIP archive. Only explicitly listed physical and display
inputs are restored; render buffers, worker state and filesystem paths are not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import tempfile
import zipfile
import zlib

import numpy as np

from .cells import StrainedCell, bases, determinant, validate_cell_vertices
from .crystal import rotation_matrix_2d
from .state import CellVertex, PatternParameters, PatternState, SelectedAtom
from .strain import SelectedCellStrain

FORMAT = "dichromatic-map-session"
SCHEMA_VERSION = 1
MAX_JSON_BYTES = 1_000_000
MAX_MEMBER_BYTES = 8_000_000
MAX_ARCHIVE_BYTES = 32_000_000
_MEMBERS = {"session.json", "counts.csv", "vectors.csv", "strain.csv", "README.txt"}
_SETTINGS = {
    "region_states", "manual_visible", "show_common_cell", "local_distance",
    "strain_percent", "search_index", "manual_strain_percent", "manual_rotation_deg",
}
_STATE_FIELDS = {
    "parameters", "interaction_mode", "display_rotation_deg", "show_reference_axes", "selected_layer",
    "visible_grain_layers", "selected_points", "selected_atoms", "manual_vertices",
    "manual_local_cutoff", "axial_repeat", "near_enabled", "near_method", "near_cell",
    "deformations", "translations", "manual_strain_fit", "manual_unstrained_vertices",
    "manual_unstrained_cutoff",
}


@dataclass(frozen=True)
class SessionSnapshot:
    """Validated, independent state ready for GUI reconstruction."""

    state: PatternState
    settings: dict
    view_range: tuple[float, float, float, float]


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Invalid {name} fields")
    return value


def _number(value, name, low=-1e8, high=1e8):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not low <= value <= high or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and between {low:g} and {high:g}")
    return float(value)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _array(value, shape, name, *, integer=False):
    # Reject strings, booleans, ragged arrays and numeric coercion before casting.
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}") from error
    if (result.shape != shape or result.dtype.kind not in "iuf" or
            any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)):
        raise ValueError(f"{name} must have numeric shape {shape}")
    if not np.all(np.isfinite(result)) or np.any(np.abs(result) > 1e8):
        raise ValueError(f"{name} contains invalid coordinates")
    if integer and np.any(result != np.rint(result)):
        raise ValueError(f"{name} must contain integers")
    return result.astype(np.int64 if integer else float)


def _list(value, maximum, name):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"Invalid {name}")
    return value


def _same(actual, expected, name):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-8):
        raise ValueError(f"Inconsistent {name}")


def _deformations(value):
    matrices = _array(value, (2, 2, 2), "deformations")
    singular = np.linalg.svd(matrices, compute_uv=False)
    if (np.any(np.linalg.det(matrices) <= 0) or
            singular.min() < 1e-3 or singular.max() > 1e3):
        raise ValueError("Deformations must be nonsingular and orientation-preserving")
    return matrices


def _optional_cutoff(value):
    return None if value is None else _number(value, "local cutoff", 0.0001, 0.5)


def _cell_payload(cell):
    return None if cell is None else {
        field: getattr(cell, field).tolist() for field in ("m1", "m2", "f1", "f2", "cell")
    } | {"max_strain": float(cell.max_strain), "lattice": cell.lattice, "axis": cell.axis}


def _read_cell(value, state):
    if value is None:
        return None
    _fields(value, ("m1", "m2", "f1", "f2", "cell", "max_strain", "lattice", "axis"), "cell")
    if value["lattice"] != state.geometry.lattice or value["axis"] != state.geometry.axis:
        raise ValueError("Cell lattice and axis do not match session")
    m1 = _array(value["m1"], (2, 2), "m1", integer=True)
    m2 = _array(value["m2"], (2, 2), "m2", integer=True)
    f1, f2 = _deformations([value["f1"], value["f2"]])
    common = _array(value["cell"], (2, 2), "common cell")
    strain = _number(value["max_strain"], "max_strain", 0, 0.1 + 1e-9)
    if determinant(m1) * determinant(m2) <= 0 or abs(np.linalg.det(common)) < 1e-12:
        raise ValueError("Common cell must have compatible nonzero area")
    for f, basis, matrix in zip((f1, f2), bases(state.angle_deg, state.geometry.lattice,
                                              state.geometry.axis), (m1, m2)):
        _same(f @ basis @ matrix, common, "common-cell translations")
    _same(strain, np.max(np.abs(np.linalg.svd([f1, f2], compute_uv=False) - 1)),
          "maximum strain")
    return StrainedCell(m1, m2, f1, f2, common, strain, value["lattice"], value["axis"])


def _vertex_payload(vertices):
    if vertices is None:
        return None
    return [dict(position=vertex.position.tolist(), layer=int(vertex.layer),
                 source=vertex.source, grain_positions=(None if vertex.grain_positions is None
                                                        else vertex.grain_positions.tolist()))
            for vertex in vertices]


def _read_vertices(value, state, name):
    vertices = []
    for item in _list(value, 4, name):
        _fields(item, ("position", "layer", "source", "grain_positions"), name)
        layer = _integer(item["layer"], "vertex layer", 0, state.geometry.layer_count - 1)
        if item["source"] not in ("CSL", "local"):
            raise ValueError("Invalid vertex source")
        position = _array(item["position"], (2,), "vertex position")
        endpoints = _array(item["grain_positions"], (2, 2), "grain positions")
        _same(position, endpoints.mean(axis=0), "vertex midpoint")
        vertices.append(CellVertex(position, layer, item["source"], endpoints))
    if len({vertex.layer for vertex in vertices}) > 1:
        raise ValueError("Manual vertices must belong to one layer")
    if len(vertices) == 4:
        validate_cell_vertices([vertex.position for vertex in vertices])
        for grain in (0, 1):
            validate_cell_vertices([vertex.position if vertex.grain_positions is None else
                                    vertex.grain_positions[grain] for vertex in vertices])
    return vertices


def _fit_payload(fit):
    return None if fit is None else {
        "cell": _cell_payload(fit.cell), "residual": float(fit.residual),
        **{field: getattr(fit, field).tolist() for field in
           ("translations", "vertices", "origin", "rotations_deg", "stretches")},
    }


def _read_fit(value, state):
    if value is None:
        if state.manual_unstrained_vertices is not None or state.manual_unstrained_cutoff is not None:
            raise ValueError("Original vertices require an applied manual strain fit")
        return None
    _fields(value, ("cell", "residual", "translations", "vertices", "origin",
                    "rotations_deg", "stretches"), "manual strain fit")
    cell = _read_cell(value["cell"], state)
    original = state.manual_unstrained_vertices
    if cell is None or state.near_cell is None or original is None or len(original) != 4:
        raise ValueError("Manual strain fit requires its cell and original vertices")
    if len(state.manual_vertices) != 4 or any(vertex.grain_positions is None for vertex in original):
        raise ValueError("Manual strain fit requires four paired vertices")
    translations = _array(value["translations"], (2, 2), "fit translations")
    vertices = _array(value["vertices"], (2, 4, 2), "fit vertices")
    origin = _array(value["origin"], (2,), "fit origin")
    rotations = _array(value["rotations_deg"], (2,), "fit rotations")
    stretches = _array(value["stretches"], (2, 2), "fit stretches")
    residual = _number(value["residual"], "fit residual", 0, 1e-7)
    for field in ("m1", "m2", "f1", "f2", "cell"):
        _same(getattr(cell, field), getattr(state.near_cell, field), "manual strain cell")
    _same(translations, state.translations, "manual translations")
    before = np.stack([vertex.grain_positions for vertex in original], axis=1)
    _same(vertices, np.einsum("gij,gnj->gni", state.deformations, before) + translations[:, None, :],
          "transformed vertices")
    _same(vertices, np.stack([vertex.grain_positions for vertex in state.manual_vertices], axis=1),
          "selected transformed vertices")
    _same(origin, vertices[:, 0].mean(axis=0), "fit origin")
    _same(residual, np.max(np.linalg.norm(vertices[0] - vertices[1], axis=1)), "fit residual")
    left, singular, right = np.linalg.svd(state.deformations)
    polar = left @ right
    _same(rotations, np.degrees(np.arctan2(polar[:, 1, 0], polar[:, 0, 0])), "polar rotations")
    _same(stretches, singular, "principal stretches")
    return SelectedCellStrain(cell, translations, vertices, origin, residual, rotations, stretches)


def _read_settings(value):
    _fields(value, _SETTINGS, "settings")
    regions = _list(value["region_states"], 4, "region states")
    if len(regions) != 4:
        raise ValueError("Four region visibility values are required")
    return {
        "region_states": tuple(_boolean(item, "region state") for item in regions),
        "manual_visible": _boolean(value["manual_visible"], "manual visibility"),
        "show_common_cell": _boolean(value["show_common_cell"], "common-cell visibility"),
        "local_distance": _number(value["local_distance"], "local distance", 0.0001, 0.5),
        "strain_percent": _number(value["strain_percent"], "strain limit", 0.01, 10),
        "search_index": _integer(value["search_index"], "search index", 2, 40),
        "manual_strain_percent": _number(value["manual_strain_percent"], "manual strain limit", 0.01, 10),
        "manual_rotation_deg": _number(value["manual_rotation_deg"], "manual rotation limit", 0, 5),
    }


def _snapshot_from_payload(payload):
    _fields(payload, ("format", "schema_version", "state", "settings", "view_range"), "session")
    if payload["format"] != FORMAT or type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported DichromaticMap session format or schema version")
    raw = _fields(payload["state"], _STATE_FIELDS, "state")
    parameters = dict(_fields(raw["parameters"], PatternParameters.__dataclass_fields__, "parameters"))
    for field in ("lattice_constant", "width", "height", "marker_size"):
        parameters[field] = _number(parameters[field], field, 1e-8, 1e4)
    parameters["angle_deg"] = _number(parameters["angle_deg"], "angle", 0, 180)
    parameters["view_scale"] = _number(parameters["view_scale"], "view scale", 0.1, 5)
    if not isinstance(parameters["lattice"], str) or not isinstance(parameters["axis"], str):
        raise ValueError("Lattice and axis must be strings")
    state = PatternState(PatternParameters(**parameters))
    # Preserve the actual stored angle, including precision beyond the GUI spinbox.
    state.angle_deg = state.pending_angle = parameters["angle_deg"]
    if raw["interaction_mode"] not in ("idle", "boundary", "vector", "cell"):
        raise ValueError("Unknown interaction mode")
    state.interaction_mode = raw["interaction_mode"]
    state.display_rotation_deg = _number(raw["display_rotation_deg"], "display rotation", -180, 180)
    state.show_reference_axes = _boolean(raw["show_reference_axes"], "reference axes")
    state.selected_layer = _integer(raw["selected_layer"], "selected layer", -1, state.geometry.layer_count - 1)
    groups = _list(raw["visible_grain_layers"], 2, "grain layers")
    if len(groups) != 2:
        raise ValueError("Two grain-layer filters are required")
    state.visible_grain_layers = []
    for group in groups:
        values = [_integer(item, "visible layer", 0, state.geometry.layer_count - 1)
                  for item in _list(group, state.geometry.layer_count, "visible layers")]
        if len(set(values)) != len(values):
            raise ValueError("Duplicate visible layers")
        state.visible_grain_layers.append(set(values))
    state.visible_layers = set.intersection(*state.visible_grain_layers)
    state.deformations = tuple(_deformations(raw["deformations"]))
    state.translations = _array(raw["translations"], (2, 2), "translations")
    state.selected_points = [_array(point, (2,), "GB point") for point in
                             _list(raw["selected_points"], 2, "GB points")]
    for item in _list(raw["selected_atoms"], 2, "selected atoms"):
        _fields(item, ("position", "grain_index", "layer", "half_indices"), "selected atom")
        position = _array(item["position"], (2,), "atom position")
        grain = _integer(item["grain_index"], "grain", 0, 1)
        layer = _integer(item["layer"], "atom layer", 0, state.geometry.layer_count - 1)
        indices = _array(item["half_indices"], (3,), "half indices", integer=True)
        reference = indices - state.geometry.layer_offsets_half_indices[layer]
        coefficients = np.linalg.lstsq(state.geometry.basis_half_indices, reference, rcond=None)[0]
        _same(state.geometry.basis_half_indices @ np.rint(coefficients), reference, "atom lattice indices")
        physical = (state.deformations[grain] @ rotation_matrix_2d((1 - 2 * grain) * state.angle_deg / 2)
                    @ (indices @ state.geometry.frame[:, :2] / 2) + state.translations[grain])
        _same(position, physical, "atom position")
        state.selected_atoms.append(SelectedAtom(position, grain, layer, indices))
    state.manual_vertices = _read_vertices(raw["manual_vertices"], state, "manual vertices")
    state.manual_local_cutoff = _optional_cutoff(raw["manual_local_cutoff"])
    state.axial_repeat = _integer(raw["axial_repeat"], "axial repeat", -4, 4)
    state.near_enabled = _boolean(raw["near_enabled"], "near-CSL enabled")
    if raw["near_method"] not in ("local", "strain"):
        raise ValueError("Unknown near-CSL method")
    state.near_method = raw["near_method"]
    state.near_cell = _read_cell(raw["near_cell"], state)
    if state.near_cell is not None:
        _same(state.deformations, [state.near_cell.f1, state.near_cell.f2], "cell deformations")
        state.near_solutions = [state.near_cell]
    state.manual_unstrained_vertices = (None if raw["manual_unstrained_vertices"] is None else
        _read_vertices(raw["manual_unstrained_vertices"], state, "original manual vertices"))
    state.manual_unstrained_cutoff = _optional_cutoff(raw["manual_unstrained_cutoff"])
    state.manual_strain_fit = _read_fit(raw["manual_strain_fit"], state)
    settings = _read_settings(payload["settings"])
    view_range = tuple(_array(payload["view_range"], (4,), "view range"))
    if view_range[1] - view_range[0] < 1e-8 or view_range[3] - view_range[2] < 1e-8:
        raise ValueError("View range must have positive width and height")
    return SessionSnapshot(state, settings, view_range)


def _state_payload(state):
    parameters = replace(state.parameters, lattice=state.geometry.lattice,
                         axis=state.geometry.axis, angle_deg=float(state.angle_deg))
    return {
        "parameters": asdict(parameters), "interaction_mode": state.interaction_mode,
        "display_rotation_deg": float(state.display_rotation_deg),
        "show_reference_axes": bool(state.show_reference_axes), "selected_layer": int(state.selected_layer),
        "visible_grain_layers": [sorted(map(int, layers)) for layers in state.visible_grain_layers],
        "selected_points": [point.tolist() for point in state.selected_points],
        "selected_atoms": [dict(position=atom.position.tolist(), grain_index=int(atom.grain_index),
                                layer=int(atom.layer), half_indices=atom.half_indices.tolist())
                           for atom in state.selected_atoms],
        "manual_vertices": _vertex_payload(state.manual_vertices),
        "manual_local_cutoff": state.manual_local_cutoff, "axial_repeat": int(state.axial_repeat),
        "near_enabled": bool(state.near_enabled), "near_method": state.near_method,
        "near_cell": _cell_payload(state.near_cell),
        "deformations": np.asarray(state.deformations).tolist(), "translations": state.translations.tolist(),
        "manual_strain_fit": _fit_payload(state.manual_strain_fit),
        "manual_unstrained_vertices": _vertex_payload(state.manual_unstrained_vertices),
        "manual_unstrained_cutoff": state.manual_unstrained_cutoff,
    }


def save_session(path, state, settings, view_range):
    """Atomically save physical inputs and current numerical tables to one ZIP.

    Numerical tables are generated before opening the destination. Any failure
    leaves an existing file intact; no user-machine or worker details are saved.
    """
    from .exports import build_export_tables

    payload = dict(format=FORMAT, schema_version=SCHEMA_VERSION, state=_state_payload(state),
                   settings=dict(settings), view_range=list(view_range))
    # JSON roundtrip normalizes tuples and independently validates exactly what
    # will be written; allow_nan=False rejects nonstandard JSON constants.
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Cannot serialize session: {error}") from error
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("Session metadata is too large")
    _snapshot_from_payload(json.loads(encoded))
    members = {"session.json": encoded}
    tables = build_export_tables(state, settings)
    if set(tables) != _MEMBERS - {"session.json"} or any(not isinstance(text, str) for text in tables.values()):
        raise ValueError("Invalid numerical export tables")
    members.update({name: text.encode("utf-8") for name, text in tables.items()})
    if any(len(data) > MAX_MEMBER_BYTES for data in members.values()):
        raise ValueError("Numerical export is too large")
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def load_session(path):
    """Validate a .dmap archive before returning new state; never extract files.

    Invalid data raises ValueError. Filesystem failures retain their OSError type.
    CSVs are human-readable outputs, not trusted inputs for reconstructing state.
    """
    try:
        with open(path, "rb") as stream:
            if os.fstat(stream.fileno()).st_size > MAX_ARCHIVE_BYTES:
                raise ValueError("Session archive is too large")
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                if len(members) != len(_MEMBERS) or {member.filename for member in members} != _MEMBERS:
                    raise ValueError("Invalid session archive members")
                if any(member.file_size > MAX_MEMBER_BYTES for member in members):
                    raise ValueError("Session archive member is too large")
                metadata = archive.getinfo("session.json")
                if metadata.file_size > MAX_JSON_BYTES:
                    raise ValueError("Session metadata is too large")
                encoded = archive.read(metadata)
        payload = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
        return _snapshot_from_payload(payload)
    except (zipfile.BadZipFile, zlib.error, EOFError, UnicodeError, json.JSONDecodeError, RuntimeError,
            NotImplementedError, OverflowError, TypeError, RecursionError, np.linalg.LinAlgError) as error:
        raise ValueError(f"Invalid DichromaticMap session: {error}") from error
