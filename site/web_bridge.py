"""JSON boundary between the browser UI and DichromaticMap's Qt-free core.

This module is loaded into Pyodide. All physical calculations call the same
Python functions used by the desktop viewer.
"""

import base64
from dataclasses import asdict
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from dichromatic_map.appearance import default_layer_symbols
from dichromatic_map.cells import StrainedCell, checked_integer_array, count_cell_atoms
from dichromatic_map.completion import cell_completion_candidates
from dichromatic_map.crystal import (
    crystal_vector_coordinates, csl_presets, format_direction_components, get_geometry,
    in_plane_reference_axes, in_plane_reference_directions,
    matching_csl_preset, misorientation_range, projected_columns,
)
from dichromatic_map.matching import (
    exact_csl_cell, local_near_pairs, same_layer_coincidence_sites,
)
from dichromatic_map.session import (
    FORMAT, SCHEMA_VERSION, _snapshot_from_payload, _state_payload, load_session,
)
from dichromatic_map.state import (
    CellVertex, PatternParameters, PatternState, SelectedAtom, default_angle_deg,
)
from dichromatic_map.strain import (
    CandidatePreparation, SelectedCellStrain, cache_cell_search, candidate_vectors,
    get_cached_cell_search, pareto_cells, solve_cells_chunk,
    selected_cell_strain_readout, strain_selected_cell, tensor_readout,
)
from dichromatic_map.exports import build_export_tables


def _array(value):
    return np.asarray(value, dtype=float)


_MAX_SAFE_INTEGER = 2**53 - 1


def _safe_integer_values(values):
    """Check integer identities before a JSON number or Float64 buffer cast."""
    array = np.asarray(values)
    if not array.size:
        return
    if array.dtype.kind in "iuf":
        outside = array.max() > _MAX_SAFE_INTEGER or array.min() < -_MAX_SAFE_INTEGER
    else:
        outside = any(abs(int(value)) > _MAX_SAFE_INTEGER for value in array.flat)
    if outside:
        raise ValueError("Result exceeds the browser's safe integer range; use the Python or desktop interface")


def _safe_json_integers(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        _safe_integer_values(value)
    elif isinstance(value, dict):
        for item in value.values():
            _safe_json_integers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _safe_json_integers(item)


def _axial_repeat(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not -4 <= value <= 4 or value != int(value)):
        raise ValueError("Axial repeat must be an integer between -4 and 4")
    return int(value)


def _cell(cell):
    if cell is None:
        return None
    atoms = cell.atoms
    for values in (cell.m1, cell.m2, atoms):
        _safe_integer_values(values)
    return {
        "m1": cell.m1.tolist(), "m2": cell.m2.tolist(),
        "f1": cell.f1.tolist(), "f2": cell.f2.tolist(),
        "cell": cell.cell.tolist(), "max_strain": cell.max_strain,
        "atoms": list(atoms), "label": cell.label(),
    }


def _from_cell(value, lattice, axis):
    if value is None:
        return None
    for field in ("m1", "m2"):
        _safe_integer_values(value[field])
    return StrainedCell(
        checked_integer_array(value["m1"]), checked_integer_array(value["m2"]),
        _array(value["f1"]), _array(value["f2"]), _array(value["cell"]),
        float(value["max_strain"]), lattice, axis,
    )


def _metadata(request):
    lattice, axis = request["lattice"], request["axis"]
    geometry = get_geometry(lattice, axis)
    presets = csl_presets(geometry.axis)
    return {
        "lattice": geometry.lattice, "axis": geometry.axis,
        "axis_label": geometry.axis_label,
        "layers": geometry.layer_count,
        "layer_spacing": geometry.layer_spacing,
        "axial_period": geometry.axial_period,
        "axial_repeat_half_indices": geometry.axial_repeat_half_indices.tolist(),
        "frame_axis": geometry.frame[:, 2].tolist(),
        "symbols": default_layer_symbols(geometry.layer_count),
        "reference_labels": ["[" + " ".join(map(str, vector)) + "]"
                             for vector in in_plane_reference_directions(geometry.axis)],
        "max_angle": misorientation_range(geometry.axis, lattice).maximum_deg,
        "default_angle": default_angle_deg(geometry.axis),
        "angle_snap_tolerance": 0.006,
        "presets": [{"name": p.name, "angle": p.angle_deg, "label": p.label}
                    for p in presets],
    }


def _render_data(request):
    lattice, axis = request["lattice"], request["axis"]
    angle = float(request["angle"])
    # Rendering consumes an already chosen physical angle. Interactive inputs
    # snap before any request; imported sessions retain their exact saved angle.
    preset = matching_csl_preset(angle, tolerance_deg=1e-10, axis=axis)
    geometry = get_geometry(lattice, axis)
    if not 0 <= angle <= misorientation_range(axis, lattice).maximum_deg:
        raise ValueError("Misorientation is outside the allowed range")
    width, height = float(request["width"]), float(request["height"])
    if not 0 < width <= 150 or not 0 < height <= 150:
        raise ValueError("View is outside the browser range")
    center = tuple(request.get("center", (0, 0)))
    f = _array(request.get("deformations", [np.eye(2), np.eye(2)]))
    t = _array(request.get("translations", [[0, 0], [0, 0]]))
    halo = 2 * float(request.get("local_distance", 0.1)) if request.get("local_matching") else 0.0
    grains = [projected_columns(width + 2 * halo, height + 2 * halo, sign * angle / 2,
                                center=center, deformation=f[g], lattice=lattice,
                                axis=axis, translation=t[g])
              for g, sign in enumerate((1, -1))]
    for grain in grains:
        _safe_integer_values(grain.half_indices)
    matches = same_layer_coincidence_sites(*grains, 1e-6)
    local = None
    if request.get("local_matching"):
        pairs = local_near_pairs(*grains, float(request.get("local_distance", 0.1)))
        local = np.column_stack((pairs.first, pairs.second, pairs.layers))
    coincidences = np.empty((sum(len(points) for points in matches), 3), dtype=float)
    start = 0
    for layer, points in enumerate(matches):
        stop = start + len(points)
        coincidences[start:stop, :2] = points
        coincidences[start:stop, 2] = layer
        start = stop
    metadata = {
        "angle": angle, "preset": preset.name if preset else None,
        "exact_cell": _cell(exact_csl_cell(angle, lattice=lattice, axis=axis))
                      if np.allclose(f, np.eye(2)) and np.allclose(t, 0) else None,
        "layers": geometry.layer_count,
        "reference_axes": [in_plane_reference_axes(sign * angle / 2, f[g]).tolist()
                           for g, sign in enumerate((1, -1))],
    }
    return metadata, [np.column_stack((grain.positions, grain.layers, grain.half_indices))
                      for grain in grains], coincidences, local


def _render(request):
    """Preserve the JSON adapter used by exports, smoke checks and clients."""
    metadata, grains, coincidences, local = _render_data(request)
    return {
        **metadata, "grains": [grain.tolist() for grain in grains],
        "coincidences": [[float(x), float(y), int(layer)] for x, y, layer in coincidences],
        "local": None if local is None else
        [[float(x1), float(y1), float(x2), float(y2), int(layer)]
         for x1, y1, x2, y2, layer in local],
    }


def web_render(serialized):
    """Provide contiguous buffers for Pyodide without boxing every atom."""
    metadata, grains, coincidences, local = _render_data(json.loads(serialized))
    return {
        "metadata": json.dumps(metadata, allow_nan=False),
        "grains": [grain.ravel() for grain in grains],
        "coincidences": coincidences.ravel(),
        "local": None if local is None else local.ravel(),
    }


def web_near_start(serialized):
    request = json.loads(serialized)
    args = (float(request["angle"]), float(request["percent"]),
            int(request["index"]), request["lattice"], request["axis"])
    cached = get_cached_cell_search(*args, rows_per_chunk=8)
    job = {"args": args, "first": None, "second": None, "cells": cached or [],
           "next": 0, "preparation": None, "complete": cached is not None,
           "cached": cached is not None}
    if cached is None:
        preparation = CandidatePreparation(*args)
        if preparation.done:
            job["first"], job["second"] = preparation.step()
        else:
            job["preparation"] = preparation
    return job


def web_near_step(job):
    """Compute the next historical eight-row block, independent of device speed."""
    if job["complete"]:
        return True
    if job["preparation"] is not None:
        candidates = job["preparation"].step(max_offsets=32)
        if candidates is not None:
            job["first"], job["second"] = candidates
            job["preparation"] = None
        # Yield between preparation blocks, including the transition into
        # solving, so obsolete searches can stop before either expensive phase.
        return False
    args, start = job["args"], job["next"]
    stop = start + 8
    if start < len(job["first"]):
        _, part = solve_cells_chunk(args[0], args[1], job["first"], job["second"],
                                    start, stop, args[3], args[4])
        job["cells"].extend(part)
    job["next"] = min(stop, len(job["first"]))
    job["complete"] = stop >= len(job["first"])
    return job["complete"]


def web_near_finish(job):
    if not job["complete"]:
        raise ValueError("Cannot publish an incomplete browser search")
    cells = job["cells"] if job["cached"] else pareto_cells(job["cells"])
    if not job["cached"]:
        cache_cell_search(cells, *job["args"], rows_per_chunk=8)
        job["cells"], job["cached"] = cells, True
    return json.dumps([{**_cell(cell), "readout": tensor_readout(cell, job["args"][0])}
                       for cell in cells], allow_nan=False)


def _near_search(request):
    job = web_near_start(json.dumps(request))
    while not web_near_step(job):
        pass
    return json.loads(web_near_finish(job))


def _count(request):
    polygons = _array(request["polygons"])
    result = count_cell_atoms(
        polygons, float(request["angle"]), _array(request["deformations"]),
        request["lattice"], request["axis"],
        boundary_points=request.get("boundary") if request.get("use_boundary") else None,
        region_states=tuple(request.get("region_states", [True] * 4)),
        layer=int(request["layer"]), translations=_array(request["translations"]),
    )
    for values in (result.interior, result.boundary, result.half_open,
                   result.half_open_edges, result.half_open_corners):
        if values is not None:
            _safe_integer_values(values)
    return {
        "interior": result.interior.tolist(), "boundary": result.boundary.tolist(),
        "half_open": result.half_open.tolist() if result.half_open is not None else None,
        "half_open_edges": result.half_open_edges.tolist() if result.half_open_edges is not None else None,
        "half_open_corners": result.half_open_corners.tolist() if result.half_open_corners is not None else None,
        "areas": result.areas.tolist(),
        "half_open_available": result.half_open_available.tolist(),
    }


def _complete(request):
    candidates = cell_completion_candidates(
        _array(request["polygons"]), float(request["angle"]),
        lattice=request["lattice"], axis=request["axis"],
        layer=int(request["layer"]), percent=float(request["percent"]),
        max_rotation_deg=float(request["rotation"]),
    )
    for candidate in candidates:
        _safe_integer_values(candidate.atoms)
    return [{"description": c.description, "source": c.source,
             "vertices": c.vertices.tolist(), "atoms": c.atoms,
             "areas": c.areas, "error": c.error,
             "strain": c.fit.cell.max_strain if c.fit else None}
            for c in candidates]


def _fit_selected(request):
    fit = strain_selected_cell(
        _array(request["polygons"]), float(request["angle"]),
        percent=float(request["percent"]), lattice=request["lattice"],
        axis=request["axis"], layer=int(request["layer"]),
        max_rotation_deg=float(request["rotation"]),
    )
    return {"cell": _cell(fit.cell), "translations": fit.translations.tolist(),
            "vertices": fit.vertices.tolist(), "origin": fit.origin.tolist(),
            "residual": fit.residual, "rotations_deg": fit.rotations_deg.tolist(),
            "stretches": fit.stretches.tolist(),
            "readout": selected_cell_strain_readout(
                fit, float(request["angle"]),
                strain_limit_percent=float(request["percent"]),
                rotation_limit_deg=float(request["rotation"]),
            )}


def _vector(request):
    a, b = request["atoms"]
    geometry = get_geometry(request["lattice"], request["axis"])
    for atom in (a, b):
        _safe_integer_values(atom["half_indices"])
    half = (checked_integer_array(b["half_indices"]) - checked_integer_array(a["half_indices"])
            + _axial_repeat(request["axial_repeat"]) * geometry.axial_repeat_half_indices)
    dz = 0.5 * half @ geometry.frame[:, 2]
    displacement = np.r_[np.asarray(b["position"]) - np.asarray(a["position"]), dz]
    grains = (0, 1) if a["grain"] != b["grain"] else (a["grain"],)
    vectors = {}
    for grain in grains:
        coordinates = crystal_vector_coordinates(
            displacement, (1 - 2 * grain) * float(request["angle"]) / 2,
            _array(request["deformations"][grain]), request["lattice"], request["axis"],
        )
        vectors[f"G{grain + 1}"] = {
            "current": coordinates.current.tolist(),
            "lattice": coordinates.lattice.tolist(),
            "current_formatted": format_direction_components(coordinates.current),
            "lattice_formatted": format_direction_components(coordinates.lattice, unit=""),
            "strained": coordinates.strained,
        }
    return {"displacement": displacement.tolist(),
            "length": float(np.linalg.norm(displacement)), "vectors": vectors}


def _state_from_browser(raw):
    parameters = PatternParameters(
        angle_deg=float(raw["angle"]), lattice_constant=float(raw.get("a0", 3.52)),
        width=12, height=9, marker_size=32, view_scale=1,
        lattice=raw["lattice"], axis=raw["axis"],
    )
    state = PatternState(parameters)
    state.angle_deg = float(raw["angle"])
    state.interaction_mode = raw.get("mode", "idle")
    state.display_rotation_deg = float(raw.get("display_rotation", 0))
    state.show_reference_axes = bool(raw.get("reference_axes", True))
    state.grain_colors = list(raw["colors"])
    state.layer_symbols = list(raw["symbols"])
    state.layer_size_scales = list(raw["sizes"])
    state.selected_layer = int(raw.get("selected_layer", -1))
    state.visible_grain_layers = [set(layers) for layers in raw["visible_layers"]]
    state.selected_points = [_array(point) for point in raw["boundary"]]
    for atom in raw["atoms"]:
        _safe_integer_values(atom["half_indices"])
    state.selected_atoms = [SelectedAtom(_array(atom["position"]), int(atom["grain"]),
                                      int(atom["layer"]), checked_integer_array(atom["half_indices"]))
                            for atom in raw["atoms"]]
    state.manual_vertices = [CellVertex(_array(v["position"]), int(v["layer"]),
                                        v["source"], _array(v["endpoints"]))
                             for v in raw["manual"]]
    state.manual_local_cutoff = raw.get("manual_local_cutoff")
    state.axial_repeat = _axial_repeat(raw.get("axial_repeat", 0))
    state.near_enabled = bool(raw.get("near_enabled", False))
    state.near_method = raw.get("near_method", "local")
    state.near_cell = _from_cell(raw.get("near_cell"), raw["lattice"], raw["axis"])
    state.deformations = _array(raw["deformations"])
    state.translations = _array(raw["translations"])
    if raw.get("manual_original") is not None:
        state.manual_unstrained_vertices = [
            CellVertex(_array(v["position"]), int(v["layer"]), v["source"],
                       _array(v["endpoints"])) for v in raw["manual_original"]
        ]
        state.manual_unstrained_cutoff = raw.get("manual_local_cutoff")
    if raw.get("manual_fit") is not None:
        fit = raw["manual_fit"]
        state.manual_strain_fit = SelectedCellStrain(
            _from_cell(fit["cell"], raw["lattice"], raw["axis"]),
            _array(fit["translations"]), _array(fit["vertices"]),
            _array(fit["origin"]), float(fit["residual"]),
            _array(fit["rotations_deg"]), _array(fit["stretches"]),
        )
    return state


def _save_session(request):
    state = _state_from_browser(request["state"])
    settings = request["settings"]
    view = request["view_range"]
    payload = {"format": FORMAT, "schema_version": SCHEMA_VERSION,
               "state": _state_payload(state), "settings": settings, "view_range": view}
    _snapshot_from_payload(payload)
    members = {"session.json": json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                           indent=2).encode("utf-8")}
    members.update({name: text.encode("utf-8")
                    for name, text in build_export_tables(state, settings).items()})
    stream = BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _load_session(request):
    encoded = base64.b64decode(request["bytes"], validate=True)
    path = "/tmp/browser-import.dmap"
    with open(path, "wb") as stream:
        stream.write(encoded)
    snapshot = load_session(path)
    return {"state": _state_payload(snapshot.state), "settings": snapshot.settings,
            "view_range": snapshot.view_range}


_ACTIONS = {
    "metadata": _metadata, "render": _render, "near_search": _near_search,
    "count": _count, "complete": _complete, "fit_selected": _fit_selected,
    "vector": _vector, "save_session": _save_session, "load_session": _load_session,
}


def web_dispatch(serialized):
    request = json.loads(serialized)
    action = request.pop("action")
    if action not in _ACTIONS:
        raise ValueError("Unknown browser action")
    result = _ACTIONS[action](request)
    _safe_json_integers(result)
    return json.dumps(result, allow_nan=False)
