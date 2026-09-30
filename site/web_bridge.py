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
from dichromatic_map.cells import StrainedCell, count_cell_atoms
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
    SelectedCellStrain, candidate_vectors, pareto_cells, solve_cells_chunk,
    selected_cell_strain_readout, strain_selected_cell, tensor_readout,
)
from dichromatic_map.exports import build_export_tables


def _array(value):
    return np.asarray(value, dtype=float)


def _cell(cell):
    if cell is None:
        return None
    return {
        "m1": cell.m1.tolist(), "m2": cell.m2.tolist(),
        "f1": cell.f1.tolist(), "f2": cell.f2.tolist(),
        "cell": cell.cell.tolist(), "max_strain": cell.max_strain,
        "atoms": list(cell.atoms), "label": cell.label(),
    }


def _from_cell(value, lattice, axis):
    if value is None:
        return None
    return StrainedCell(
        np.asarray(value["m1"], dtype=int), np.asarray(value["m2"], dtype=int),
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
        "presets": [{"name": p.name, "angle": p.angle_deg, "label": p.label}
                    for p in presets],
    }


def _render(request):
    lattice, axis = request["lattice"], request["axis"]
    angle = float(request["angle"])
    preset = matching_csl_preset(angle, axis=axis)
    if preset is not None:
        angle = preset.angle_deg
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
    matches = same_layer_coincidence_sites(*grains, 1e-6)
    local = None
    if request.get("local_matching"):
        pairs = local_near_pairs(*grains, float(request.get("local_distance", 0.1)))
        local = [[*a.tolist(), *b.tolist(), int(layer)]
                 for a, b, layer in zip(pairs.first, pairs.second, pairs.layers)]
    return {
        "angle": angle, "preset": preset.name if preset else None,
        "grains": [np.column_stack((grain.positions, grain.layers,
                                    grain.half_indices)).tolist() for grain in grains],
        "coincidences": [[float(x), float(y), layer]
                         for layer, points in enumerate(matches) for x, y in points],
        "local": local,
        "exact_cell": _cell(exact_csl_cell(angle, lattice=lattice, axis=axis))
                      if np.allclose(f, np.eye(2)) and np.allclose(t, 0) else None,
        "layers": geometry.layer_count,
        "reference_axes": [in_plane_reference_axes(sign * angle / 2, f[g]).tolist()
                           for g, sign in enumerate((1, -1))],
    }


def _near_search(request):
    args = (float(request["angle"]), float(request["percent"]),
            int(request["index"]), request["lattice"], request["axis"])
    first, second = candidate_vectors(*args)
    cells = []
    for start in range(0, len(first), 8):
        _, part = solve_cells_chunk(args[0], args[1], first, second,
                                    start, start + 8, args[3], args[4])
        cells.extend(part)
    return [{**_cell(cell), "readout": tensor_readout(cell, args[0])}
            for cell in pareto_cells(cells)]


def _count(request):
    polygons = _array(request["polygons"])
    result = count_cell_atoms(
        polygons, float(request["angle"]), _array(request["deformations"]),
        request["lattice"], request["axis"],
        boundary_points=request.get("boundary") if request.get("use_boundary") else None,
        region_states=tuple(request.get("region_states", [True] * 4)),
        layer=int(request["layer"]), translations=_array(request["translations"]),
    )
    return {
        "interior": result.interior.tolist(), "boundary": result.boundary.tolist(),
        "half_open": result.half_open.tolist() if result.half_open is not None else None,
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
    half = (np.asarray(b["half_indices"]) - np.asarray(a["half_indices"])
            + int(request["axial_repeat"]) * geometry.axial_repeat_half_indices)
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
    state.selected_atoms = [SelectedAtom(_array(atom["position"]), int(atom["grain"]),
                                      int(atom["layer"]), np.asarray(atom["half_indices"], dtype=int))
                            for atom in raw["atoms"]]
    state.manual_vertices = [CellVertex(_array(v["position"]), int(v["layer"]),
                                        v["source"], _array(v["endpoints"]))
                             for v in raw["manual"]]
    state.manual_local_cutoff = raw.get("manual_local_cutoff")
    state.axial_repeat = int(raw.get("axial_repeat", 0))
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
    return json.dumps(_ACTIONS[action](request), allow_nan=False)
