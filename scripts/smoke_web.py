"""Run the built web adapter against the same core package shipped to Pages."""

import json
from pathlib import Path
import runpy
import sys
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "site" / "_build" / "vendor" / "dichromatic_map.zip"))
dispatch = runpy.run_path(str(ROOT / "site" / "_build" / "web_bridge.py"))["web_dispatch"]


def call(action, **kwargs):
    return json.loads(dispatch(json.dumps({"action": action, **kwargs})))


info = call("metadata", lattice="FCC", axis="110")
assert info["layers"] == 2 and info["max_angle"] == 90
for lattice in ("SC", "FCC", "BCC"):
    for axis in ("100", "110", "111", "112"):
        assert call("metadata", lattice=lattice, axis=axis)["layers"] >= 1
assert call("metadata", lattice="FCC", axis="1 -1 3")["axis"] == "1 -1 3"
angle = next(p["angle"] for p in info["presets"] if p["name"] == "Σ9")
rendered = call("render", lattice="FCC", axis="110", angle=angle,
                width=12, height=9, center=[0, 0])
assert len(rendered["grains"][0]) > 200
assert len(rendered["coincidences"]) > 10
assert rendered["exact_cell"] is not None
assert len(rendered["grains"][0][0]) == 6
cell = rendered["exact_cell"]["cell"]
v1, v2 = [cell[0][0], cell[1][0]], [cell[0][1], cell[1][1]]
corners = [[0, 0], v1, [v1[0] + v2[0], v1[1] + v2[1]], v2]
counted = call("count", polygons=[corners, corners], angle=angle,
               deformations=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
               translations=[[0, 0], [0, 0]], lattice="FCC", axis="110", layer=0)
assert counted["interior"][0][0] >= 0
fit = call("fit_selected", polygons=[corners, corners], angle=angle,
           lattice="FCC", axis="110", layer=0, percent=2, rotation=1)
assert fit["residual"] < 1e-8
atom1, atom2 = rendered["grains"][0][:2]
vector = call("vector", atoms=[
    {"position": atom1[:2], "grain": 0, "layer": int(atom1[2]), "half_indices": atom1[3:]},
    {"position": atom2[:2], "grain": 0, "layer": int(atom2[2]), "half_indices": atom2[3:]},
], axial_repeat=0, angle=angle, lattice="FCC", axis="110",
    deformations=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
assert vector["length"] > 0
assert not vector["vectors"]["G1"]["strained"]
assert "[" in vector["vectors"]["G1"]["current_formatted"]
local = call("render", lattice="FCC", axis="110", angle=22,
             width=12, height=9, center=[0, 0], local_matching=True, local_distance=0.1)
assert local["local"] is not None
near_cells = call("near_search", lattice="FCC", axis="110", angle=22,
                  percent=2, index=8)
assert near_cells
strained_render = call("render", lattice="FCC", axis="110", angle=22,
                       width=12, height=9, center=[0, 0],
                       deformations=[near_cells[0]["f1"], near_cells[0]["f2"]])
strained_atom1, strained_atom2 = strained_render["grains"][0][:2]
strained_vector = call("vector", atoms=[
    {"position": atom[:2], "grain": 0, "layer": int(atom[2]), "half_indices": atom[3:]}
    for atom in (strained_atom1, strained_atom2)
], axial_repeat=0, angle=22, lattice="FCC", axis="110",
    deformations=[near_cells[0]["f1"], near_cells[0]["f2"]])["vectors"]["G1"]
assert strained_vector["strained"]
assert strained_vector["current_formatted"].startswith("≈ a₀[")
assert strained_vector["lattice_formatted"] == "1/2[1 -1 0]"
other_grain_atom = strained_render["grains"][1][0]
cross_grain_vectors = call("vector", atoms=[
    {"position": strained_atom1[:2], "grain": 0, "layer": int(strained_atom1[2]),
     "half_indices": strained_atom1[3:]},
    {"position": other_grain_atom[:2], "grain": 1, "layer": int(other_grain_atom[2]),
     "half_indices": other_grain_atom[3:]},
], axial_repeat=0, angle=22, lattice="FCC", axis="110",
    deformations=[near_cells[0]["f1"], near_cells[0]["f2"]])["vectors"]
assert set(cross_grain_vectors) == {"G1", "G2"}
assert all(vector["strained"] and "[" in vector["lattice_formatted"]
           for vector in cross_grain_vectors.values())

visible = [[0, 1], [0, 1]]
browser_state = {
    "lattice": "FCC", "axis": "110", "angle": angle, "a0": 3.52,
    "mode": "idle", "display_rotation": 0, "reference_axes": True,
    "colors": ["#468ec6", "#ef643f"], "symbols": info["symbols"],
    "sizes": [1, 1], "visible_layers": visible, "boundary": [],
    "atoms": [], "manual": [], "manual_local_cutoff": None,
    "axial_repeat": 0, "near_enabled": False, "near_method": "local",
    "near_cell": None,
    "deformations": [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
    "translations": [[0, 0], [0, 0]],
}
settings = {
    "region_states": [True] * 4, "manual_visible": False,
    "show_common_cell": False, "local_distance": 0.1,
    "strain_percent": 2, "search_index": 12,
    "manual_strain_percent": 2, "manual_rotation_deg": 1,
}
encoded = call("save_session", state=browser_state, settings=settings,
               view_range=[-6, 6, -4.5, 4.5])
loaded = call("load_session", bytes=encoded)
assert loaded["state"]["parameters"]["axis"] == "110"
assert loaded["settings"]["search_index"] == 12
manual_state = dict(browser_state)
manual_state["manual"] = [
    {"position": point, "layer": 0, "source": "CSL",
     "endpoints": [point, point]} for point in corners
]
encoded_manual = call("save_session", state=manual_state, settings=settings,
                      view_range=[-6, 6, -4.5, 4.5])
assert len(call("load_session", bytes=encoded_manual)["state"]["manual_vertices"]) == 4
strained_state = dict(manual_state)
strained_state["near_cell"] = fit["cell"]
strained_state["manual_original"] = manual_state["manual"]
strained_state["manual_fit"] = fit
strained_state["deformations"] = [fit["cell"]["f1"], fit["cell"]["f2"]]
strained_state["translations"] = fit["translations"]
encoded_fit = call("save_session", state=strained_state, settings=settings,
                   view_range=[-6, 6, -4.5, 4.5])
assert call("load_session", bytes=encoded_fit)["state"]["manual_strain_fit"] is not None
near_state = dict(browser_state)
near_state.update(angle=22, near_enabled=True, near_method="strain",
                  near_cell=near_cells[0],
                  deformations=[near_cells[0]["f1"], near_cells[0]["f2"]])
encoded_near = call("save_session", state=near_state, settings=settings,
                    view_range=[-6, 6, -4.5, 4.5])
assert call("load_session", bytes=encoded_near)["state"]["near_cell"] is not None

from dichromatic_map.crystal import projected_columns
from dichromatic_map.matching import local_near_pairs
first = projected_columns(25, 20, 11)
second = projected_columns(25, 20, -11)
pairs = local_near_pairs(first, second, 0.05)
indices = np.flatnonzero(pairs.layers == 1)
chosen = [indices[np.argmin(np.linalg.norm(pairs.midpoints[indices] - target, axis=1))]
          for target in ([0, 5.6], [-2.5, 0], [0, -5.6], [2.5, 0])]
original_pairs = np.stack((pairs.first[chosen], pairs.second[chosen]))
real_fit = call("fit_selected", polygons=original_pairs.tolist(), angle=22,
                lattice="FCC", axis="110", layer=1, percent=2, rotation=1)
original_vertices = [
    {"position": np.mean(original_pairs[:, i], axis=0).tolist(), "layer": 1,
     "source": "local", "endpoints": original_pairs[:, i].tolist()}
    for i in range(4)
]
current_vertices = [
    {"position": np.mean(np.asarray(real_fit["vertices"])[:, i], axis=0).tolist(),
     "layer": 1, "source": "CSL",
     "endpoints": np.asarray(real_fit["vertices"])[:, i].tolist()}
    for i in range(4)
]
real_state = dict(browser_state)
real_state.update(angle=22, near_enabled=True, near_method="local",
                  near_cell=real_fit["cell"],
                  deformations=[real_fit["cell"]["f1"], real_fit["cell"]["f2"]],
                  translations=real_fit["translations"],
                  manual=current_vertices, manual_original=original_vertices,
                  manual_fit=real_fit, manual_local_cutoff=0.05)
encoded_real = call("save_session", state=real_state, settings=settings,
                    view_range=[-6, 6, -4.5, 4.5])
assert call("load_session", bytes=encoded_real)["state"]["manual_strain_fit"] is not None

print("Web adapter smoke passed: geometry, CSL, near search, count, vector, strain, .dmap")
