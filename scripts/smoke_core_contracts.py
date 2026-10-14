"""Check repaired scientific edge cases in Native and the deployed WASM core."""

import inspect


def run_core_contracts(bridge=None):
    """Independent small oracles; this function also runs verbatim in Pyodide."""
    import numpy as np
    from dichromatic_map import cells, crystal, matching, strain

    def grain(points):
        return crystal.ProjectedGrain(
            points, np.zeros(len(points), dtype=np.int16),
            np.zeros((len(points), 3), dtype=np.int64), layer_count=1,
        )

    matrix = np.diag(np.array([3_100_000_000, 3_100_000_000], dtype=np.int64))
    count = cells.determinant(matrix)
    assert count == 3_100_000_000 ** 2, "integer determinant overflowed"

    polygon = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    try:
        cells.cell_membership([[np.nan, 0.]], polygon)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite membership input was silently accepted")

    first = np.array([[-32., 0.], [1.8e-6, 0.]], dtype=np.float32)
    second = np.array([[2.7e-6, 0.]], dtype=np.float32)
    assert np.hypot(*(first[1].astype(float) - second[0].astype(float))) < 1e-6
    sites = matching.same_layer_coincidence_sites(grain(first), grain(second), 1e-6)
    assert len(sites[0]) == 1, "float32 bin arithmetic lost a valid coincidence"

    pairs = matching.local_near_pairs(
        grain(np.array([[0., 0.]])), grain(np.array([[5e-201, 0.]])),
        distance=1e-200, exact_tolerance=0,
    )
    assert len(pairs.first) == 1, "a nonzero near pair underflowed to coincidence"
    assert pairs.distances[0] == 5e-201, "near-pair readout underflowed"

    center = np.array([1e12 + .2, 0.])
    generated = crystal.projected_columns(2, 2, 0, center=tuple(center),
                                           lattice="SC", axis="100")
    expected = np.array([[1e12 + x, y] for y in (-1., 0., 1.) for x in (0., 1.)])
    ordered = np.lexsort((generated.positions[:, 0], generated.positions[:, 1]))
    assert np.array_equal(generated.positions[ordered], expected), "crop escaped its local view"

    options = []
    for amount in (.0100000003, .01):
        deformation = np.eye(2) * (1 + amount)
        options.append(cells.StrainedCell(np.eye(2, dtype=np.int64),
            np.eye(2, dtype=np.int64), deformation, deformation.copy(),
            deformation @ crystal.get_geometry("SC", "100").planar_basis,
            amount, "SC", "100"))
    chosen = strain.pareto_cells(options)
    assert len(chosen) == 1 and chosen[0].max_strain == .01, "deduplication kept the worse fit"

    vertices = np.stack([polygon * side + 60_000_000 for side in (49, 51)])
    fitted = strain.strain_selected_cell(vertices, 0, 2, "SC", "100", max_rotation_deg=0)
    assert fitted.residual <= 1e-8, "translation broke a feasible bounded fit"
    assert fitted.cell.atoms == (49 ** 2, 51 ** 2)
    assert np.allclose(fitted.stretches, [[1.02, 1.02], [.98, .98]], rtol=0, atol=2e-12)
    for index, deformation in enumerate((fitted.cell.f1, fitted.cell.f2)):
        regenerated = crystal.projected_columns(54, 54, 0,
            center=tuple(fitted.vertices[index].mean(axis=0)), lattice="SC", axis="100",
            deformation=deformation, translation=fitted.translations[index])
        for point in fitted.vertices[index]:
            assert np.min(np.hypot(*(regenerated.positions - point).T)) <= 1e-8, "fitted atoms did not regenerate"
    result = {"determinant": str(count), "membershipRejectsNonfinite": True,
            "float32Coincidences": len(sites[0]), "tinyNearPairs": len(pairs.first),
            "cropPoints": len(generated.positions), "paretoStrain": chosen[0].max_strain,
            "translatedFitAtoms": list(fitted.cell.atoms)}
    if bridge is not None:
        matrix = np.diag(np.array([3_100_000_000, 1], dtype=np.int64))
        cell = cells.StrainedCell(matrix, matrix.copy(), np.eye(2), np.eye(2),
                                 matrix.astype(float), 0, "SC", "100")
        restored = bridge["_from_cell"](bridge["_cell"](cell), "SC", "100")
        assert restored.m1.dtype == np.dtype(np.int64), "browser restoration used platform-width integers"
        assert np.array_equal(restored.m1, matrix) and restored.atoms == cell.atoms
        request = {"atoms":[{"position":[0,0],"grain":0,"half_indices":[0,0,0]},
                            {"position":[1,0],"grain":0,"half_indices":[2,0,0]}],
                   "lattice":"FCC", "axis":"110", "angle":22,
                   "deformations":[np.eye(2).tolist(),np.eye(2).tolist()], "axial_repeat":1.5}
        try:
            bridge["_vector"](request)
        except ValueError:
            pass
        else:
            raise AssertionError("browser vector entry truncated a fractional axial repeat")
        result["webInt64Restored"] = True
        result["webRejectsFractionalRepeat"] = True
    return result


def check_core_contracts(page, runtime_url, archive_url, bridge_source):
    """Load the actual deployed archive into an isolated test-only Worker."""
    native_bridge = {}
    exec(compile(bridge_source, "deployed_web_bridge.py", "exec"), native_bridge)
    expected = run_core_contracts(native_bridge)
    source = inspect.getsource(run_core_contracts)
    result = page.evaluate("""async ({runtime, archive, source, bridge}) => {
      const code = String.raw`self.onmessage = async ({data}) => {
        try {
          const {loadPyodide} = await import(data.runtime + 'pyodide.mjs');
          const py = await loadPyodide({indexURL: data.runtime});
          await py.loadPackage('numpy');
          const response = await fetch(data.archive);
          if (!response.ok) throw new Error('Cannot load the deployed core archive');
          py.FS.writeFile('/core.zip', new Uint8Array(await response.arrayBuffer()));
          py.runPython("import sys; sys.path.insert(0, '/core.zip')");
          py.runPython(data.bridge);
          const result = py.runPython('import json\\n' + data.source + '\\njson.dumps(run_core_contracts(globals()))');
          self.postMessage({result: JSON.parse(result)});
        } catch (error) { self.postMessage({error: String(error)}); }
      };`;
      const url = URL.createObjectURL(new Blob([code], {type: 'text/javascript'}));
      const worker = new Worker(url, {type: 'module'});
      try {
        return await new Promise((resolve, reject) => {
          worker.onmessage = ({data}) => data.error ? reject(new Error(data.error)) : resolve(data.result);
          worker.onerror = event => reject(new Error(event.message));
          worker.postMessage({runtime, archive, source, bridge});
        });
      } finally { worker.terminate(); URL.revokeObjectURL(url); }
    }""", {"runtime": runtime_url, "archive": archive_url, "source": source, "bridge":bridge_source})
    assert result == expected, (result, expected)
    return result
