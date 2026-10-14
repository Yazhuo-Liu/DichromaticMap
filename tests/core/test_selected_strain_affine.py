"""Large-origin fits must obey the same actual F*x+t model as regeneration."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from dichromatic_map.cells import bases
from dichromatic_map.crystal import get_geometry, projected_columns
from dichromatic_map.strain import strain_selected_cell


def square_pair():
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    return np.stack([uv @ (basis * side).T
                     for basis, side in zip(bases(0, "SC", "100"), (49, 51))])


def decimal_affine(points, matrix, translation):
    # Independent exact arithmetic on the actual saved Float64 F,x,t inputs.
    # There is no atom averaging, tolerance relaxation or platform longdouble.
    with localcontext() as context:
        context.prec = 100
        number = lambda value: Decimal.from_float(float(value))
        return np.array([
            [float(sum(number(matrix[i, j]) * number(point[j]) for j in (0, 1))
                   + number(translation[i])) for i in (0, 1)]
            for point in np.asarray(points).reshape(-1, 2)
        ]).reshape(np.asarray(points).shape)


@pytest.mark.parametrize("offset", [60_000_000, -60_000_000, 90_000_000])
def test_feasible_large_origin_fit_retains_f_and_matches_independent_affine_oracle(offset):
    original = square_pair()
    unshifted = strain_selected_cell(original, 0, 2, "SC", "100", max_rotation_deg=0)
    vertices = original + offset
    before = vertices.copy()
    fit = strain_selected_cell(vertices, 0, 2, "SC", "100", max_rotation_deg=0)
    np.testing.assert_array_equal(vertices, before)
    for name in ("f1", "f2"):
        np.testing.assert_array_equal(getattr(fit.cell, name), getattr(unshifted.cell, name))
    target = vertices.mean(axis=1).mean(axis=0)
    for grain, f in enumerate((fit.cell.f1, fit.cell.f2)):
        expected_shift = decimal_affine(vertices[grain].mean(axis=0), -f, target)
        np.testing.assert_array_equal(fit.translations[grain], expected_shift)
        expected = decimal_affine(vertices[grain], f, expected_shift)
        np.testing.assert_array_equal(fit.vertices[grain], expected)
    assert fit.residual <= 1e-8
    assert fit.cell.max_strain <= .02 + 1e-12


def test_actual_large_origin_generated_sites_obey_the_saved_f_t_model():
    original = square_pair()
    base_fit = strain_selected_cell(original, 0, 2, "SC", "100", max_rotation_deg=0)
    vertices = original + 60_000_000
    centroids = vertices.mean(axis=1)
    target = centroids.mean(axis=0)
    geometry = get_geometry("SC", "100")
    actual = []
    for grain, f in enumerate((base_fit.cell.f1, base_fit.cell.f2)):
        shift = decimal_affine(centroids[grain], -f, target)
        expected = decimal_affine(vertices[grain], f, shift)
        regenerated = projected_columns(
            54, 54, 0, center=target, deformation=f, translation=shift,
            lattice="SC", axis="100",
        )
        ids = np.rint(vertices[grain]).astype(np.int64) @ geometry.basis_half_indices.T
        endpoints = []
        for identifier in ids:
            rows = np.flatnonzero(np.all(regenerated.half_indices == identifier, axis=1))
            assert len(rows) == 1
            endpoints.append(regenerated.positions[rows[0]])
        endpoints = np.array(endpoints)
        np.testing.assert_array_equal(endpoints, expected)
        actual.append(endpoints)
    assert np.max(np.linalg.norm(actual[0] - actual[1], axis=1)) <= 1e-8
