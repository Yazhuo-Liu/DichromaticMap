"""Exact shared column bases, with unchanged grain meaning and physical F."""

from dataclasses import replace

import numpy as np
import pytest

from dichromatic_map.cells import StrainedCell, bases, determinant
from dichromatic_map.strain import canonical_cell_basis, find_strained_cells


@pytest.mark.parametrize("transform", [
    [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[-1, 0], [0, -1]],
    [[1, 7], [0, 1]], [[1, 0], [-5, 1]], [[2, 3], [1, 2]],
])
def test_shared_unimodular_changes_have_exact_same_canonical_cell(transform):
    cell = find_strained_cells(39.5, 2, 12)[-1]
    transform = np.array(transform)
    variant = replace(cell, m1=cell.m1 @ transform, m2=cell.m2 @ transform,
                      cell=cell.cell @ transform)
    before = {field: getattr(variant, field).copy() for field in ("m1", "m2", "f1", "f2", "cell")}
    expected = canonical_cell_basis(cell, 39.5)
    actual = canonical_cell_basis(variant, 39.5)
    for field in ("m1", "m2", "cell"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert abs(determinant(actual.transform)) == 1
    np.testing.assert_array_equal(variant.m1.astype(object) @ actual.transform, actual.m1)
    np.testing.assert_array_equal(variant.m2.astype(object) @ actual.transform, actual.m2)
    assert actual.m1[0, 0] > 0 and actual.m1[1, 1] > 0
    assert actual.m1[1, 0] == 0 and 0 <= actual.m1[0, 1] < actual.m1[0, 0]
    for field, original in before.items():
        np.testing.assert_array_equal(getattr(variant, field), original)
    for f, basis, indices in zip((cell.f1, cell.f2), bases(39.5), (actual.m1, actual.m2)):
        np.testing.assert_allclose(f @ basis @ indices.astype(float), actual.cell,
                                   atol=1e-8, rtol=1e-9)


def test_canonical_integer_arithmetic_does_not_overflow_or_round_parameters():
    size = 10**10
    indices = np.array([[size, size-1], [size+1, size]], dtype=np.int64)
    basis = bases(0, "SC", "100")[0]
    cell = StrainedCell(indices, indices.copy(), np.eye(2), np.eye(2),
                        basis @ indices, 0, "SC", "100")
    result = canonical_cell_basis(cell, 0)
    np.testing.assert_array_equal(result.m1, np.eye(2, dtype=int))
    np.testing.assert_array_equal(result.m2, np.eye(2, dtype=int))
    np.testing.assert_array_equal(result.cell, basis)
    assert determinant(result.transform) == 1
    np.testing.assert_array_equal(indices.astype(object) @ result.transform, result.m1)


@pytest.mark.parametrize("field,invalid", [
    ("m1", [[1.1, 0], [0, 1]]), ("m2", [[1, 0], [0, 0]]),
    ("m1", [[1, 0], [0, np.nan]]), ("f1", [[1, 0], [0, np.inf]]),
    ("cell", [[100, 0], [0, 100]]),
])
def test_canonical_basis_rejects_noninteger_or_inconsistent_cells(field, invalid):
    cell = StrainedCell(np.eye(2, dtype=int), np.eye(2, dtype=int), np.eye(2),
                        np.eye(2), bases(0, "SC", "100")[0], 0, "SC", "100")
    with pytest.raises(ValueError):
        canonical_cell_basis(replace(cell, **{field: np.array(invalid)}), 0)
