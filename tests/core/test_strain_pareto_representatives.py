"""Physical Pareto representatives use one improvement tolerance and stable ties."""

import numpy as np

from dichromatic_map.cells import StrainedCell
from dichromatic_map.strain import pareto_cells


def uniform_cell(strain, side=1, basis_change=None):
    # An independent, valid SC [100] construction: both grains undergo the
    # same scalar dilation and therefore have exactly common translations.
    edges = side * np.eye(2, dtype=np.int64)
    if basis_change is not None:
        edges = edges @ basis_change
    f = (1 + strain) * np.eye(2)
    actual_strain = abs(float(f[0, 0]) - 1)
    return StrainedCell(edges, edges.copy(), f, f.copy(), f @ edges,
                        actual_strain, "SC", "100")


def test_round9_duplicate_group_keeps_a_materially_better_physical_metric():
    worse, better = uniform_cell(.01 + 3e-10), uniform_cell(.01)
    assert worse.atoms == better.atoms == (1, 1)
    assert worse.max_strain - better.max_strain > 1e-10
    np.testing.assert_array_equal(np.round(worse.f1, 9), np.round(better.f1, 9))
    for order in ([worse, better], [better, worse]):
        result = pareto_cells(order)
        assert len(result) == 1 and result[0] is better


def test_round9_group_keeps_first_representative_inside_the_existing_tie_tolerance():
    first, lower_tie = uniform_cell(.01 + 5e-11), uniform_cell(.01)
    assert 0 < first.max_strain - lower_tie.max_strain < 1e-10
    result = pareto_cells([first, lower_tie])
    assert len(result) == 1 and result[0] is first
    # Different integer bases do not justify changing a physically equal tie.
    equivalent = uniform_cell(.01 + 5e-11, basis_change=np.array([[0, 1], [-1, 0]]))
    result = pareto_cells([first, equivalent])
    assert len(result) == 1 and result[0] is first


def test_cross_chunk_group_improvement_preserves_the_complete_frontier_order():
    cheapest = uniform_cell(.02)
    middle_worse, middle_best = uniform_cell(.01 + 3e-10, 2), uniform_cell(.01, 2)
    lowest_strain = uniform_cell(.009, 3)
    chunks = [[middle_worse, cheapest], [lowest_strain], [middle_best]]
    expected = [cheapest, middle_best, lowest_strain]
    actual = pareto_cells([cell for chunk in chunks for cell in chunk])
    assert len(actual) == 3
    assert all(result is reference for result, reference in zip(actual, expected))
