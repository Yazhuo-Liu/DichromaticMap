"""Small deterministic numerical examples shared by core tests."""

from functools import lru_cache
import numpy as np
from dichromatic_map import crystal, matching, strain


def rotation(angle):
    theta = np.deg2rad(angle)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@lru_cache(maxsize=16)
def solve(lattice, axis, angle, percent=2.0, extent=12):
    first, second = strain.candidate_vectors(angle, percent, extent, lattice, axis)
    return strain.pareto_cells(strain.solve_cells_chunk(
        angle, percent, first, second, 0, len(first), lattice, axis
    )[1])


def corners(cell):
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ cell.cell.T


def fcc22_diamond_pairs():
    """Four paired FCC [110] B-layer atoms around a 22-degree cell."""
    first = crystal.projected_columns(25, 20, 11, lattice="FCC", axis="110")
    second = crystal.projected_columns(25, 20, -11, lattice="FCC", axis="110")
    pairs = matching.local_near_pairs(first, second, 0.05)
    ids = np.flatnonzero(pairs.layers == 1)
    picked = [
        ids[np.argmin(np.linalg.norm(pairs.midpoints[ids] - target, axis=1))]
        for target in ([0, 5.6], [-2.5, 0], [0, -5.6], [2.5, 0])
    ]
    return pairs.midpoints[picked], np.stack((pairs.first[picked], pairs.second[picked]))
