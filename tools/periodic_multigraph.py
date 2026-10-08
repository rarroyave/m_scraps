"""Reference periodic-image multigraph for SQS neighbor validation.

Unlike a minimum-image adjacency matrix, this representation retains distinct
periodic images of the same site.  It is deliberately calculator-independent
and serves as the oracle for future C++ SCRAPS graph construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Structure


@dataclass(frozen=True)
class PeriodicShell:
    index: int
    distance: float
    centers: np.ndarray
    neighbors: np.ndarray
    images: np.ndarray

    @property
    def coordination(self) -> int:
        counts = np.bincount(self.centers)
        if counts.size == 0 or not np.all(counts == counts[0]):
            raise ValueError("shell does not have uniform per-site coordination")
        return int(counts[0])


def warren_cowley(structure: Structure, shell: PeriodicShell):
    """Return image-aware Warren--Cowley alpha and directed pair counts."""
    names = [str(site.specie) for site in structure]
    species = list(dict.fromkeys(names))
    index = {name: i for i, name in enumerate(species)}
    site_species = np.array([index[name] for name in names])
    counts = np.zeros((len(species), len(species)), dtype=int)
    np.add.at(counts, (site_species[shell.centers], site_species[shell.neighbors]), 1)
    population = np.bincount(site_species, minlength=len(species))
    alpha = np.empty_like(counts, dtype=float)
    for i in range(len(species)):
        for j in range(len(species)):
            alpha[i, j] = 1.0 - counts[i, j] * len(structure) / (
                shell.coordination * population[i] * population[j]
            )
    return alpha, counts, species


def periodic_shells(structure: Structure, n_shells: int, *, relative_tolerance: float = 2e-2) -> list[PeriodicShell]:
    """Return the first ``n_shells`` complete periodic-image shells.

    The search radius grows until all requested shells are separated from the
    cutoff.  Edges retain `(center, neighbor, image)` identity and therefore
    correctly handle Wigner--Seitz-boundary ties.
    """
    if n_shells < 1:
        raise ValueError("n_shells must be positive")
    cutoff = min(structure.lattice.abc) / 2
    for _ in range(12):
        centers, neighbors, images, distances = structure.get_neighbor_list(r=cutoff)
        keep = distances > 1e-8
        distances, centers, neighbors, images = (value[keep] for value in (distances, centers, neighbors, images))
        unique = np.unique(np.round(distances, 10))
        groups: list[list[float]] = []
        for distance in unique:
            if not groups or (distance - groups[-1][-1]) / groups[-1][-1] > relative_tolerance:
                groups.append([float(distance)])
            else:
                groups[-1].append(float(distance))
        if len(groups) > n_shells:
            shells = []
            for index, group in enumerate(groups[:n_shells]):
                low, high = group[0] - 1e-8, group[-1] + 1e-8
                mask = (distances >= low) & (distances <= high)
                shells.append(PeriodicShell(index, float(np.mean(group)), centers[mask], neighbors[mask], images[mask]))
            return shells
        cutoff *= 1.5
    raise RuntimeError("could not isolate requested shells below periodic cutoff")
