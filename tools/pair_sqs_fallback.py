"""General simulated-annealing fallback for exact random pair correlations.

This deliberately independent Python implementation is a recovery path for
periodic graphs on which the primary SCRAPS optimizer stalls.  It uses the
same periodic-image shell oracle as verification, preserves species counts,
and returns its best candidate with an explicit residual rather than claiming
success implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence

import numpy as np
from pymatgen.core import Structure

from periodic_multigraph import periodic_shells, warren_cowley


@dataclass(frozen=True)
class PairSQSResult:
    structure: Structure
    objective: float
    steps: int
    accepted_moves: int
    alpha: np.ndarray
    pair_counts: np.ndarray
    species: tuple[str, ...]


def _directed_counts(labels: np.ndarray, centers: np.ndarray, neighbors: np.ndarray, ntypes: int) -> np.ndarray:
    counts = np.zeros((ntypes, ntypes), dtype=int)
    np.add.at(counts, (labels[centers], labels[neighbors]), 1)
    return counts


def random_pair_targets(populations: Sequence[int], coordination: int) -> np.ndarray:
    """Expected directed pair counts for alpha=0 random mixing."""
    pop = np.asarray(populations, dtype=float)
    total = int(pop.sum())
    target = coordination * np.outer(pop, pop) / total
    if not np.allclose(target, np.rint(target), atol=1e-10):
        raise ValueError("exact alpha=0 pair counts are nonintegral for this cell")
    return np.rint(target).astype(int)


def anneal_random_pairs(
    structure: Structure,
    species: Sequence[str],
    populations: Sequence[int],
    *,
    shell_index: int = 0,
    seed: int = 0,
    steps: int = 250_000,
    initial_temperature: float = 12.0,
    final_temperature: float = 0.02,
) -> PairSQSResult:
    """Minimize exact first-/selected-shell random-pair residual by swaps.

    The objective is the squared residual of every directed pair count.  Each
    proposed composition-preserving swap updates only incident graph edges,
    making the fallback usable for skewed periodic SQS cells.
    """
    if len(species) != len(populations) or sum(populations) != len(structure):
        raise ValueError("species/populations must exactly describe structure")
    if steps < 1 or initial_temperature <= 0 or final_temperature <= 0:
        raise ValueError("positive annealing schedule required")
    shell = periodic_shells(structure, shell_index + 1)[shell_index]
    target = random_pair_targets(populations, shell.coordination)
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(species), dtype=int), np.asarray(populations, dtype=int))
    rng.shuffle(labels)
    centers, neighbors = shell.centers.astype(int), shell.neighbors.astype(int)
    incident: list[list[int]] = [[] for _ in range(len(labels))]
    for edge, (i, j) in enumerate(zip(centers, neighbors, strict=True)):
        incident[i].append(edge)
        if j != i:
            incident[j].append(edge)
    counts = _directed_counts(labels, centers, neighbors, len(species))
    objective = float(np.square(counts - target).sum())
    best_objective, best_labels, best_counts = objective, labels.copy(), counts.copy()
    accepted = 0
    for step in range(steps):
        i, j = rng.integers(0, len(labels), size=2)
        if i == j or labels[i] == labels[j]:
            continue
        edges = np.fromiter(set(incident[i]).union(incident[j]), dtype=int)
        old_i, old_j = labels[i], labels[j]
        changed = counts.copy()
        for edge in edges:
            changed[labels[centers[edge]], labels[neighbors[edge]]] -= 1
        labels[i], labels[j] = old_j, old_i
        for edge in edges:
            changed[labels[centers[edge]], labels[neighbors[edge]]] += 1
        new_objective = float(np.square(changed - target).sum())
        fraction = step / max(steps - 1, 1)
        temperature = initial_temperature * (final_temperature / initial_temperature) ** fraction
        if new_objective <= objective or rng.random() < exp((objective - new_objective) / temperature):
            counts, objective = changed, new_objective
            accepted += 1
            if objective < best_objective:
                best_objective, best_labels, best_counts = objective, labels.copy(), counts.copy()
                if best_objective == 0:
                    break
        else:
            labels[i], labels[j] = old_i, old_j
    output = structure.copy()
    for index, value in enumerate(best_labels):
        output.replace(index, species[int(value)])
    alpha, verified_counts, ordered_species = warren_cowley(output, shell)
    if not np.array_equal(best_counts, verified_counts):
        raise RuntimeError("fallback internal pair count disagrees with independent oracle")
    return PairSQSResult(output, best_objective, step + 1, accepted, alpha, verified_counts, tuple(ordered_species))
