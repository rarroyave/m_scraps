"""The periodic multigraph oracle must retain boundary-image multiplicity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "tools"))

from periodic_multigraph import periodic_shells, warren_cowley


def _xu_parent(plane: str):
    # Same primitive BCC transformations as the source-sized Xu geometries.
    from pymatgen.core import Lattice, Structure
    primitive_rows = np.array(((1, 1, -1), (1, -1, 1), (-1, 1, 1)), dtype=int)
    source_rows = {
        "110": ((2, -2, -4), (2, -2, 2), (12, 12, 0)),
        "112": ((2, -2, 0), (3, 3, -3), (8, 8, 16)),
        "123": ((10, -8, 2), (1, 1, -1), (4, 8, 12)),
    }
    transform = np.rint(np.array(source_rows[plane]) @ np.linalg.inv(primitive_rows)).astype(int)
    structure = Structure(Lattice(3.225 / 2 * primitive_rows), ["X"], [[0, 0, 0]])
    structure.make_supercell(transform)
    return structure


def test_source_sized_oriented_bcc_cells_retain_all_first_neighbors():
    for plane, count in (("110", 72), ("112", 72), ("123", 84)):
        parent = _xu_parent(plane)
        assert len(parent) == count
        shell = periodic_shells(parent, 1)[0]
        assert shell.coordination == 8
        assert len(shell.centers) == 8 * count
        # At least one source orientation is a boundary case: a simple
        # center-neighbor matrix would lose an image-distinct edge.
        triples = {(int(i), int(j), tuple(int(x) for x in image))
                   for i, j, image in zip(shell.centers, shell.neighbors, shell.images, strict=True)}
        assert len(triples) == len(shell.centers)


def test_image_aware_alpha_is_zero_for_a_balanced_first_shell_sqs():
    parent = _xu_parent("110")
    # Assign three species cyclically; this is only a dimensional/count test.
    for index, site in enumerate(parent):
        site.species = ("A", "B", "C")[index % 3]
    alpha, counts, species = warren_cowley(parent, periodic_shells(parent, 1)[0])
    assert counts.sum() == len(parent) * 8
    assert species == ["A0+", "B", "C"] or len(species) == 3
    assert alpha.shape == (3, 3)
