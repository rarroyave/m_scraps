#!/usr/bin/env python3
"""Independently verify SRO with explicit periodic-image multiplicities."""
from __future__ import annotations

import argparse
import json

from pymatgen.core import Structure

from periodic_multigraph import periodic_shells, warren_cowley


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure")
    parser.add_argument("--shells", type=int, default=1)
    args = parser.parse_args()
    structure = Structure.from_file(args.structure)
    report = []
    for shell in periodic_shells(structure, args.shells):
        alpha, counts, species = warren_cowley(structure, shell)
        report.append({
            "index": shell.index, "distance_angstrom": shell.distance,
            "coordination": shell.coordination, "edge_count": int(len(shell.centers)),
            "alpha_matrix": alpha.tolist(), "pair_counts": counts.tolist(),
        })
    print(json.dumps({"n_atoms": len(structure), "species": species,
                      "periodic_image_aware": True, "shells": report}, indent=2))


if __name__ == "__main__":
    main()
