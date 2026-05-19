#!/usr/bin/env python3
"""Suggest a feasible SCRAPS Cell_DIM for SQS at a given composition.

For a target with all `SRO i j shell 0` (i.e. α=0, the standard SQS
case), the per-shell, per-pair target pair count is
    T_ij^k = N_i * z_k * N_j / N
where N_i is the per-species atom count, z_k is the coordination of
shell k, and N is the total atom count. A cell is *feasible* for
exact SQS only if T_ij^k is an integer for every (i, j) and every
weighted shell — otherwise the SCRAPS fitness has a strictly
positive lower bound and the optimizer plateaus above zero.

This tool walks candidate Cell_DIM values (cubic k×k×k) near a user-
supplied target, builds the supercell, detects shells, and reports
which cells are exact-SQS-feasible and what fitness floor the
infeasible ones face. It can also diagnose a single (Cell_DIM,
ELEMENT) configuration.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from verify_sro import detect_shells, load_structure  # noqa: E402


@dataclass
class CellReport:
    cell_dim: tuple[int, int, int]
    total_atoms: int
    species_counts: tuple[int, ...]
    fitness_sq_lower_bound: float
    feasible: bool
    nonintegral_pairs: list[tuple[int, int, int, float]]  # (shell, i, j, target)
    # Anisotropy penalty: 0 for cubic (k1=k2=k3), grows with shape mismatch.
    aspect_penalty: float

    @property
    def fitness_lower_bound(self) -> float:
        return math.sqrt(self.fitness_sq_lower_bound)

    @property
    def is_cubic(self) -> bool:
        return self.cell_dim[0] == self.cell_dim[1] == self.cell_dim[2]

    def cell_dim_str(self) -> str:
        return " ".join(str(k) for k in self.cell_dim)


def integer_counts(fractions: list[float], total: int) -> Optional[tuple[int, ...]]:
    """Round fractions to integers summing to total.

    Uses largest-remainder rounding so the result respects the target
    exactly (the optimization is over assignments, not stochiometry).
    Returns None if any species would round to 0.
    """
    raw = [f * total for f in fractions]
    floors = [math.floor(x) for x in raw]
    remainder = total - sum(floors)
    if remainder < 0 or remainder > len(fractions):
        return None
    # distribute remainder to the largest fractional parts
    fracs = sorted(range(len(fractions)),
                   key=lambda i: raw[i] - floors[i],
                   reverse=True)
    counts = list(floors)
    for i in fracs[:remainder]:
        counts[i] += 1
    if any(c == 0 for c in counts):
        return None
    return tuple(counts)


def evaluate_cell(
    primitive_struct,
    cell_dim,
    species_counts: tuple[int, ...],
    n_shells: int,
    weights: list[float],
) -> CellReport:
    """Build the supercell and check shell-by-shell feasibility.

    `cell_dim` may be an int (cubic k×k×k for back-compat) or a tuple
    `(k1, k2, k3)`.
    """
    if isinstance(cell_dim, int):
        cell_dim = (cell_dim, cell_dim, cell_dim)
    k1, k2, k3 = cell_dim
    supercell = primitive_struct.copy()
    supercell.make_supercell((k1, k2, k3))
    total = len(supercell)
    if sum(species_counts) != total:
        raise ValueError(f"species_counts {species_counts} (sum {sum(species_counts)}) "
                         f"!= supercell atoms ({total})")

    # Wigner-Seitz cutoff (half min cell vector). For very anisotropic cells
    # this may include fewer shells than a cubic supercell of similar volume.
    shells = detect_shells(supercell, n_shells=n_shells)
    n_species = len(species_counts)

    fit2_lb = 0.0
    nonint: list[tuple[int, int, int, float]] = []
    for sh in shells:
        w = weights[sh.index] if sh.index < len(weights) else 1.0
        if w == 0:
            continue
        z = sh.z
        for i in range(n_species):
            for j in range(n_species):
                if i == j:
                    continue
                target = species_counts[i] * z * species_counts[j] / total
                near = round(target)
                gap = target - near
                if abs(gap) > 1e-9:
                    fit2_lb += w * (gap ** 2)
                    nonint.append((sh.index, i, j, target))
    feasible = (fit2_lb == 0.0)
    aspect = (max(cell_dim) / min(cell_dim)) - 1.0  # 0 for cubic
    return CellReport(
        cell_dim=(k1, k2, k3),
        total_atoms=total,
        species_counts=species_counts,
        fitness_sq_lower_bound=fit2_lb,
        feasible=feasible,
        nonintegral_pairs=nonint,
        aspect_penalty=aspect,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("poscar", type=Path, help="Path to the primitive/basis POSCAR")
    p.add_argument("--composition", type=str, default=None,
                   help="Target species fractions, comma-separated (e.g. 0.25,0.5,0.25)")
    p.add_argument("--element-counts", type=str, default=None,
                   help="Explicit ELEMENT counts (e.g. 27,54,27); diagnose mode")
    p.add_argument("--cell-dim", type=str, default=None,
                   help="Cell_DIM to diagnose: either 'k' (cubic) or 'k1,k2,k3' (non-cubic)")
    p.add_argument("--target-atoms", type=int, default=None,
                   help="Target supercell atom count (suggest mode)")
    p.add_argument("--shells", type=int, default=2,
                   help="Number of shells to weight (default 2)")
    p.add_argument("--weights", type=str, default=None,
                   help="Per-shell weights, comma-separated (default uniform)")
    p.add_argument("--k-range", type=str, default="2:8",
                   help="k search range as kmin:kmax (default 2:8)")
    p.add_argument("--non-cubic", action="store_true",
                   help="Also search non-cubic Cell_DIM (k1,k2,k3) shapes")
    p.add_argument("--top", type=int, default=5,
                   help="Show top N suggestions (default 5)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Suppress pymatgen warnings about non-standard POSCAR
    warnings.filterwarnings("ignore", message=".*POSCAR.*")

    # Load primitive cell (no species override; the basis order doesn't
    # matter here because we only care about how many basis atoms there are).
    struct, _ = load_structure(args.poscar)
    basis_atoms = len(struct)

    if args.weights is not None:
        weights = [float(w) for w in args.weights.split(",")]
    else:
        weights = [1.0] * args.shells

    # Diagnose mode: explicit cell + counts
    if args.cell_dim is not None and args.element_counts is not None:
        cd_parts = args.cell_dim.split(",")
        if len(cd_parts) == 1:
            cell = int(cd_parts[0])
        elif len(cd_parts) == 3:
            cell = tuple(int(x) for x in cd_parts)
        else:
            print("ERROR: --cell-dim must be 'k' or 'k1,k2,k3'", file=sys.stderr)
            return 2
        counts = tuple(int(x) for x in args.element_counts.split(","))
        report = evaluate_cell(struct, cell, counts, args.shells, weights)
        print_diagnose(report, weights)
        return 0

    # Suggest mode: composition + target atoms
    if args.composition is None or args.target_atoms is None:
        print("ERROR: need either (--cell-dim AND --element-counts) for diagnose mode,\n"
              "       or (--composition AND --target-atoms) for suggest mode.",
              file=sys.stderr)
        return 2

    fractions = [float(x) for x in args.composition.split(",")]
    if abs(sum(fractions) - 1.0) > 1e-6:
        print(f"ERROR: composition fractions sum to {sum(fractions):.6f}, not 1.0",
              file=sys.stderr)
        return 2

    kmin, kmax = (int(x) for x in args.k_range.split(":"))
    candidate_dims: list[tuple[int, int, int]] = []
    if args.non_cubic:
        # Search the unordered (k1<=k2<=k3) grid; symmetric shapes are the
        # same supercell volume so we only keep canonical ordering.
        for k1 in range(kmin, kmax + 1):
            for k2 in range(k1, kmax + 1):
                for k3 in range(k2, kmax + 1):
                    candidate_dims.append((k1, k2, k3))
    else:
        for k in range(kmin, kmax + 1):
            candidate_dims.append((k, k, k))

    reports: list[CellReport] = []
    for cell in candidate_dims:
        total = (cell[0] * cell[1] * cell[2]) * basis_atoms
        counts = integer_counts(fractions, total)
        if counts is None:
            continue
        try:
            r = evaluate_cell(struct, cell, counts, args.shells, weights)
            reports.append(r)
        except Exception as e:
            print(f"  Cell_DIM {cell}: skipped ({e})", file=sys.stderr)

    target = args.target_atoms
    # Sort by (closeness to target, then cubic-ness as tiebreaker).
    sort_key = lambda r: (abs(r.total_atoms - target), r.aspect_penalty)
    feasible = sorted([r for r in reports if r.feasible], key=sort_key)
    infeasible = sorted([r for r in reports if not r.feasible], key=sort_key)

    print(f"Composition fractions: {fractions}")
    print(f"Target atoms: {target}")
    print(f"Weighted shells: {args.shells} (weights = {weights})")
    print(f"Primitive basis atoms: {basis_atoms}")
    print()
    print("=== FEASIBLE cells (exact-SQS achievable, fitness can reach 0) ===")
    if feasible:
        for r in feasible[:args.top]:
            dn = r.total_atoms - target
            sign = "+" if dn >= 0 else ""
            tag = "" if r.is_cubic else "  [non-cubic]"
            print(f"  Cell_DIM {r.cell_dim_str()}  ->  N={r.total_atoms} ({sign}{dn} from target),  "
                  f"ELEMENT {' '.join(str(c) for c in r.species_counts)}{tag}")
    else:
        print("  (none in search range)")
    print()
    print("=== INFEASIBLE cells (fitness floor > 0, plateau guaranteed) ===")
    if infeasible:
        for r in infeasible[:args.top]:
            dn = r.total_atoms - target
            sign = "+" if dn >= 0 else ""
            tag = "" if r.is_cubic else "  [non-cubic]"
            print(f"  Cell_DIM {r.cell_dim_str()}  ->  N={r.total_atoms} ({sign}{dn}),  "
                  f"ELEMENT {' '.join(str(c) for c in r.species_counts)},  "
                  f"min fitness >= {r.fitness_lower_bound:.4f}{tag}")
    else:
        print("  (none in search range; all configurations exact-SQS achievable)")
    return 0


def print_diagnose(report: CellReport, weights: list[float]) -> None:
    print(f"Cell_DIM {report.cell_dim_str()}  (N = {report.total_atoms})")
    print(f"ELEMENT {' '.join(str(c) for c in report.species_counts)}")
    print(f"Weights: {weights}")
    print()
    if report.feasible:
        print("FEASIBLE: every weighted target pair count is integer.")
        print("Exact SQS (fitness = 0) is achievable.")
    else:
        print(f"NOT FEASIBLE: fitness floor = sqrt({report.fitness_sq_lower_bound:.4f}) "
              f"= {report.fitness_lower_bound:.4f}")
        print(f"  ({len(report.nonintegral_pairs)} non-integer target(s) below)")
        print()
        for sh, i, j, target in report.nonintegral_pairs[:10]:
            near = round(target)
            print(f"  shell {sh}: pair[{i}][{j}] target = {target:.4f} "
                  f"(nearest integer {near}, off by {target - near:+.4f})")


if __name__ == "__main__":
    sys.exit(main())
