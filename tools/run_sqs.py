#!/usr/bin/env python3
"""High-level SCRAPS launcher: composition + target atoms -> SQS.

Composes the three feasibility / convergence heuristics into a single
opinionated wrapper:

  1. Calls suggest_cell.py to pick the smallest fully-feasible
     Cell_DIM near the requested atom count (falls back to closest
     infeasible if no feasible option exists in the search range).
  2. Generates INPUT with AUTO_MCSTEP and EARLY_STOP_PATIENCE so the
     run finishes as soon as the cuckoo search plateaus.
  3. Launches `mpirun -np <ranks> scraps` and prints the FINAL
     fitness, achieved max|alpha|, and wall time.

For full control, write your own INPUT — this wrapper exists for the
"give me an SQS at this composition" case.

Usage:
    tools/run_sqs.py POSCAR --composition 0.35,0.35,0.30 \\
        --target-atoms 200 --shells 2 [--ranks 4] [--work-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from suggest_cell import (  # noqa: E402
    evaluate_cell,
    integer_counts,
    parse_spectator_spec,
)
from verify_sro import load_structure  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("poscar", type=Path, help="Primitive POSCAR (single unit cell + basis)")
    p.add_argument("--composition", required=True,
                   help="Species fractions, comma-separated (e.g. 0.35,0.35,0.30)")
    p.add_argument("--target-atoms", type=int, required=True,
                   help="Target supercell atom count")
    p.add_argument("--shells", type=int, default=2,
                   help="Number of shells to weight (default 2)")
    p.add_argument("--weights", type=str, default=None,
                   help="Per-shell weights, comma-separated (default uniform)")
    p.add_argument("--k-range", type=str, default="2:8",
                   help="k search range for Cell_DIM (default 2:8)")
    p.add_argument("--non-cubic", action="store_true", default=True,
                   help="Allow non-cubic supercell shapes (default: on)")
    p.add_argument("--cubic-only", action="store_true",
                   help="Force cubic k*k*k supercell")
    p.add_argument("--spectator-basis-species", type=str, default="",
                   help="HEB-style spectator sublattice spec: 'basis_idx:species_idx,...' "
                        "(e.g. '1:5,2:5' for AlB2 with B as species 5 on basis 1,2). "
                        "When set, --composition lists fractions for ACTIVE species only.")
    p.add_argument("--ranks", type=int, default=4,
                   help="MPI ranks for mpirun (default 4)")
    p.add_argument("--nests", type=int, default=24,
                   help="Number of nests (default 24)")
    p.add_argument("--total-iter", type=int, default=20,
                   help="Outer iterations (default 20; early-stop usually triggers sooner)")
    p.add_argument("--mcstep-per-atom", type=int, default=7,
                   help="MCSTEP / N ratio for AUTO_MCSTEP (default 7)")
    p.add_argument("--patience", type=int, default=3,
                   help="EARLY_STOP_PATIENCE (default 3, 0 disables)")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="EARLY_STOP_TOL (default 1e-6)")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Working directory (default: ./run_sqs_<timestamp>)")
    p.add_argument("--scraps", type=Path,
                   default=REPO_ROOT / "SCRAPs" / "scraps",
                   help="Path to scraps binary")
    p.add_argument("--mpirun", type=str, default="mpirun",
                   help="mpirun executable (default mpirun)")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate the work directory and INPUT but skip mpirun")
    return p.parse_args()


def pick_cell(struct, fractions, target_atoms, n_shells, weights, k_range,
              cubic_only, spectator_pairs):
    """Return the best CellReport given the search options.

    In spectator mode, `fractions` are active-sublattice fractions and
    we attach the spectator allocations on top to build a full
    species_counts tuple before evaluating.
    """
    kmin, kmax = (int(x) for x in k_range.split(":"))
    candidate_dims: list[tuple[int, int, int]] = []
    if cubic_only:
        for k in range(kmin, kmax + 1):
            candidate_dims.append((k, k, k))
    else:
        for k1 in range(kmin, kmax + 1):
            for k2 in range(k1, kmax + 1):
                for k3 in range(k2, kmax + 1):
                    candidate_dims.append((k1, k2, k3))

    basis_atoms = len(struct)
    spectator_basis = [bi for bi, _ in spectator_pairs]
    n_species_active = len(fractions)
    if spectator_pairs:
        spectator_species_set = sorted({sp for _, sp in spectator_pairs})
        n_species_total = max(n_species_active, max(spectator_species_set) + 1)
    else:
        n_species_total = n_species_active

    reports = []
    for cell in candidate_dims:
        cell_volume = cell[0] * cell[1] * cell[2]
        total = cell_volume * basis_atoms
        n_active = (basis_atoms - len(spectator_basis)) * cell_volume
        if n_active <= 0:
            continue
        active_counts = integer_counts(fractions, n_active)
        if active_counts is None:
            continue
        if spectator_pairs:
            counts_full = list(active_counts) + [0] * (n_species_total - n_species_active)
            for _, sp in spectator_pairs:
                counts_full[sp] += cell_volume
            counts = tuple(counts_full)
        else:
            counts = active_counts
        try:
            r = evaluate_cell(struct, cell, counts, n_shells, weights,
                              spectator_basis_species=spectator_pairs)
            reports.append(r)
        except Exception:
            continue

    if not reports:
        return None

    sort_key = lambda r: (abs(r.total_atoms - target_atoms), r.aspect_penalty)
    feasible = sorted([r for r in reports if r.feasible], key=sort_key)
    infeasible = sorted([r for r in reports if not r.feasible], key=sort_key)
    return feasible[0] if feasible else infeasible[0]


def write_input(path: Path, report, fractions, n_shells, weights,
                nests, total_iter, mcstep_per_atom, patience, tol,
                spectator_pairs: list[tuple[int, int]] = None) -> None:
    spectator_pairs = spectator_pairs or []
    typeofatoms = len(report.species_counts)
    weights_str = " ".join(f"{w:g}" for w in weights[:n_shells])
    element_str = " ".join(str(c) for c in report.species_counts)
    cell_dim_str = " ".join(str(k) for k in report.cell_dim)
    feasibility_note = (
        f"# FEASIBLE: exact-SQS achievable (fitness can reach 0)."
        if report.feasible
        else f"# INFEASIBLE: fitness floor >= {report.fitness_lower_bound:.4f} "
             f"(see --shells {n_shells} integer constraints)."
    )
    spectator_lines = ""
    if spectator_pairs:
        spectator_lines = "\n".join(
            f"SPECTATOR_BASIS_SPECIES {bi} {sp}" for bi, sp in spectator_pairs
        ) + "\n"
    body = f"""TYP     {typeofatoms}
GITER   1000
ELEMENT {element_str}
Cell_DIM        {cell_dim_str}
NESTS   {nests}
MCSTEP  100
AUTO_MCSTEP {mcstep_per_atom}
TOTALIter {total_iter}
MAXSHELLNUM {n_shells}
WEIGHT {weights_str}
EARLY_STOP_PATIENCE {patience}
EARLY_STOP_TOL {tol:g}
{spectator_lines}
# Auto-generated by tools/run_sqs.py
# Composition target (active species): {','.join(f'{f:g}' for f in fractions)}
{feasibility_note}
# MCSTEP is overridden by AUTO_MCSTEP at runtime: max(50, {mcstep_per_atom} * N).
# Early-stop triggers after {patience} iterations with no improvement > {tol:g}.
"""
    path.write_text(body)


def main() -> int:
    warnings.filterwarnings("ignore")
    args = parse_args()

    fractions = [float(x) for x in args.composition.split(",")]
    if abs(sum(fractions) - 1.0) > 1e-6:
        print(f"ERROR: composition fractions sum to {sum(fractions):.6f}, not 1.0",
              file=sys.stderr)
        return 2

    if args.weights is not None:
        weights = [float(w) for w in args.weights.split(",")]
    else:
        weights = [1.0] * args.shells

    struct, _ = load_structure(args.poscar)
    spectator_pairs = parse_spectator_spec(args.spectator_basis_species)

    print(f"Step 1/3: picking Cell_DIM near target atoms={args.target_atoms}")
    if spectator_pairs:
        print(f"  Spectator sublattice: {spectator_pairs} "
              f"(active composition only in --composition)")
    cubic_only = args.cubic_only
    report = pick_cell(struct, fractions, args.target_atoms,
                       args.shells, weights, args.k_range, cubic_only,
                       spectator_pairs)
    if report is None:
        print(f"ERROR: no valid Cell_DIM in range {args.k_range} for "
              f"composition {fractions} and basis {len(struct)}",
              file=sys.stderr)
        return 2

    cell_str = " x ".join(str(k) for k in report.cell_dim)
    tag = "" if report.is_cubic else " [non-cubic]"
    print(f"  -> Cell_DIM {cell_str} = {report.total_atoms} atoms"
          f"  ELEMENT {' '.join(str(c) for c in report.species_counts)}{tag}")
    if report.feasible:
        print(f"  -> FEASIBLE (exact SQS achievable)")
    else:
        print(f"  -> INFEASIBLE: fitness floor >= {report.fitness_lower_bound:.4f}")

    # Set up work dir
    if args.work_dir is None:
        args.work_dir = Path.cwd() / f"run_sqs_{int(time.time())}"
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Step 2/3: preparing work dir {args.work_dir}")
    shutil.copy(args.poscar, args.work_dir / "POSCAR")
    if not args.scraps.exists():
        print(f"ERROR: scraps binary not found at {args.scraps}; "
              f"build with ./build.sh", file=sys.stderr)
        return 2
    shutil.copy(args.scraps, args.work_dir / "scraps")
    os.chmod(args.work_dir / "scraps", 0o755)
    write_input(args.work_dir / "INPUT", report, fractions, args.shells,
                weights, args.nests, args.total_iter,
                args.mcstep_per_atom, args.patience, args.tol,
                spectator_pairs=spectator_pairs)

    if args.dry_run:
        print(f"  (dry run; INPUT and POSCAR are in {args.work_dir})")
        return 0

    cmd = [args.mpirun, "-np", str(args.ranks), "./scraps"]
    print(f"Step 3/3: {' '.join(cmd)}  (cwd={args.work_dir})")
    t0 = time.time()
    log = args.work_dir / "scraps.log"
    with log.open("w") as f:
        rc = subprocess.run(cmd, cwd=args.work_dir, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0:
        print(f"ERROR: scraps exited with code {rc}; see {log}", file=sys.stderr)
        return rc

    # Parse output
    final_fit = None
    for line in log.read_text().splitlines():
        if line.startswith("FINAL fitness"):
            try:
                final_fit = float(line.split("=")[-1].strip())
            except Exception:
                pass
            break

    print()
    print(f"  Wall time: {wall:.1f} s")
    print(f"  FINAL fitness: {final_fit}")
    print(f"  Output:        {args.work_dir / 'SCRAPS.vasp'}")
    print(f"  Log:           {log}")
    if (args.work_dir / "SCRAPS.vasp").exists():
        n_species_total = len(report.species_counts)
        species = ",".join(chr(ord('A') + i) for i in range(n_species_total))
        verify_cmd = [
            sys.executable, str(THIS_DIR / "verify_sro.py"),
            str(args.work_dir / "SCRAPS.vasp"),
            "--species", species, "--shells", str(args.shells),
        ]
        if spectator_pairs:
            spec_species_letters = ",".join(
                chr(ord('A') + sp) for _, sp in spectator_pairs
            )
            verify_cmd += ["--spectator-species", spec_species_letters]
        print(f"  Verify with:   {' '.join(verify_cmd)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
