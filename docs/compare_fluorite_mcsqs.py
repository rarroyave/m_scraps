#!/usr/bin/env python3
"""(Ca,Sr)(F,Cl) fluorite-type multi-sublattice SQS comparison:
scraps-perpair (v0.4.2 per-Wyckoff SUBLATTICE) vs ATAT mcsqs.

Fluorite is the canonical non-uniform-coordination case: cation
sublattice (Wyckoff 4a in Fm-3m, z=8 to anions at 1NN) and anion
sublattice (Wyckoff 8c, z=4 to cations + 6 to other anions at the
next shell). SCRAPs's v0.4.1 path aborted on this geometry; v0.4.2
unlocked it by generalizing the uniformity check to per-sublattice.

Mixed composition: 50/50 Ca/Sr on the cation orbit and 50/50 F/Cl
on the anion orbit, targeting an SQS (alpha = 0 on every intra-
sublattice pair). For each cell size, both tools run with the same
wall budget; we report per-sublattice max|alpha| split by
intra-sub (the chemistry that swaps can change) and inter-sub
(determined by composition).

mcsqs handles multi-sublattice rndstr.in natively via per-site
mixed occupations -- the apples-to-apples comparison for SQS works
here exactly like the B2 case in compare_b2_multisub.py.

Requires: scraps built at SCRAPs/scraps, ATAT mcsqs + str2cif on
PATH, pymatgen.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRAPS_BIN = REPO / "SCRAPs" / "scraps"
# Reuse the per-sublattice alpha calculator + the mcsqs->POSCAR parser
# already written for the B2 comparison.
sys.path.insert(0, str(REPO / "docs"))
from compare_b2_multisub import (
    compute_alpha_per_shell, fix_scraps_species, atat_to_poscar,
)

A_FLUORITE = 5.46  # Angstrom (close to CaF2 ~5.463)


def find_mpirun() -> str:
    if env := os.environ.get("MPIRUN"):
        if Path(env).exists():
            return env
    sib = Path("/opt/homebrew/bin/mpirun")
    if sib.exists():
        return str(sib)
    found = shutil.which("mpirun")
    if not found:
        raise RuntimeError("no mpirun on PATH")
    return found


MPIRUN = find_mpirun()


# ----- structures ----------------------------------------------------------

FLUORITE_POSCAR = """fluorite (4 cation + 8 anion basis)
1.0
{a:.4f} 0.0 0.0
0.0 {a:.4f} 0.0
0.0 0.0 {a:.4f}
12
Direct
0.00 0.00 0.00
0.50 0.50 0.00
0.50 0.00 0.50
0.00 0.50 0.50
0.25 0.25 0.25
0.75 0.25 0.25
0.25 0.75 0.25
0.75 0.75 0.25
0.25 0.25 0.75
0.75 0.25 0.75
0.25 0.75 0.75
0.75 0.75 0.75
"""


RNDSTR_FLUORITE = """{a:.4f} {a:.4f} {a:.4f} 90 90 90
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.00 0.00 0.00 Ca=0.5,Sr=0.5
0.50 0.50 0.00 Ca=0.5,Sr=0.5
0.50 0.00 0.50 Ca=0.5,Sr=0.5
0.00 0.50 0.50 Ca=0.5,Sr=0.5
0.25 0.25 0.25 F=0.5,Cl=0.5
0.75 0.25 0.25 F=0.5,Cl=0.5
0.25 0.75 0.25 F=0.5,Cl=0.5
0.75 0.75 0.25 F=0.5,Cl=0.5
0.25 0.25 0.75 F=0.5,Cl=0.5
0.75 0.25 0.75 F=0.5,Cl=0.5
0.25 0.75 0.75 F=0.5,Cl=0.5
0.75 0.75 0.75 F=0.5,Cl=0.5
"""


SUBLATTICE_OF = {"Ca": 0, "Sr": 0, "F": 1, "Cl": 1}


def scraps_input(cell_dim: int, mcstep: int, total_iter: int,
                 nests: int) -> str:
    n_cells = cell_dim ** 3
    half_cation = (4 * n_cells) // 2     # 4 cation basis positions per cell
    half_anion = (8 * n_cells) // 2      # 8 anion basis positions per cell
    return (
        f"TYP     4\n"
        f"GITER   1000\n"
        f"ELEMENT {half_cation} {half_cation} {half_anion} {half_anion}\n"
        f"Cell_DIM {cell_dim} {cell_dim} {cell_dim}\n"
        f"NESTS   {nests}\n"
        f"MCSTEP  {mcstep}\n"
        f"TOTALIter {total_iter}\n"
        f"MAXSHELLNUM 3\n"
        f"WEIGHT 1 1 1\n"
        f"RNG_SEED 1\n"
        f"EARLY_STOP_PATIENCE 5\n"
        f"SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1\n"
        f"SUBLATTICE 1 BASIS 4 5 6 7 8 9 10 11 SPECIES 2 3\n"
    )


def run_scraps(work_dir: Path, cell_dim: int, ranks: int,
               timeout: int) -> tuple[float, Path, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "POSCAR").write_text(FLUORITE_POSCAR.format(a=A_FLUORITE))
    n_atoms = 12 * cell_dim ** 3
    mcstep = max(500, 30 * n_atoms)
    inp = scraps_input(cell_dim, mcstep=mcstep, total_iter=30, nests=8)
    (work_dir / "INPUT").write_text(inp)
    shutil.copy(SCRAPS_BIN, work_dir / "scraps")
    os.chmod(work_dir / "scraps", 0o755)
    log_path = work_dir / "scraps.log"
    t0 = time.time()
    with log_path.open("w") as f:
        subprocess.run(
            [MPIRUN, "-np", str(ranks), "./scraps"],
            cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
        )
    wall = time.time() - t0
    return wall, work_dir / "SCRAPS.vasp", log_path.read_text()


def run_mcsqs(work_dir: Path, cell_dim: int, budget: int
              ) -> tuple[float, Path | None, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "rndstr.in").write_text(RNDSTR_FLUORITE.format(a=A_FLUORITE))
    # Cluster cutoff: cover cation-anion 1NN (~ 0.433a), anion-anion
    # 1NN (~ 0.5a), and cation-cation FCC 1NN (~ 0.707a). 0.8 * a gives
    # comfortable headroom.
    cutoff_2 = 0.8 * A_FLUORITE
    clus_log = work_dir / "clus.log"
    subprocess.run(
        ["mcsqs", f"-2={cutoff_2:.4f}"],
        cwd=work_dir, stdout=clus_log.open("w"), stderr=subprocess.STDOUT,
        timeout=30,
    )
    natom = 12 * cell_dim ** 3
    mcsqs_log = work_dir / "mcsqs.log"
    t0 = time.time()
    proc = subprocess.Popen(
        ["mcsqs", f"-n={natom}"],
        cwd=work_dir, stdout=mcsqs_log.open("w"), stderr=subprocess.STDOUT,
    )
    try:
        proc.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        (work_dir / "stopsqs").write_text("")
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    wall = time.time() - t0
    bestsqs = work_dir / "bestsqs.out"
    if not bestsqs.exists():
        return wall, None, "no bestsqs.out written (mcsqs setup may have timed out)"
    poscar = atat_to_poscar(bestsqs, work_dir / "bestsqs.poscar")
    return wall, poscar, "ok"


# ----- runner / reporting --------------------------------------------------

@dataclass
class CaseResult:
    tool: str
    cell_dim: int
    n_atoms: int
    wall_seconds: float
    shells: list = field(default_factory=list)
    notes: str = ""


def run_case(cell_dim: int, ranks: int, budget: int,
             work_root: Path) -> tuple[CaseResult, CaseResult]:
    n_atoms = 12 * cell_dim ** 3
    sc_dir = work_root / f"n{n_atoms}_scraps"
    sc_wall, sc_poscar, _ = run_scraps(
        sc_dir, cell_dim, ranks=ranks, timeout=budget + 60
    )
    fix_scraps_species(sc_poscar, ["Ca", "Sr", "F", "Cl"])
    sc_shells = compute_alpha_per_shell(
        sc_poscar, species=["Ca", "Sr", "F", "Cl"],
        sublattice_of=SUBLATTICE_OF, n_shells=3,
    )
    sc = CaseResult(
        tool="scraps-perpair", cell_dim=cell_dim, n_atoms=n_atoms,
        wall_seconds=round(sc_wall, 2), shells=sc_shells,
    )
    mc_dir = work_root / f"n{n_atoms}_mcsqs"
    mc_wall, mc_poscar, mc_note = run_mcsqs(mc_dir, cell_dim, budget=budget)
    if mc_poscar is None:
        mc = CaseResult(
            tool="mcsqs", cell_dim=cell_dim, n_atoms=n_atoms,
            wall_seconds=round(mc_wall, 2), shells=[], notes=mc_note,
        )
    else:
        mc_shells = compute_alpha_per_shell(
            mc_poscar, species=["Ca", "Sr", "F", "Cl"],
            sublattice_of=SUBLATTICE_OF, n_shells=3,
        )
        mc = CaseResult(
            tool="mcsqs", cell_dim=cell_dim, n_atoms=n_atoms,
            wall_seconds=round(mc_wall, 2), shells=mc_shells, notes=mc_note,
        )
    return sc, mc


def print_table(results: list[tuple[CaseResult, CaseResult]]) -> None:
    print()
    print("| atoms | tool          | wall (s) | "
          "max|α| intra (any shell) | max|α| inter (any shell) |")
    print("| ----- | ------------- | -------- | "
          "----------------------- | ----------------------- |")
    for sc, mc in results:
        for r in (sc, mc):
            if not r.shells:
                print(f"| {r.n_atoms:>5} | {r.tool:<13} | "
                      f"{r.wall_seconds:>8.2f} | "
                      f"{'n/a (' + r.notes + ')':<23} | "
                      f"{'n/a':<23} |")
                continue
            max_intra = max((s["max_alpha_intra"] for s in r.shells),
                            default=0.0)
            max_inter = max((s["max_alpha_inter"] for s in r.shells),
                            default=0.0)
            print(f"| {r.n_atoms:>5} | {r.tool:<13} | "
                  f"{r.wall_seconds:>8.2f} | "
                  f"{max_intra:>23.4f} | {max_inter:>23.4f} |")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2],
                   help="Cell_DIM values (NxNxN). Default 1 2 (12 and "
                        "96 atoms).")
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--budget", type=int, default=120,
                   help="Wall budget per case in seconds")
    p.add_argument("--work-root", type=Path,
                   default=Path("/tmp/fluorite_compare"))
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not SCRAPS_BIN.exists():
        print("ERROR: scraps not built. Run ./build.sh", file=sys.stderr)
        return 2
    if shutil.which("mcsqs") is None:
        print("ERROR: mcsqs not on PATH", file=sys.stderr)
        return 2
    args.work_root.mkdir(parents=True, exist_ok=True)
    results = []
    for cd in args.sizes:
        n_atoms = 12 * cd ** 3
        print(f"\n==== cell_dim={cd} (n_atoms={n_atoms}) ====", file=sys.stderr)
        sc, mc = run_case(cd, ranks=args.ranks, budget=args.budget,
                          work_root=args.work_root)
        print(f"  scraps wall = {sc.wall_seconds:.2f}s; "
              f"mcsqs wall = {mc.wall_seconds:.2f}s", file=sys.stderr)
        results.append((sc, mc))
    print_table(results)
    if args.json is not None:
        payload = {
            "lattice_param": A_FLUORITE,
            "ranks": args.ranks,
            "budget_seconds": args.budget,
            "cases": [
                {"scraps": asdict(sc), "mcsqs": asdict(mc)}
                for sc, mc in results
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"JSON summary -> {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
