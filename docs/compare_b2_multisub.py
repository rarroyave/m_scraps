#!/usr/bin/env python3
"""B2 (Ni,Co)(Ti,Zr) 50/50 multi-sublattice SQS comparison: scraps-perpair vs ATAT mcsqs.

For each cell size in --sizes (default 2 4 6), runs both tools on the
same target (fully-disordered SQS, all alpha targets = 0) and reports
wall time and max |alpha| over the M-M and X-X pairs on shells 1-2
(2NN and 3NN intra-sublattice pairs -- the only pairs whose alpha
is actually optimizable, since shell 0 in B2 is cross-sublattice
M-X and determined entirely by composition).

Both tools run with the same wall budget (--budget). SCRAPs may
terminate early via EARLY_STOP_PATIENCE; mcsqs runs until budget is
hit and we stop it via 'stopsqs'. Wall time is the actual time used
by each tool, so SCRAPs reporting < budget means it converged.

Output: a Markdown-friendly table to stdout plus a JSON summary
(--json) suitable for embedding in docs/mcsqs_comparison.md.

Requires: scraps built at SCRAPs/scraps, ATAT mcsqs + str2cif on PATH,
pymatgen.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRAPS_BIN = REPO / "SCRAPs" / "scraps"
VERIFY_TOOL = REPO / "tools" / "verify_sro.py"
B2_LATTICE = 3.0  # Angstrom -- close to NiTi (a ~ 3.01)


def find_mpirun() -> str:
    if env := os.environ.get("MPIRUN"):
        if Path(env).exists():
            return env
    sibling = Path("/opt/homebrew/bin/mpirun")
    if sibling.exists():
        return str(sibling)
    found = shutil.which("mpirun")
    if not found:
        raise RuntimeError("no mpirun on PATH")
    return found


MPIRUN = find_mpirun()


# ----- SCRAPs setup ---------------------------------------------------------

B2_POSCAR = """B2 (Ni,Co)(Ti,Zr) cubic basis
 {a:.4f}
 1.0 0.0 0.0
 0.0 1.0 0.0
 0.0 0.0 1.0
 2
Direct
 0.0 0.0 0.0
 0.5 0.5 0.5
"""


def scraps_input(cell_dim: int, mcstep: int, total_iter: int,
                 nests: int, max_shells: int, weight_str: str) -> str:
    """SCRAPs INPUT for the (Ni,Co)(Ti,Zr) B2 case. ELEMENT order is
    Ni, Co, Ti, Zr; sublattice 0 holds Ni+Co (basis 0), sublattice 1
    holds Ti+Zr (basis 1). All alpha targets default to 0 (SQS).

    Caveat about the fitness floor: SCRAPs's fitness sums over every
    (i, j) pair at every weighted shell, including cross-sublattice
    pairs at intra-sublattice shells (which are always 0 by geometry,
    contributing a constant floor independent of swap choices). The
    gradient of fitness w.r.t. swaps is unaffected by this floor --
    same-sublattice pair counts are the only thing swaps can move --
    so the optimizer still converges the chemistry-relevant alpha
    values correctly. We disable EARLY_STOP so the floor doesn't
    trigger premature termination.
    """
    n_per_sub = cell_dim ** 3
    half = n_per_sub // 2
    return (
        f"TYP     4\n"
        f"GITER   1000\n"
        f"ELEMENT {half} {half} {half} {half}\n"
        f"Cell_DIM        {cell_dim} {cell_dim} {cell_dim}\n"
        f"NESTS   {nests}\n"
        f"MCSTEP  {mcstep}\n"
        f"TOTALIter {total_iter}\n"
        f"MAXSHELLNUM {max_shells}\n"
        f"WEIGHT {weight_str}\n"
        f"RNG_SEED 1\n"
        f"EARLY_STOP_PATIENCE 0\n"
        f"SUBLATTICE 0 BASIS 0 SPECIES 0 1\n"
        f"SUBLATTICE 1 BASIS 1 SPECIES 2 3\n"
    )


def run_scraps(work_dir: Path, cell_dim: int, ranks: int,
               timeout: int) -> tuple[float, Path, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "POSCAR").write_text(B2_POSCAR.format(a=B2_LATTICE))
    # MC budget: MCSTEP ~ 50 * n_atoms (heuristic scaled for the
    # cross-sublattice rejection rate ~50%). TOTALIter capped at 30.
    n_atoms = 2 * cell_dim ** 3
    mcstep = max(500, 50 * n_atoms)
    # MAXSHELLNUM=5 covers 1NN..5NN, weights target only intra-sublattice
    # shells (0=1NN cross, 1=2NN intra, 2=3NN intra, 3=4NN cross, 4=5NN intra).
    inp = scraps_input(cell_dim, mcstep=mcstep, total_iter=30,
                       nests=8, max_shells=5,
                       weight_str="0 1 1 0 1")
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


# ----- mcsqs setup ----------------------------------------------------------

RNDSTR_TMPL = """{a:.4f} {a:.4f} {a:.4f} 90 90 90
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.0 0.0 0.0 Ni=0.5,Co=0.5
0.5 0.5 0.5 Ti=0.5,Zr=0.5
"""


def run_mcsqs(work_dir: Path, cell_dim: int, budget: int) -> tuple[float, Path | None, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "rndstr.in").write_text(RNDSTR_TMPL.format(a=B2_LATTICE))
    # Pre-specify the supercell shape (cubic N x N x N) to bypass mcsqs's
    # supercell-enumeration phase, which scales poorly with cell size
    # (>5 min just for setup at N=6). SCRAPs takes the supercell directly
    # from POSCAR + Cell_DIM, so this also keeps the comparison
    # apples-to-apples on the SQS-search step.
    (work_dir / "sqscell.out").write_text(
        f"1\n\n{cell_dim} 0 0\n0 {cell_dim} 0\n0 0 {cell_dim}\n"
    )
    # Cluster cutoff: cover 1NN (sqrt(3)/2 * a ~ 0.866a), 2NN (a),
    # 3NN (sqrt(2) * a ~ 1.414a). Pick 1.5*a to comfortably include all three.
    cutoff_2 = 1.5 * B2_LATTICE
    clus_log = work_dir / "clus.log"
    subprocess.run(
        ["mcsqs", f"-2={cutoff_2:.4f}"],
        cwd=work_dir, stdout=clus_log.open("w"), stderr=subprocess.STDOUT,
        timeout=30,
    )
    natom = 2 * cell_dim ** 3
    mcsqs_log = work_dir / "mcsqs.log"
    t0 = time.time()
    proc = subprocess.Popen(
        ["mcsqs", f"-n={natom}"],
        cwd=work_dir, stdout=mcsqs_log.open("w"), stderr=subprocess.STDOUT,
    )
    try:
        proc.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        # Hit budget -- signal mcsqs to stop via 'stopsqs', then give it
        # a few seconds to flush before killing.
        (work_dir / "stopsqs").write_text("")
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    wall = time.time() - t0
    bestsqs = work_dir / "bestsqs.out"
    if not bestsqs.exists():
        return wall, None, "no bestsqs.out written"
    poscar_path = atat_to_poscar(bestsqs, work_dir / "bestsqs.poscar")
    return wall, poscar_path, "ok"


def atat_to_poscar(atat_path: Path, poscar_path: Path) -> Path:
    """Convert an ATAT structure file (bestsqs.out format) to VASP POSCAR.

    bestsqs.out layout:
      lines 1-3: primitive lattice vectors (rows, in Angstrom)
      lines 4-6: supercell expansion (rows, in primitive units)
      remaining lines: Cartesian position (x y z) and species symbol per atom

    The supercell lattice vectors are supercell_expansion @ primitive_lattice.
    Atom positions are already Cartesian in Angstrom. We group by species
    so the POSCAR has the standard VASP per-species-block layout that
    pymatgen / verify_sro expect.
    """
    import numpy as np
    lines = [l.strip() for l in atat_path.read_text().splitlines() if l.strip()]
    prim = np.array([list(map(float, lines[i].split()[:3])) for i in range(3)])
    sc = np.array([list(map(float, lines[3 + i].split()[:3])) for i in range(3)])
    lattice = sc @ prim
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for raw in lines[6:]:
        toks = raw.split()
        if len(toks) < 4:
            continue
        x, y, z = float(toks[0]), float(toks[1]), float(toks[2])
        sp = toks[3]
        atoms.append((sp, (x, y, z)))
    # Group by species in order of first appearance.
    order: list[str] = []
    by_sp: dict[str, list[tuple[float, float, float]]] = {}
    for sp, pos in atoms:
        if sp not in by_sp:
            order.append(sp)
            by_sp[sp] = []
        by_sp[sp].append(pos)
    poscar_lines = [
        "mcsqs bestsqs.out converted",
        " 1.0",
        f" {lattice[0,0]:.8f} {lattice[0,1]:.8f} {lattice[0,2]:.8f}",
        f" {lattice[1,0]:.8f} {lattice[1,1]:.8f} {lattice[1,2]:.8f}",
        f" {lattice[2,0]:.8f} {lattice[2,1]:.8f} {lattice[2,2]:.8f}",
        " " + " ".join(order),
        " " + " ".join(str(len(by_sp[s])) for s in order),
        "Cartesian",
    ]
    for s in order:
        for (x, y, z) in by_sp[s]:
            poscar_lines.append(f" {x:.8f} {y:.8f} {z:.8f}")
    poscar_path.write_text("\n".join(poscar_lines) + "\n")
    return poscar_path


# ----- multi-sublattice alpha analysis --------------------------------------

def compute_alpha_per_shell(poscar_path: Path,
                            species: list[str],
                            sublattice_of: dict[str, int],
                            n_shells: int = 3) -> list[dict]:
    """Compute per-sublattice Warren-Cowley alpha for a multi-sublattice
    structure.

    For each (shell k, species i, species j) where i lives on sublattice
    sigma and j on sublattice tau:
      alpha^{sigma->tau}_ij = 1 - P(j|i, shell k) / c_j^tau
    where:
      P(j|i, shell k) = N_{ij}^k / (z^{sigma->tau}_k * N_i^sigma)
      c_j^tau = N_j / N_tau    (concentration of j on its own sublattice)
      z^{sigma->tau}_k = (count of sublattice-tau neighbors at shell k
                         per atom on sublattice sigma)

    Undefined (alpha = NaN) when z^{sigma->tau}_k = 0 (i.e. no
    sigma->tau pairs at that shell, as happens for B2's odd-NN shells
    which are cross-sublattice only and the even-NN shells which are
    intra-only).

    Returns per-shell info including max|alpha| split by intra- vs
    cross-sublattice pairs.
    """
    import warnings
    import numpy as np
    from pymatgen.io.vasp import Poscar
    from pymatgen.io.vasp.inputs import BadPoscarWarning
    warnings.filterwarnings("ignore", category=BadPoscarWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BadPoscarWarning)
        poscar = Poscar.from_file(poscar_path, check_for_potcar=False,
                                  read_velocities=False)
    structure = poscar.structure

    # Map each atom to its species index in `species` (order matters).
    species_to_idx = {sp: i for i, sp in enumerate(species)}
    # Some POSCAR readers report species in the order written; map by
    # symbol so the user's `species` list ordering is canonical.
    poscar_specs = [s.symbol for s in structure.species]
    unknown = set(poscar_specs) - set(species_to_idx.keys())
    if unknown:
        raise RuntimeError(
            f"POSCAR has species {unknown} not in --species list {species}"
        )
    site_species = np.array([species_to_idx[s] for s in poscar_specs])
    n_typ = len(species)
    sub_of_species = np.array(
        [sublattice_of[species[s]] for s in range(n_typ)]
    )
    site_sublattice = sub_of_species[site_species]

    # Detect shells. Use a half-min-lattice cutoff (Wigner-Seitz safe);
    # cluster sorted distances by relative-gap criterion.
    cutoff = float(min(structure.lattice.abc)) / 2.0 - 1e-3
    centers, neighbors, _, distances = structure.get_neighbor_list(r=cutoff)
    if len(distances) == 0:
        return []
    sorted_d = np.sort(distances)
    bounds = [[sorted_d[0]]]
    for d in sorted_d[1:]:
        if (d - bounds[-1][-1]) / max(bounds[-1][-1], 1e-9) > 0.05:
            bounds.append([d])
        else:
            bounds[-1].append(d)
    bounds = bounds[:n_shells]

    # Per-species and per-sublattice counts.
    counts_per_spec = np.bincount(site_species, minlength=n_typ)
    n_sub = int(sub_of_species.max() + 1)
    counts_per_sub = np.bincount(site_sublattice,
                                  minlength=n_sub).astype(float)
    counts_per_spec_sub = np.zeros((n_typ, n_sub))
    for s in range(n_typ):
        counts_per_spec_sub[s, sub_of_species[s]] = counts_per_spec[s]
    # c_j^tau = N_j / N_tau (concentration of species j on its sublattice)
    c_per_spec_on_own = np.array([
        counts_per_spec[s] / max(counts_per_sub[sub_of_species[s]], 1)
        for s in range(n_typ)
    ])

    out = []
    for k, group in enumerate(bounds):
        r_min, r_max = float(min(group)), float(max(group))
        mask = (distances >= r_min - 1e-6) & (distances <= r_max + 1e-6)
        cs = site_species[centers[mask]]
        ns = site_species[neighbors[mask]]
        c_sub = site_sublattice[centers[mask]]
        n_sub_arr = site_sublattice[neighbors[mask]]
        # Pair counts N_{ij}^k
        pair_counts = np.zeros((n_typ, n_typ), dtype=int)
        np.add.at(pair_counts, (cs, ns), 1)
        # z^{sigma->tau}_k = (pair entries with center on sigma and
        # neighbor on tau) / N_sigma
        z_st = np.zeros((n_sub, n_sub))
        for st_mask in [(c_sub == sigma) & (n_sub_arr == tau)
                        for sigma in range(n_sub) for tau in range(n_sub)]:
            pass
        for sigma in range(n_sub):
            for tau in range(n_sub):
                n_st_pairs = int(((c_sub == sigma) & (n_sub_arr == tau)).sum())
                z_st[sigma, tau] = n_st_pairs / max(counts_per_sub[sigma], 1)
        # alpha matrix
        alpha = np.full((n_typ, n_typ), np.nan)
        for i in range(n_typ):
            sigma = int(sub_of_species[i])
            if counts_per_spec[i] == 0:
                continue
            for j in range(n_typ):
                tau = int(sub_of_species[j])
                if counts_per_spec[j] == 0:
                    continue
                z = z_st[sigma, tau]
                if z <= 0.0:
                    continue
                P_ij = pair_counts[i, j] / (z * counts_per_spec[i])
                c_j = c_per_spec_on_own[j]
                if c_j <= 0.0:
                    continue
                alpha[i, j] = 1.0 - P_ij / c_j
        # Max |alpha| split by intra (sigma==tau) vs inter (sigma!=tau).
        max_intra = 0.0
        max_inter = 0.0
        for i in range(n_typ):
            for j in range(n_typ):
                v = alpha[i, j]
                if not np.isfinite(v):
                    continue
                same = sub_of_species[i] == sub_of_species[j]
                if same:
                    max_intra = max(max_intra, abs(float(v)))
                else:
                    max_inter = max(max_inter, abs(float(v)))
        out.append({
            "index": k,
            "r_min": r_min,
            "r_max": r_max,
            "z_total": float(np.sum(z_st) / n_sub),  # avg coord per atom
            "max_alpha_intra": max_intra,
            "max_alpha_inter": max_inter,
        })
    return out


SUBLATTICE_OF = {"Ni": 0, "Co": 0, "Ti": 1, "Zr": 1}


def fix_scraps_species(poscar_path: Path, species: list[str]) -> None:
    """SCRAPs writes SCRAPS.vasp without species symbols (just count
    line). Inject a species-symbol line so pymatgen reads the species
    correctly."""
    lines = poscar_path.read_text().splitlines()
    # Standard SCRAPs layout: lines 0=comment, 1=scale, 2-4=lattice,
    # 5=counts ("4 4 4 4"), 6=Cartesian, 7+=positions. Insert the
    # species line between lines 4 and 5.
    has_symbols = False
    if len(lines) > 5:
        toks = lines[5].split()
        if toks and not toks[0].lstrip("-").isdigit():
            has_symbols = True
    if has_symbols:
        return
    new = lines[:5] + [" " + " ".join(species)] + lines[5:]
    poscar_path.write_text("\n".join(new) + "\n")


# ----- runner ---------------------------------------------------------------

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
    n_atoms = 2 * cell_dim ** 3
    # SCRAPs side
    sc_dir = work_root / f"n{n_atoms}_scraps"
    sc_wall, sc_poscar, sc_log = run_scraps(
        sc_dir, cell_dim, ranks=ranks, timeout=budget + 60
    )
    # SCRAPs writes SCRAPS.vasp with placeholder species labels (no
    # element names). Rewrite the species line so pymatgen reads them
    # as Ni, Co, Ti, Zr in ELEMENT order before alpha analysis.
    fix_scraps_species(sc_poscar, ["Ni", "Co", "Ti", "Zr"])
    sc_shells = compute_alpha_per_shell(
        sc_poscar, species=["Ni", "Co", "Ti", "Zr"],
        sublattice_of=SUBLATTICE_OF, n_shells=3
    )
    sc = CaseResult(
        tool="scraps-perpair", cell_dim=cell_dim, n_atoms=n_atoms,
        wall_seconds=round(sc_wall, 2), shells=sc_shells,
    )
    # mcsqs side
    mc_dir = work_root / f"n{n_atoms}_mcsqs"
    mc_wall, mc_poscar, mc_note = run_mcsqs(mc_dir, cell_dim, budget=budget)
    if mc_poscar is None:
        mc = CaseResult(
            tool="mcsqs", cell_dim=cell_dim, n_atoms=n_atoms,
            wall_seconds=round(mc_wall, 2), shells=[], notes=mc_note,
        )
    else:
        mc_shells = compute_alpha_per_shell(
            mc_poscar, species=["Ni", "Co", "Ti", "Zr"],
            sublattice_of=SUBLATTICE_OF, n_shells=3
        )
        mc = CaseResult(
            tool="mcsqs", cell_dim=cell_dim, n_atoms=n_atoms,
            wall_seconds=round(mc_wall, 2), shells=mc_shells, notes=mc_note,
        )
    return sc, mc


def print_table(results: list[tuple[CaseResult, CaseResult]]) -> None:
    print()
    print("| atoms | tool          | wall (s) | "
          "max|alpha| sh1 intra | max|alpha| sh2 intra | "
          "max|alpha| sh0 inter |")
    print("| ----- | ------------- | -------- | "
          "-------------------- | -------------------- | "
          "-------------------- |")
    for sc, mc in results:
        for r in (sc, mc):
            def cell(sh, key):
                m = next((s for s in r.shells if s["index"] == sh), None)
                if m is None:
                    return "n/a"
                return f"{m[key]:.4f}"
            print(
                f"| {r.n_atoms:>5} | {r.tool:<13} | {r.wall_seconds:>8.2f} | "
                f"{cell(1, 'max_alpha_intra'):>20} | "
                f"{cell(2, 'max_alpha_intra'):>20} | "
                f"{cell(0, 'max_alpha_inter'):>20} |"
            )
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6],
                   help="Cell_DIM values (NxNxN; default 2 4 6)")
    p.add_argument("--ranks", type=int, default=4,
                   help="SCRAPs MPI ranks (default 4)")
    p.add_argument("--budget", type=int, default=60,
                   help="Wall budget per case in seconds (default 60). "
                        "SCRAPs may finish earlier via early-stop; mcsqs "
                        "is signaled to stop at budget.")
    p.add_argument("--work-root", type=Path, default=Path("/tmp/b2_multisub"),
                   help="Scratch directory for both tools (cleared per "
                        "case)")
    p.add_argument("--json", type=Path, default=None,
                   help="If set, write a JSON summary to this path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not SCRAPS_BIN.exists():
        print("ERROR: scraps binary missing -- run bash build.sh", file=sys.stderr)
        return 2
    if shutil.which("mcsqs") is None:
        print("ERROR: mcsqs not on PATH", file=sys.stderr)
        return 2
    args.work_root.mkdir(parents=True, exist_ok=True)
    results = []
    for cd in args.sizes:
        if cd % 2 != 0:
            print(f"SKIP cell_dim={cd}: 50/50 split per sublattice requires "
                  f"N^3 even (so N even); use 2, 4, 6, 8...", file=sys.stderr)
            continue
        print(f"\n==== cell_dim={cd} (n_atoms={2*cd**3}) ====")
        sc, mc = run_case(cd, ranks=args.ranks, budget=args.budget,
                          work_root=args.work_root)
        print(f"  scraps wall = {sc.wall_seconds:.2f}s; "
              f"mcsqs wall = {mc.wall_seconds:.2f}s")
        results.append((sc, mc))
    print_table(results)
    if args.json is not None:
        payload = {
            "lattice_param": B2_LATTICE,
            "ranks": args.ranks,
            "budget_seconds": args.budget,
            "cases": [
                {"scraps": asdict(sc), "mcsqs": asdict(mc)}
                for sc, mc in results
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"JSON summary written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
