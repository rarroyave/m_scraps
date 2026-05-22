#!/usr/bin/env python3
"""Multi-orbit non-uniform-coordination SQS comparison: scraps-perpair
(v0.4.2 per-Wyckoff SUBLATTICE) vs ATAT mcsqs.

Generalization of compare_fluorite_mcsqs.py to handle several distinct
structure classes via a --structure CLI flag. Each structure is
defined by:
  - space group + Wyckoff generator coords (pymatgen builds the cell)
  - per-orbit binary mixed-species pair (the chemistry to design)
  - mcsqs cluster cutoff (multiple of the lattice parameter)

Currently supported:
  - fluorite (CaF2-type, Fd-3m): 4a + 8c, z=8 vs z=4
  - laves_c15 (MgCu2-type, Fd-3m): 8a + 16d
  - heusler_l21 (Cu2MnAl-type, Fm-3m): 4a + 4b + 8c

The cation/anion partition is encoded in `sublattice_of` per structure;
mcsqs rndstr.in is generated from the fully expanded pymatgen Structure
(one line per atom with per-site mixed occupation), matching mcsqs's
multi-sublattice native syntax.

Output: per-sublattice max|alpha| split by intra-sub vs inter-sub for
each tool, plus wall time. JSON output for downstream embedding.

Requires: scraps built at SCRAPs/scraps, ATAT mcsqs on PATH, pymatgen.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRAPS_BIN = REPO / "SCRAPs" / "scraps"
sys.path.insert(0, str(REPO / "docs"))
from compare_b2_multisub import (
    compute_alpha_per_shell, fix_scraps_species, atat_to_poscar,
)


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


# ----- structure specs ------------------------------------------------------

@dataclass
class Orbit:
    """One Wyckoff orbit: a pymatgen-from-spacegroup generator + the
    chemistry to design on that orbit (binary 50/50)."""
    generator_label: str           # element symbol for from_spacegroup
    coords: list[float]            # one set of fractional coords
    mixed_pair: tuple[str, str]    # the (A, B) binary mixture
    sublattice_id: int             # which SUBLATTICE this becomes in INPUT


@dataclass
class StructureSpec:
    name: str
    space_group: str
    a: float                        # cubic lattice parameter (Angstrom)
    orbits: list[Orbit]
    cluster_cutoff_factor: float    # multiple of `a` for mcsqs -2
    # Optional override for the species order in SCRAPs ELEMENT (sometimes
    # we want a specific ordering for verify_sro / alpha-calc plumbing).
    species_order: list[str] = field(default_factory=list)


def all_species(spec: StructureSpec) -> list[str]:
    if spec.species_order:
        return spec.species_order
    out: list[str] = []
    for orb in spec.orbits:
        for s in orb.mixed_pair:
            if s not in out:
                out.append(s)
    return out


def sublattice_of(spec: StructureSpec) -> dict[str, int]:
    out: dict[str, int] = {}
    for orb in spec.orbits:
        for s in orb.mixed_pair:
            out[s] = orb.sublattice_id
    return out


STRUCTURES: dict[str, StructureSpec] = {
    "fluorite": StructureSpec(
        name="fluorite",
        space_group="Fd-3m",
        a=5.46,
        orbits=[
            Orbit("Mg", [0.0, 0.0, 0.0],     ("Ca", "Sr"), sublattice_id=0),
            Orbit("Cu", [0.625, 0.625, 0.625], ("F",  "Cl"), sublattice_id=1),
        ],
        cluster_cutoff_factor=0.80,
    ),
    "laves_c15": StructureSpec(
        name="laves_c15",
        space_group="Fd-3m",
        a=7.05,
        orbits=[
            Orbit("Mg", [0.0, 0.0, 0.0],     ("Mg", "Ca"), sublattice_id=0),
            Orbit("Cu", [0.625, 0.625, 0.625], ("Cu", "Ni"), sublattice_id=1),
        ],
        cluster_cutoff_factor=0.50,
    ),
    "heusler_l21": StructureSpec(
        name="heusler_l21",
        space_group="Fm-3m",
        a=5.95,
        orbits=[
            Orbit("Mn", [0.0,  0.0,  0.0 ], ("Mn", "Fe"), sublattice_id=0),
            Orbit("Al", [0.5,  0.5,  0.5 ], ("Al", "Ga"), sublattice_id=1),
            Orbit("Cu", [0.25, 0.25, 0.25], ("Cu", "Au"), sublattice_id=2),
        ],
        cluster_cutoff_factor=0.60,
    ),
}


# ----- POSCAR + rndstr.in generation ---------------------------------------

def build_structure(spec: StructureSpec):
    """pymatgen Structure expanded from the space-group + Wyckoff
    generators. Used both to build the SCRAPs POSCAR (with the
    generator labels distinguishing orbits) and to walk per-atom
    coordinates for the mcsqs rndstr.in."""
    pymatgen = __import__("pymatgen.core", fromlist=["Lattice", "Structure"])
    Lattice = pymatgen.Lattice
    Structure = pymatgen.Structure
    lat = Lattice.cubic(spec.a)
    return Structure.from_spacegroup(
        spec.space_group, lat,
        species=[o.generator_label for o in spec.orbits],
        coords=[o.coords for o in spec.orbits],
    )


def write_scraps_poscar(spec: StructureSpec, path: Path) -> list[str]:
    """Write a SCRAPs-compatible POSCAR with the generator labels in
    each species block. Returns the per-site species list so the caller
    can compute basis groupings."""
    struct = build_structure(spec)
    struct.to(filename=str(path), fmt="poscar")
    return [s.symbol for s in struct.species]


def write_mcsqs_rndstr(spec: StructureSpec, path: Path) -> None:
    """Write an mcsqs rndstr.in with per-atom mixed occupations.
    Each pymatgen-expanded site gets the binary mixed-pair string for
    its orbit. The cell is the conventional unit cell (cubic axes,
    angles 90)."""
    struct = build_structure(spec)
    site_species = [s.symbol for s in struct.species]
    # Map orbit label -> mixed-pair string for rndstr.in
    occ_str = {
        orb.generator_label: f"{orb.mixed_pair[0]}=0.5,{orb.mixed_pair[1]}=0.5"
        for orb in spec.orbits
    }
    lines = [
        f"{spec.a:.4f} {spec.a:.4f} {spec.a:.4f} 90 90 90",
        "1.0 0.0 0.0",
        "0.0 1.0 0.0",
        "0.0 0.0 1.0",
    ]
    for site, sp in zip(struct, site_species):
        fc = site.frac_coords
        lines.append(
            f"{fc[0]:.6f} {fc[1]:.6f} {fc[2]:.6f} {occ_str[sp]}"
        )
    path.write_text("\n".join(lines) + "\n")


def build_sublattice_lines(spec: StructureSpec,
                           site_species: list[str]
                           ) -> tuple[str, list[int]]:
    """SUBLATTICE INPUT lines + ELEMENT counts (in species-index order)
    derived from the per-site species list. Each orbit's basis indices
    are the positions where site_species equals its generator label.
    Within each orbit, species are split 50/50 between the orbit's
    binary mixed-pair (in canonical species order)."""
    species_canon = all_species(spec)
    sp_idx = {s: i for i, s in enumerate(species_canon)}
    counts: dict[int, int] = {i: 0 for i in range(len(species_canon))}
    lines: list[str] = []
    for orb in spec.orbits:
        basis = [i for i, lab in enumerate(site_species)
                 if lab == orb.generator_label]
        if len(basis) % 2 != 0:
            raise RuntimeError(
                f"orbit {orb.generator_label} multiplicity {len(basis)} "
                f"is odd -- can't 50/50 split"
            )
        half = len(basis) // 2
        a_idx = sp_idx[orb.mixed_pair[0]]
        b_idx = sp_idx[orb.mixed_pair[1]]
        counts[a_idx] += half
        counts[b_idx] += half
        lines.append(
            f"SUBLATTICE {orb.sublattice_id} "
            f"BASIS {' '.join(str(b) for b in basis)} "
            f"SPECIES {a_idx} {b_idx}"
        )
    return "\n".join(lines), [counts[i] for i in range(len(species_canon))]


# ----- SCRAPs run ----------------------------------------------------------

def run_scraps(spec: StructureSpec, work_dir: Path, cell_dim: int,
               ranks: int, timeout: int) -> tuple[float, Path, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    site_species = write_scraps_poscar(spec, work_dir / "POSCAR")
    sub_lines, counts = build_sublattice_lines(spec, site_species)
    counts_str = " ".join(str(c * cell_dim ** 3) for c in counts)
    n_atoms = len(site_species) * cell_dim ** 3
    mcstep = max(500, 30 * n_atoms)
    inp = (
        f"TYP     {len(counts)}\n"
        f"GITER   1000\n"
        f"ELEMENT {counts_str}\n"
        f"Cell_DIM {cell_dim} {cell_dim} {cell_dim}\n"
        f"NESTS   8\n"
        f"MCSTEP  {mcstep}\n"
        f"TOTALIter 30\n"
        f"MAXSHELLNUM 3\n"
        f"WEIGHT 1 1 1\n"
        f"RNG_SEED 1\n"
        f"EARLY_STOP_PATIENCE 5\n"
        f"{sub_lines}\n"
    )
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


# ----- mcsqs run -----------------------------------------------------------

def run_mcsqs(spec: StructureSpec, work_dir: Path, cell_dim: int,
              budget: int) -> tuple[float, Path | None, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    write_mcsqs_rndstr(spec, work_dir / "rndstr.in")
    cutoff_2 = spec.cluster_cutoff_factor * spec.a
    clus_log = work_dir / "clus.log"
    subprocess.run(
        ["mcsqs", f"-2={cutoff_2:.4f}"],
        cwd=work_dir, stdout=clus_log.open("w"), stderr=subprocess.STDOUT,
        timeout=60,
    )
    site_species = write_scraps_poscar(spec,
                                        work_dir / "_for_count.poscar")
    natom = len(site_species) * cell_dim ** 3
    (work_dir / "_for_count.poscar").unlink(missing_ok=True)
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
        return wall, None, "no bestsqs.out (mcsqs setup may have timed out)"
    poscar = atat_to_poscar(bestsqs, work_dir / "bestsqs.poscar")
    return wall, poscar, "ok"


# ----- runner --------------------------------------------------------------

@dataclass
class CaseResult:
    tool: str
    structure: str
    cell_dim: int
    n_atoms: int
    wall_seconds: float
    shells: list = field(default_factory=list)
    notes: str = ""


def run_case(spec: StructureSpec, cell_dim: int, ranks: int, budget: int,
             work_root: Path) -> tuple[CaseResult, CaseResult]:
    base_atoms = len(build_structure(spec))
    n_atoms = base_atoms * cell_dim ** 3
    species = all_species(spec)
    sub_of = sublattice_of(spec)

    sc_dir = work_root / f"{spec.name}_n{n_atoms}_scraps"
    sc_wall, sc_poscar, _ = run_scraps(
        spec, sc_dir, cell_dim, ranks=ranks, timeout=budget + 60
    )
    fix_scraps_species(sc_poscar, species)
    sc_shells = compute_alpha_per_shell(
        sc_poscar, species=species, sublattice_of=sub_of, n_shells=3
    )
    sc = CaseResult(
        tool="scraps-perpair", structure=spec.name, cell_dim=cell_dim,
        n_atoms=n_atoms, wall_seconds=round(sc_wall, 2), shells=sc_shells,
    )

    mc_dir = work_root / f"{spec.name}_n{n_atoms}_mcsqs"
    mc_wall, mc_poscar, mc_note = run_mcsqs(spec, mc_dir, cell_dim, budget=budget)
    if mc_poscar is None:
        mc = CaseResult(
            tool="mcsqs", structure=spec.name, cell_dim=cell_dim,
            n_atoms=n_atoms, wall_seconds=round(mc_wall, 2),
            shells=[], notes=mc_note,
        )
    else:
        mc_shells = compute_alpha_per_shell(
            mc_poscar, species=species, sublattice_of=sub_of, n_shells=3
        )
        mc = CaseResult(
            tool="mcsqs", structure=spec.name, cell_dim=cell_dim,
            n_atoms=n_atoms, wall_seconds=round(mc_wall, 2),
            shells=mc_shells, notes=mc_note,
        )
    return sc, mc


def print_results(structure_name: str,
                  results: list[tuple[CaseResult, CaseResult]]) -> None:
    print(f"\n=== {structure_name} ===")
    print("| atoms | tool          | wall (s) | "
          "max|α| intra | max|α| inter |")
    print("| ----- | ------------- | -------- | "
          "------------ | ------------ |")
    for sc, mc in results:
        for r in (sc, mc):
            if not r.shells:
                print(f"| {r.n_atoms:>5} | {r.tool:<13} | "
                      f"{r.wall_seconds:>8.2f} | "
                      f"{'n/a':>12} | {'n/a':>12} |"
                      + (f"  ({r.notes})" if r.notes else ""))
                continue
            max_intra = max((s["max_alpha_intra"] for s in r.shells),
                            default=0.0)
            max_inter = max((s["max_alpha_inter"] for s in r.shells),
                            default=0.0)
            print(f"| {r.n_atoms:>5} | {r.tool:<13} | "
                  f"{r.wall_seconds:>8.2f} | "
                  f"{max_intra:>12.4f} | {max_inter:>12.4f} |")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structures", nargs="+",
                   default=["laves_c15", "heusler_l21"],
                   choices=list(STRUCTURES.keys()),
                   help="Which structure classes to run")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2],
                   help="Cell_DIM values (NxNxN). Default 1 2.")
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--budget", type=int, default=120,
                   help="Wall budget per case in seconds")
    p.add_argument("--work-root", type=Path,
                   default=Path("/tmp/multi_orbit_compare"))
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
    all_results: dict[str, list[tuple[CaseResult, CaseResult]]] = {}
    for sname in args.structures:
        spec = STRUCTURES[sname]
        results = []
        for cd in args.sizes:
            base = len(build_structure(spec))
            n_atoms = base * cd ** 3
            print(f"\n==== {sname} cell_dim={cd} "
                  f"(n_atoms={n_atoms}) ====", file=sys.stderr)
            sc, mc = run_case(spec, cd, ranks=args.ranks, budget=args.budget,
                              work_root=args.work_root)
            print(f"  scraps wall = {sc.wall_seconds:.2f}s; "
                  f"mcsqs wall = {mc.wall_seconds:.2f}s", file=sys.stderr)
            results.append((sc, mc))
        all_results[sname] = results
        print_results(sname, results)
    if args.json is not None:
        payload = {
            "structures": {
                name: {
                    "lattice_param": STRUCTURES[name].a,
                    "space_group": STRUCTURES[name].space_group,
                    "orbits": [asdict(o) for o in STRUCTURES[name].orbits],
                    "cases": [
                        {"scraps": asdict(sc), "mcsqs": asdict(mc)}
                        for sc, mc in cases
                    ],
                }
                for name, cases in all_results.items()
            }
        }
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"JSON summary -> {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
