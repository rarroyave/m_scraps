#!/usr/bin/env python3
"""
Independent Warren-Cowley alpha verifier for SCRAPS supercells.

Computes the alpha_ij matrix per neighbor shell directly from atomic
positions, with no dependence on SCRAPS internal accounting. Reads any
VASP POSCAR via pymatgen, so species-symbol lines, selective dynamics,
per-direction scale factors, fractional vs Cartesian modes, and
non-orthogonal cells are all handled correctly.

Optionally compares against a target-alpha file (SCRAPS INPUT-style
"SRO i j shell alpha" lines) and reports pass/fail with a tolerance.
Exit code 0 = all pairs within tolerance, 1 = any pair out.

Usage:
    verify_sro.py POSCAR [options]

See --help for full flags.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import warnings
    from pymatgen.io.vasp import Poscar
    from pymatgen.io.vasp.inputs import BadPoscarWarning
    from pymatgen.core import Structure
    # SCRAPS POSCARs have no species-symbol line — pymatgen warns and
    # uses placeholder symbols. We always override via --species, so the
    # placeholder names never reach the user. Suppress the noise.
    warnings.filterwarnings("ignore", category=BadPoscarWarning)
except ImportError:
    print("ERROR: pymatgen is required. Install with: pip install pymatgen",
          file=sys.stderr)
    sys.exit(2)


# ---- Structure loading ------------------------------------------------------

def load_structure(poscar_path: Path, species_override: Optional[list[str]] = None) -> tuple[Structure, list[str]]:
    """Load a POSCAR. If species_override is given, use it instead of any
    symbols in the POSCAR — convenient for SCRAPS output which writes
    counts without symbol labels."""
    with warnings.catch_warnings():
        # SCRAPS POSCARs have no species line; pymatgen complains. We always
        # override via --species, so the warning is noise.
        warnings.simplefilter("ignore", BadPoscarWarning)
        poscar = Poscar.from_file(poscar_path, check_for_potcar=False,
                                  read_velocities=False)
    structure = poscar.structure

    if species_override is not None:
        # Replace species labels by index, preserving count groups
        # from the POSCAR's species ordering.
        # SCRAPS POSCAR has no species symbols, so pymatgen assigns
        # placeholder labels (X0, X1, ...). We map those to the user's names.
        unique_specs_in_order = list(dict.fromkeys(s.symbol for s in structure.species))
        if len(species_override) != len(unique_specs_in_order):
            raise ValueError(
                f"--species has {len(species_override)} names but POSCAR has "
                f"{len(unique_specs_in_order)} species blocks "
                f"({unique_specs_in_order})"
            )
        mapping = dict(zip(unique_specs_in_order, species_override))
        new_species = [mapping[s.symbol] for s in structure.species]
        structure = Structure(structure.lattice, new_species, structure.frac_coords,
                              coords_are_cartesian=False)

    unique = list(dict.fromkeys(s.symbol for s in structure.species))
    return structure, unique


# ---- Shell detection --------------------------------------------------------

@dataclass
class Shell:
    """A neighbor shell defined by an inclusive distance range."""
    index: int
    r_min: float
    r_max: float
    z: float  # average coordination across all sites


def detect_shells(structure: Structure, n_shells: int,
                  cutoff: Optional[float] = None,
                  rel_gap: float = 0.05) -> list[Shell]:
    """Cluster pairwise distances into discrete shells.

    A new shell starts when the relative gap to the previous distance
    exceeds rel_gap (default 5%). Returns the first n_shells shells.

    The cutoff defaults to half the smallest lattice-vector length,
    which keeps us safely within the Wigner-Seitz radius (no
    minimum-image ambiguity).
    """
    if cutoff is None:
        cutoff = float(min(structure.lattice.abc)) / 2.0 - 1e-3

    # get_neighbor_list returns every (center, neighbor) within r.
    centers, neighbors, _, distances = structure.get_neighbor_list(r=cutoff)
    n_sites = len(structure)
    if len(distances) == 0:
        return []

    # Cluster sorted unique distances into shells via relative-gap criterion.
    sorted_d = np.sort(distances)
    shells_bounds: list[list[float]] = [[sorted_d[0]]]
    for d in sorted_d[1:]:
        if (d - shells_bounds[-1][-1]) / shells_bounds[-1][-1] > rel_gap:
            shells_bounds.append([d])
        else:
            shells_bounds[-1].append(d)

    shells: list[Shell] = []
    for i, group in enumerate(shells_bounds[:n_shells]):
        r_min = float(min(group))
        r_max = float(max(group))
        # Coordination = (total pair entries in this shell) / n_sites
        # because get_neighbor_list lists each (i,j) and (j,i) separately.
        in_shell = (distances >= r_min - 1e-6) & (distances <= r_max + 1e-6)
        z = float(in_shell.sum()) / n_sites
        shells.append(Shell(index=i, r_min=r_min, r_max=r_max, z=z))
    return shells


# ---- Alpha computation ------------------------------------------------------

def compute_alpha(structure: Structure, shells: list[Shell],
                  spectator_species: Optional[list[str]] = None
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute Warren-Cowley alpha per shell.

    If spectator_species is given (e.g. ["B"] for an HEB), spectator
    atoms are excluded from both the center and the neighbor side of
    every pair count. The reference concentration becomes the per-
    sublattice c_j^M = N_j^M / N^M (M = non-spectator atoms), the
    per-shell coordination becomes z^M (count of non-spectator neighbors
    of a non-spectator center), and rows/columns for spectator species
    are returned as NaN.

    Returns (alpha[shell, i, j], counts[shell, i, j], species_names).
    """
    # Build species index by group order in the structure.
    species_order = list(dict.fromkeys(s.symbol for s in structure.species))
    spec_idx = {sym: k for k, sym in enumerate(species_order)}
    site_species = np.array([spec_idx[s.symbol] for s in structure.species])
    n_typ = len(species_order)

    # Spectator handling. is_spectator_species[idx] is True if that species
    # was declared as spectator; is_spectator_atom[i] is the per-atom mask.
    spectator_set = set(spectator_species or [])
    is_spectator_species = np.array(
        [sym in spectator_set for sym in species_order]
    )
    is_spectator_atom = is_spectator_species[site_species]
    n_M = int((~is_spectator_atom).sum())

    counts_per_spec = np.bincount(site_species, minlength=n_typ)
    if (counts_per_spec == 0).any():
        zero = [species_order[i] for i in range(n_typ) if counts_per_spec[i] == 0]
        raise ValueError(f"Species with zero atoms: {zero}")
    # Per-sublattice species counts: spectator species get 0 on the M
    # sublattice; non-spectator species use their full count.
    counts_per_spec_M = np.where(is_spectator_species, 0, counts_per_spec)
    c_M = counts_per_spec_M / max(n_M, 1)

    # One pass over the global neighbor list for the largest cutoff we need.
    cutoff = max(s.r_max for s in shells) + 1e-3
    centers, neighbors, _, distances = structure.get_neighbor_list(r=cutoff)

    # Restrict to non-spectator centers AND non-spectator neighbors.
    if spectator_set:
        keep = (~is_spectator_atom[centers]) & (~is_spectator_atom[neighbors])
        centers = centers[keep]
        neighbors = neighbors[keep]
        distances = distances[keep]

    alpha = np.zeros((len(shells), n_typ, n_typ))
    counts = np.zeros((len(shells), n_typ, n_typ), dtype=int)
    for k, sh in enumerate(shells):
        mask = (distances >= sh.r_min - 1e-6) & (distances <= sh.r_max + 1e-6)
        cs = site_species[centers[mask]]
        ns = site_species[neighbors[mask]]
        np.add.at(counts[k], (cs, ns), 1)

        # alpha_ij = 1 - P(j|i) / c_j   (per-sublattice reference for HEB)
        # z_k^M = (non-spectator pair entries in this shell) / N^M
        if spectator_set:
            z_k = int(mask.sum()) / max(n_M, 1)
        else:
            z_k = sh.z
        for i in range(n_typ):
            for j in range(n_typ):
                if counts_per_spec_M[i] == 0 or counts_per_spec_M[j] == 0 or z_k <= 0:
                    alpha[k, i, j] = np.nan
                    continue
                Pji = counts[k, i, j] / (z_k * counts_per_spec_M[i])
                alpha[k, i, j] = 1.0 - Pji / c_M[j]

    return alpha, counts, species_order


# ---- Target file parsing ----------------------------------------------------

@dataclass
class TargetEntry:
    i: int
    j: int
    shell: int
    alpha: float


def parse_target_file(path: Path) -> list[TargetEntry]:
    """Read SCRAPS-style target lines:
        SRO <species_i> <species_j> <shell> <alpha>
    Blank lines and comments (# ...) are ignored. Any other lines are
    silently ignored (so a SCRAPS INPUT file can be passed directly).
    """
    entries: list[TargetEntry] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if not tokens or tokens[0].upper() != "SRO":
            continue
        if len(tokens) < 5:
            print(f"WARNING: malformed SRO line ignored: {raw_line}",
                  file=sys.stderr)
            continue
        try:
            entries.append(TargetEntry(
                i=int(tokens[1]), j=int(tokens[2]),
                shell=int(tokens[3]), alpha=float(tokens[4])
            ))
        except ValueError:
            print(f"WARNING: malformed SRO line ignored: {raw_line}",
                  file=sys.stderr)
    return entries


# ---- Output formatting ------------------------------------------------------

def format_matrix_table(alpha: np.ndarray, names: list[str], width: int = 9) -> str:
    """Return a human-readable matrix table for one shell."""
    n = len(names)
    header = "         " + "".join(f"{nm:>{width}}" for nm in names)
    lines = [header]
    for i in range(n):
        row = f"  {names[i]:>4}  " + "".join(
            f"{alpha[i, j]:+{width-2}.4f} " for j in range(n)
        )
        lines.append(row)
    return "\n".join(lines)


def report_human(structure: Structure, shells: list[Shell],
                 alpha: np.ndarray, names: list[str],
                 targets: Optional[list[TargetEntry]] = None,
                 tolerance: float = 0.02) -> bool:
    """Print human-readable report. Returns True iff all target pairs pass."""
    print(f"Lattice: {structure.lattice.abc} (Angstrom), "
          f"angles {structure.lattice.angles}")
    print(f"Atoms: {len(structure)} total, species {names}")
    print(f"Detected shells: {len(shells)}")
    for sh in shells:
        print(f"  shell {sh.index}: r ∈ [{sh.r_min:.4f}, {sh.r_max:.4f}], z = {sh.z:g}")
    print()

    all_pass = True
    for k, sh in enumerate(shells):
        # Suppress all-NaN shells (typical with spectator filtering: shells
        # where every entry between non-spectator species has zero count
        # because the shell is dominated by spectator-involving pairs).
        if np.all(np.isnan(alpha[k])):
            continue
        print(f"=== Shell {sh.index} (z = {sh.z:g}) ===")
        print(format_matrix_table(alpha[k], names))
        print()

        if targets is not None:
            shell_targets = [t for t in targets if t.shell == sh.index]
            if shell_targets:
                print(f"Target comparison (tolerance = {tolerance}):")
                print(f"  {'pair':<10} {'target':>10} {'achieved':>10} "
                      f"{'|err|':>8}   status")
                for t in shell_targets:
                    if t.i >= len(names) or t.j >= len(names):
                        print(f"  WARNING: species index out of range in target: "
                              f"{t.i} {t.j}", file=sys.stderr)
                        all_pass = False
                        continue
                    achieved = float(alpha[k, t.i, t.j])
                    err = abs(achieved - t.alpha)
                    status = "PASS" if err <= tolerance else "FAIL"
                    if status == "FAIL":
                        all_pass = False
                    label = f"{names[t.i]}-{names[t.j]}"
                    print(f"  {label:<10} {t.alpha:>+10.4f} {achieved:>+10.4f} "
                          f"{err:>8.4f}   {status}")
                print()
    return all_pass


def report_json(structure: Structure, shells: list[Shell],
                alpha: np.ndarray, counts: np.ndarray, names: list[str],
                targets: Optional[list[TargetEntry]] = None,
                tolerance: float = 0.02) -> bool:
    """Emit JSON to stdout. Returns True iff all target pairs pass."""
    all_pass = True
    out = {
        "n_atoms": len(structure),
        "species": names,
        "lattice_abc": list(structure.lattice.abc),
        "lattice_angles": list(structure.lattice.angles),
        "shells": [],
    }
    for k, sh in enumerate(shells):
        sh_out = {
            "index": sh.index,
            "r_min": sh.r_min,
            "r_max": sh.r_max,
            "z": sh.z,
            "alpha_matrix": alpha[k].tolist(),
            "pair_counts": counts[k].tolist(),
        }
        if targets is not None:
            comparisons = []
            for t in targets:
                if t.shell != sh.index:
                    continue
                if t.i >= len(names) or t.j >= len(names):
                    continue
                achieved = float(alpha[k, t.i, t.j])
                err = abs(achieved - t.alpha)
                passed = err <= tolerance
                if not passed:
                    all_pass = False
                comparisons.append({
                    "i": t.i, "j": t.j,
                    "pair": f"{names[t.i]}-{names[t.j]}",
                    "target": t.alpha,
                    "achieved": achieved,
                    "abs_error": err,
                    "passed": passed,
                })
            sh_out["target_comparisons"] = comparisons
        out["shells"].append(sh_out)
    out["all_pass"] = all_pass if targets is not None else None
    out["tolerance"] = tolerance if targets is not None else None
    print(json.dumps(out, indent=2))
    return all_pass


# ---- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="verify_sro.py",
        description=(
            "Compute Warren-Cowley alpha per shell from a VASP POSCAR and "
            "optionally compare against a target file (SCRAPS INPUT-style)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic — just print alpha matrices for the first 3 shells
  verify_sro.py SCRAPS.vasp

  # SCRAPS output has no species symbols; supply them
  verify_sro.py SCRAPS.vasp --species Ni,Co,Cr

  # Compare against the same INPUT file you fed SCRAPS
  verify_sro.py SCRAPS.vasp --species Ni,Co,Cr --target INPUT \\
      --tolerance 0.02

  # Machine-readable output for CI / regression sweeps
  verify_sro.py SCRAPS.vasp --species Ni,Co,Cr --target INPUT --json
""")
    p.add_argument("poscar", type=Path, help="Path to VASP POSCAR file")
    p.add_argument("--species", type=str, default=None,
                   help="Comma-separated species symbols, in POSCAR-block order "
                        "(e.g. 'Ni,Co,Cr'). Required for SCRAPS output, which "
                        "writes no symbols.")
    p.add_argument("--shells", type=int, default=3,
                   help="Number of neighbor shells to compute (default: 3)")
    p.add_argument("--cutoff", type=float, default=None,
                   help="Max neighbor distance in Angstrom "
                        "(default: half of smallest lattice vector)")
    p.add_argument("--rel-gap", type=float, default=0.05,
                   help="Relative-gap threshold for shell clustering "
                        "(default: 0.05 = 5%%)")
    p.add_argument("--target", type=Path, default=None,
                   help="Target file with SCRAPS-style 'SRO i j shell alpha' "
                        "lines. A SCRAPS INPUT file can be passed directly.")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="Pass/fail tolerance on |achieved - target| "
                        "(default: 0.02)")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of human table")
    p.add_argument("--spectator-species", type=str, default=None,
                   help="Comma-separated list of species names to treat as "
                        "spectator (e.g. 'B' for HEB-type AlB2 diborides). "
                        "Spectator atoms are excluded from neighbor counts; "
                        "alpha is computed with per-sublattice reference "
                        "c_j^M and rows/cols for spectator species are NaN.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    species_override = (
        [s.strip() for s in args.species.split(",")]
        if args.species else None
    )

    try:
        structure, _ = load_structure(args.poscar, species_override)
    except Exception as e:
        print(f"ERROR loading POSCAR: {e}", file=sys.stderr)
        return 2

    shells = detect_shells(structure, args.shells,
                           cutoff=args.cutoff, rel_gap=args.rel_gap)
    if not shells:
        print("ERROR: no shells detected within the cutoff. Try --cutoff <larger>.",
              file=sys.stderr)
        return 2

    spectator = (
        [s.strip() for s in args.spectator_species.split(",")]
        if args.spectator_species else None
    )
    alpha, counts, names = compute_alpha(structure, shells, spectator)

    targets = parse_target_file(args.target) if args.target else None

    if args.json:
        all_pass = report_json(structure, shells, alpha, counts, names,
                               targets, args.tolerance)
    else:
        all_pass = report_human(structure, shells, alpha, names,
                                targets, args.tolerance)

    if targets is not None and not all_pass:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
