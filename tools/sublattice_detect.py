#!/usr/bin/env python3
"""
Detect crystallographic sublattices in a POSCAR via Wyckoff-orbit analysis.

For substitutional SQS targeting in multi-sublattice structures (HEA carbides,
double perovskites, Heusler-type, spinels, ...), the right default partition
of basis positions into sublattices is the Wyckoff-orbit partition: two
basis positions belong to the same sublattice if they're related by a space-
group symmetry operation, because they then share identical pair-correlation
environments and any SRO target set on one applies to all its orbit-mates.

This tool reads a POSCAR, identifies the space group, groups basis positions
by Wyckoff orbit, and emits draft SUBLATTICE lines for SCRAPs' INPUT file.
The user fills in per-sublattice SPECIES (chemistry is a user-intent decision,
not a symmetry one).

Caveat: pymatgen distinguishes basis positions by species identity when
computing orbits. A POSCAR with explicit per-sublattice species labels
("Na Cl" with counts "4 4") gives the orbit partition you'd expect. A POSCAR
with all-identical species (e.g. a SCRAPs-style bare-lattice POSCAR with a
single counts line) will report a single orbit no matter the geometry. For
multi-sublattice intent, write the POSCAR with distinct per-sublattice
species blocks -- this is what pymatgen / ASE / mcsqs all expect.

Usage:
    sublattice_detect.py POSCAR [options]

See --help for full flags.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from pymatgen.io.vasp import Poscar
    from pymatgen.io.vasp.inputs import BadPoscarWarning
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    warnings.filterwarnings("ignore", category=BadPoscarWarning)
except ImportError:
    print("ERROR: pymatgen is required. Install with: pip install pymatgen",
          file=sys.stderr)
    sys.exit(2)


# ---- Data classes ---------------------------------------------------------

@dataclass
class Orbit:
    """A Wyckoff orbit -- a set of basis positions related by symmetry."""
    representative: int                 # basis index of the orbit representative
    members: list[int]                  # all basis indices in this orbit
    wyckoff_letter: str                 # Wyckoff letter ('a', 'b', 'c', ...)
    site_symmetry: Optional[str]        # site-symmetry symbol (e.g. 'm-3m')
    species: list[str]                  # species label per member (informational)
    frac_coords: list[list[float]]      # fractional coords of each member


@dataclass
class DetectionResult:
    space_group_number: int
    space_group_symbol: str
    symprec: float
    n_basis: int
    orbits: list[Orbit]


# ---- Detection ------------------------------------------------------------

def load_structure(poscar_path: Path) -> Structure:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BadPoscarWarning)
        poscar = Poscar.from_file(poscar_path, check_for_potcar=False,
                                  read_velocities=False)
    return poscar.structure


def detect_orbits(structure: Structure, symprec: float = 1e-3,
                  angle_tolerance: float = 5.0) -> DetectionResult:
    """Run Wyckoff-orbit detection on the structure's basis.

    Orbits are returned in order of first appearance in the POSCAR, so
    basis index 0 always belongs to orbit 0. This matches SCRAPs' own
    basis-outermost atom indexing and keeps the SUBLATTICE block IDs
    aligned with the POSCAR layout.
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec,
                             angle_tolerance=angle_tolerance)
    dataset = sga.get_symmetry_dataset()

    # spglib >= 2.0 returns a SpglibDataset (attribute access); older
    # versions returned a dict. Support both.
    def _field(name):
        if hasattr(dataset, name):
            return getattr(dataset, name)
        return dataset[name]

    equiv = [int(x) for x in _field("equivalent_atoms")]
    wyckoffs = [str(x) for x in _field("wyckoffs")]
    try:
        site_syms = [str(x) for x in _field("site_symmetry_symbols")]
    except (KeyError, AttributeError):
        site_syms = [""] * len(equiv)

    # Group basis indices by representative, in order of first appearance.
    seen_reps: list[int] = []
    members_by_rep: dict[int, list[int]] = {}
    for i, rep in enumerate(equiv):
        if rep not in members_by_rep:
            members_by_rep[rep] = []
            seen_reps.append(rep)
        members_by_rep[rep].append(i)

    orbits: list[Orbit] = []
    for rep in seen_reps:
        members = members_by_rep[rep]
        orbits.append(Orbit(
            representative=rep,
            members=members,
            wyckoff_letter=wyckoffs[rep],
            site_symmetry=site_syms[rep] or None,
            species=[structure.species[i].symbol for i in members],
            frac_coords=[
                [float(c) for c in structure.frac_coords[i]] for i in members
            ],
        ))

    return DetectionResult(
        space_group_number=sga.get_space_group_number(),
        space_group_symbol=sga.get_space_group_symbol(),
        symprec=symprec,
        n_basis=len(structure),
        orbits=orbits,
    )


# ---- Output formatting ----------------------------------------------------

def format_human(result: DetectionResult) -> str:
    out: list[str] = []
    out.append(f"Space group: {result.space_group_symbol} "
               f"(#{result.space_group_number})")
    out.append(f"symprec: {result.symprec:g}")
    out.append(f"Basis positions: {result.n_basis}")
    out.append(f"Wyckoff orbits: {len(result.orbits)}")
    out.append("")
    out.append(f"  {'orbit':>5}  {'Wyck':>4}  {'mult':>4}  "
               f"{'site sym':<10}  {'species':<14}  basis indices")
    for k, orb in enumerate(result.orbits):
        site = orb.site_symmetry or "?"
        # Compact species listing: collapse runs of identical labels.
        unique_specs = list(dict.fromkeys(orb.species))
        spec_str = ",".join(unique_specs)
        idxs = ",".join(str(i) for i in orb.members)
        out.append(
            f"  {k:>5}  {orb.wyckoff_letter:>4}  {len(orb.members):>4}  "
            f"{site:<10}  {spec_str:<14}  [{idxs}]"
        )
    out.append("")
    out.append("Draft SUBLATTICE block (replace <fill in> with SPECIES indices "
               "matching ELEMENT order):")
    for k, orb in enumerate(result.orbits):
        basis_list = " ".join(str(i) for i in orb.members)
        out.append(f"SUBLATTICE {k} BASIS {basis_list} SPECIES <fill in>")
    if len(result.orbits) == 1:
        out.append("")
        out.append("NOTE: only one orbit detected -- explicit SUBLATTICE is "
                   "unnecessary for this POSCAR (the default single-sublattice "
                   "path already covers this case). If you intend to break "
                   "symmetry by hand, declare SUBLATTICE blocks anyway.")
    return "\n".join(out)


def format_json(result: DetectionResult) -> str:
    payload = {
        "space_group_number": result.space_group_number,
        "space_group_symbol": result.space_group_symbol,
        "symprec": result.symprec,
        "n_basis": result.n_basis,
        "n_orbits": len(result.orbits),
        "orbits": [
            {
                "orbit_id": k,
                "wyckoff_letter": o.wyckoff_letter,
                "site_symmetry": o.site_symmetry,
                "multiplicity": len(o.members),
                "basis_indices": o.members,
                "species": o.species,
                "frac_coords": o.frac_coords,
                "draft_sublattice_line": (
                    f"SUBLATTICE {k} BASIS "
                    f"{' '.join(str(i) for i in o.members)} "
                    f"SPECIES <fill in>"
                ),
            }
            for k, o in enumerate(result.orbits)
        ],
    }
    return json.dumps(payload, indent=2)


# ---- Write-back to INPUT --------------------------------------------------

def write_to_input(result: DetectionResult, input_path: Path,
                   overwrite_existing: bool = False) -> None:
    """Append draft SUBLATTICE lines to an INPUT file.

    Refuses if any SUBLATTICE line already exists, unless overwrite_existing.
    Lines contain '<fill in>' SPECIES placeholders that the user must edit
    before running scraps; the parser will abort otherwise, which is the
    intended forcing function.
    """
    existing = input_path.read_text() if input_path.exists() else ""
    has_sublattice = any(
        line.strip().upper().startswith("SUBLATTICE")
        for line in existing.splitlines()
    )
    if has_sublattice and not overwrite_existing:
        raise RuntimeError(
            f"{input_path} already contains SUBLATTICE lines; "
            f"pass --overwrite to replace them."
        )
    if has_sublattice:
        kept = [
            line for line in existing.splitlines()
            if not line.strip().upper().startswith("SUBLATTICE")
        ]
        existing = "\n".join(kept)
        if existing and not existing.endswith("\n"):
            existing += "\n"

    header = (
        f"# Sublattice detection ({len(result.orbits)} orbits, "
        f"{result.space_group_symbol} #{result.space_group_number}, "
        f"symprec={result.symprec:g})"
    )
    body_lines = [header]
    for k, orb in enumerate(result.orbits):
        basis_list = " ".join(str(i) for i in orb.members)
        body_lines.append(
            f"SUBLATTICE {k} BASIS {basis_list} SPECIES <fill in>"
            f"   # Wyckoff {orb.wyckoff_letter}, "
            f"site sym {orb.site_symmetry or '?'}"
        )
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    input_path.write_text(existing + sep + "\n".join(body_lines) + "\n")


# ---- CLI ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sublattice_detect.py",
        description=(
            "Detect crystallographic sublattices in a POSCAR via Wyckoff-orbit "
            "analysis. Emits draft SUBLATTICE lines for SCRAPs' INPUT file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print the orbit partition and draft SUBLATTICE lines
  sublattice_detect.py POSCAR

  # Use a looser symmetry tolerance for noisy POSCARs (which would
  # otherwise spuriously split one orbit into many)
  sublattice_detect.py POSCAR --symprec 1e-2

  # Machine-readable output for downstream tooling
  sublattice_detect.py POSCAR --json

  # Append draft SUBLATTICE lines to an existing INPUT file. Lines
  # contain '<fill in>' SPECIES placeholders that you MUST edit before
  # running scraps (the parser will abort on the placeholder otherwise).
  sublattice_detect.py POSCAR --write INPUT
""")
    p.add_argument("poscar", type=Path, help="Path to VASP POSCAR file")
    p.add_argument("--symprec", type=float, default=1e-3,
                   help="Symmetry tolerance for spglib (default: 1e-3). "
                        "Increase for noisy POSCARs that split orbits "
                        "spuriously; decrease to detect symmetry-broken "
                        "structures you expect to be lower-symmetry.")
    p.add_argument("--angle-tolerance", type=float, default=5.0,
                   help="Angle tolerance in degrees (default: 5.0)")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of the human-readable table")
    p.add_argument("--write", type=Path, default=None,
                   help="Append draft SUBLATTICE lines to the given INPUT "
                        "file. Refuses if SUBLATTICE lines already exist "
                        "unless --overwrite is also given.")
    p.add_argument("--overwrite", action="store_true",
                   help="With --write, replace any existing SUBLATTICE lines.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        structure = load_structure(args.poscar)
    except Exception as e:
        print(f"ERROR loading POSCAR: {e}", file=sys.stderr)
        return 2

    try:
        result = detect_orbits(structure, symprec=args.symprec,
                               angle_tolerance=args.angle_tolerance)
    except Exception as e:
        print(f"ERROR running symmetry analysis: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(format_json(result))
    else:
        print(format_human(result))

    if args.write is not None:
        try:
            write_to_input(result, args.write,
                           overwrite_existing=args.overwrite)
            print(f"Wrote draft SUBLATTICE lines to {args.write}",
                  file=sys.stderr)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
