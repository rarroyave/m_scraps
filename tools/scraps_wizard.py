#!/usr/bin/env python3
"""Novice-friendly CLI for designing SQS supercells with scraps-perpair.

Orchestrates the existing tooling -- sublattice_detect, suggest_cell,
run_sqs_from_atoms (with AUTO_BUDGET), verify_sro -- behind a small set
of subcommands that ask only the questions a non-expert can answer:
which lattice class, what composition, how many atoms.

Subcommands:
    quick         One-shot SQS for a common structure template.
    from-structure  Read a POSCAR/CIF, auto-detect sublattices, prompt
                  for composition per orbit.
    interactive   Step-by-step REPL covering all three paths above.
    list-templates  Show the built-in structure templates.

Examples:
    # Fast path: I know my chemistry, I know what kind of cell
    scraps_wizard.py quick --template fcc --composition Ni:0.5,Co:0.5

    # I have a real POSCAR (Materials Project, paper SI, ...)
    scraps_wizard.py from-structure my_perovskite.poscar

    # I'm new and want hand-holding
    scraps_wizard.py interactive
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRAPS_BIN = REPO_ROOT / "SCRAPs" / "scraps"
sys.path.insert(0, str(REPO_ROOT / "tools"))

# Defer pymatgen import so `list-templates` and `--help` work even when
# pymatgen isn't installed.


# ----- structure templates -------------------------------------------------

@dataclass
class TemplateOrbit:
    """One Wyckoff orbit in a structure template."""
    placeholder: str       # element symbol pymatgen uses to seed the orbit
    coords: tuple[float, float, float]  # one Wyckoff representative (frac)
    user_name: str         # human-readable orbit name shown in prompts
    sublattice_id: int     # SCRAPs SUBLATTICE id


@dataclass
class Template:
    name: str
    short_desc: str       # one-liner shown in `list-templates`
    space_group: str      # pymatgen.Structure.from_spacegroup input
    a: float              # default lattice parameter (Angstrom); user override OK
    orbits: list[TemplateOrbit]
    # Some templates need extra hints, e.g. CSRO defaults from literature.
    notes: str = ""


TEMPLATES: dict[str, Template] = {
    "fcc": Template(
        name="fcc",
        short_desc="FCC primitive — single sublattice "
                   "(Cu/Ag/Au, NiCoCr-type CCAs and HEAs)",
        space_group="Fm-3m", a=3.6,
        orbits=[TemplateOrbit("Cu", (0.0, 0.0, 0.0),
                              "metal sublattice", 0)],
    ),
    "bcc": Template(
        name="bcc",
        short_desc="BCC primitive — single sublattice "
                   "(Fe/Cr, NbMoTaW refractory HEAs)",
        space_group="Im-3m", a=3.0,
        orbits=[TemplateOrbit("Fe", (0.0, 0.0, 0.0),
                              "metal sublattice", 0)],
    ),
    "b2": Template(
        name="b2",
        short_desc="B2 CsCl-type — 2 sublattices "
                   "(NiTi, CuZn, AlNi shape-memory and intermetallic CCAs)",
        space_group="Pm-3m", a=3.0,
        orbits=[
            TemplateOrbit("Mg", (0.0, 0.0, 0.0),
                          "A site (cube corner)", 0),
            TemplateOrbit("Cu", (0.5, 0.5, 0.5),
                          "B site (body-center)", 1),
        ],
        notes="Each A neighbor at 1NN is on B and vice versa; 2NN is "
              "intra-orbit. Target intra-orbit alpha to design CSRO on "
              "each sublattice independently.",
    ),
    "rocksalt": Template(
        name="rocksalt",
        short_desc="NaCl rocksalt — 2 sublattices "
                   "(HEA carbides MoNbTaVW-C, nitrides, halides, MgO-style oxides)",
        space_group="Fm-3m", a=4.5,
        orbits=[
            TemplateOrbit("Mg", (0.0, 0.0, 0.0),
                          "cation sublattice", 0),
            TemplateOrbit("Cl", (0.5, 0.5, 0.5),
                          "anion sublattice", 1),
        ],
        notes="Cation z=6 anions at 1NN (cross-sub); anion z=6 cations "
              "at the same distance; cation-cation 2NN at d=a*sqrt(2)/2 "
              "is intra-orbit (FCC).",
    ),
    "heusler-l21": Template(
        name="heusler-l21",
        short_desc="Full Heusler L2_1 — 3 sublattices "
                   "(Cu2MnAl, X2YZ magnetic + thermoelectric Heuslers)",
        space_group="Fm-3m", a=5.95,
        orbits=[
            TemplateOrbit("Mn", (0.0, 0.0, 0.0),
                          "4a Wyckoff (Y site)", 0),
            TemplateOrbit("Al", (0.5, 0.5, 0.5),
                          "4b Wyckoff (Z site)", 1),
            TemplateOrbit("Cu", (0.25, 0.25, 0.25),
                          "8c Wyckoff (X site, double-occupied)", 2),
        ],
        notes="In X2YZ, the 8c orbit has twice the multiplicity of the "
              "4a and 4b orbits.",
    ),
    "perovskite-abo3": Template(
        name="perovskite-abo3",
        short_desc="Cubic perovskite ABO_3 — 3 sublattices "
                   "(BaTiO_3, SrTiO_3, doped perovskite oxides)",
        space_group="Pm-3m", a=3.9,
        orbits=[
            TemplateOrbit("Ba", (0.0, 0.0, 0.0),
                          "A site (cube corner, 12-coord)", 0),
            TemplateOrbit("Ti", (0.5, 0.5, 0.5),
                          "B site (cube center, 6-coord)", 1),
            TemplateOrbit("O",  (0.5, 0.5, 0.0),
                          "O site (face center, 2-coord to B)", 2),
        ],
        notes="A:B:O = 1:1:3 stoichiometric. Standard A/B doping "
              "studies (e.g. Ba/Sr on A, Ti/Zr on B) become single-line "
              "compositions here.",
    ),
}


# ----- composition parsing -------------------------------------------------

def parse_composition(s: str) -> dict[str, float]:
    """Parse 'Ni:0.5,Co:0.5' (or 'Ni=0.5,Co=0.5') into {'Ni': 0.5, 'Co': 0.5}.
    Whitespace around tokens is ignored. Raises ValueError on malformed
    input or if fractions don't sum to 1 within 0.01."""
    out: dict[str, float] = {}
    for token in s.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        for sep in (":", "="):
            if sep in token:
                sp, frac = token.split(sep, 1)
                break
        else:
            raise ValueError(
                f"Composition token {token!r} missing ':' or '=' separator "
                f"(expected 'Ni:0.5')"
            )
        sp = sp.strip()
        try:
            f = float(frac.strip())
        except ValueError:
            raise ValueError(f"Composition token {token!r}: "
                             f"fraction {frac!r} is not a number")
        if sp in out:
            raise ValueError(f"Composition: species {sp!r} listed twice")
        out[sp] = f
    if not out:
        raise ValueError("Empty composition")
    total = sum(out.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Composition fractions sum to {total:.4f}, "
                         f"expected 1.0 (within 0.01)")
    return out


def split_to_integers(fractions: dict[str, float], n_total: int) -> dict[str, int]:
    """Convert fractional composition to integer atom counts summing to n_total.
    Uses Hamilton's largest-remainder method so the rounding error is < 1
    per species and the result is deterministic given the input order."""
    floats = [(sp, frac * n_total) for sp, frac in fractions.items()]
    floors = [(sp, int(f)) for sp, f in floats]
    remainders = sorted(
        ((f - int(f), sp) for sp, f in floats),
        reverse=True,
    )
    assigned = sum(c for _, c in floors)
    extras = n_total - assigned
    extra_for = {sp for _, sp in remainders[:extras]}
    return {sp: floor + (1 if sp in extra_for else 0)
            for sp, floor in floors}


# ----- cell-size selection -------------------------------------------------

def pick_cell_dim(template_atoms: int, target_atoms: int,
                  fractions_per_orbit: list[dict[str, float]],
                  template: Template,
                  max_search: int = 8) -> tuple[int, int]:
    """Pick a cubic cell_dim N where template_atoms * N^3 is closest to
    target_atoms AND every species' atom count works out to an integer
    given its fractional composition (per orbit). Returns (N, n_atoms)."""
    best = (None, None, float("inf"))
    for N in range(1, max_search + 1):
        n_atoms = template_atoms * N ** 3
        # Per-orbit feasibility: each orbit has multiplicity_per_cell *
        # N^3 atoms; the fractions for that orbit must give integer
        # counts. We don't know exact per-orbit multiplicities until we
        # build the structure, so estimate by template_atoms / n_orbits
        # for a quick filter. The exact check happens at run time.
        per_orbit_mult = template_atoms // len(template.orbits)
        orbit_total = per_orbit_mult * N ** 3
        feasible = True
        for fracs in fractions_per_orbit:
            for sp, f in fracs.items():
                count = orbit_total * f
                if abs(count - round(count)) > 0.001:
                    feasible = False
                    break
            if not feasible:
                break
        if not feasible:
            continue
        delta = abs(n_atoms - target_atoms)
        if delta < best[2]:
            best = (N, n_atoms, delta)
    if best[0] is None:
        raise ValueError(
            f"No cell size in 1..{max_search} gives an integer-feasible "
            f"composition for this template + target = {target_atoms} atoms."
        )
    return best[0], best[1]


# ----- structure assembly --------------------------------------------------

def build_primitive(template: Template, a: float | None = None):
    """Build the primitive ASE Atoms cell from a template, returning
    (atoms, per_site_orbit_id) so callers know which template orbit
    each basis position belongs to."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    a_use = a if a is not None else template.a
    species = [orb.placeholder for orb in template.orbits]
    coords = [list(orb.coords) for orb in template.orbits]
    lat = Lattice.cubic(a_use)
    struct = Structure.from_spacegroup(template.space_group, lat,
                                       species=species, coords=coords)
    # Map each site back to its template orbit by species placeholder.
    label_to_id = {orb.placeholder: orb.sublattice_id
                   for orb in template.orbits}
    per_site_sub = [label_to_id[s.symbol] for s in struct.species]
    atoms = AseAtomsAdaptor().get_atoms(struct)
    return atoms, per_site_sub


def sublattices_arg(per_site_sub: list[int],
                    fractions_per_orbit: list[dict[str, float]],
                    species_order: list[str]
                    ) -> list[tuple[int, list[int], list[int]]]:
    """Build the sublattices= argument for run_sqs_from_atoms.
    Each tuple: (sub_id, basis_positions, species_indices_on_that_sub)."""
    sp_to_idx = {s: i for i, s in enumerate(species_order)}
    out = []
    for sub_id, fracs in enumerate(fractions_per_orbit):
        basis = [i for i, s in enumerate(per_site_sub) if s == sub_id]
        species_indices = [sp_to_idx[sp] for sp in fracs.keys()]
        out.append((sub_id, basis, species_indices))
    return out


# ----- the orchestration pipeline -----------------------------------------

@dataclass
class WizardResult:
    cell_dim: int
    n_atoms: int
    work_dir: Path
    output_path: Path
    fitness: float | None
    species_counts: dict[str, int]


def run_pipeline(template: Template,
                 fractions_per_orbit: list[dict[str, float]],
                 target_atoms: int = 100,
                 work_dir: Path | None = None,
                 lattice_param: float | None = None,
                 ranks: int = 4,
                 output_format: str = "cif",
                 budget: int = 2,
                 quiet: bool = False) -> WizardResult:
    """End-to-end: build primitive -> pick cell -> run SCRAPs -> verify -> export."""
    try:
        from scraps_ase import run_sqs_from_atoms
    except ImportError as e:
        raise RuntimeError(
            "scraps_ase not importable; ensure tools/ is on PYTHONPATH "
            f"or run from the repo root (saw: {e})"
        )

    if len(fractions_per_orbit) != len(template.orbits):
        raise ValueError(
            f"Template {template.name!r} has {len(template.orbits)} orbits "
            f"but got {len(fractions_per_orbit)} composition entries"
        )

    # Flatten compositions into one global species order (preserving
    # per-orbit order). Disjoint species across orbits is the SCRAPs
    # multi-sublattice contract; the wizard enforces this.
    species_order: list[str] = []
    for fracs in fractions_per_orbit:
        for sp in fracs:
            if sp in species_order:
                raise ValueError(
                    f"Species {sp!r} appears on multiple sublattices; "
                    f"the SCRAPs multi-sublattice path requires disjoint "
                    f"species per orbit."
                )
            species_order.append(sp)

    primitive, per_site_sub = build_primitive(template, lattice_param)
    template_atoms = len(primitive)
    cell_dim, n_atoms = pick_cell_dim(
        template_atoms, target_atoms, fractions_per_orbit, template,
    )

    # Per-orbit element counts. For each sublattice the orbit gets
    # exactly len(orbit_in_primitive) * N^3 atoms; split into species
    # by the user's fractions.
    element_counts: list[int] = [0] * len(species_order)
    sp_to_idx = {s: i for i, s in enumerate(species_order)}
    for sub_id, fracs in enumerate(fractions_per_orbit):
        orbit_atoms_in_primitive = per_site_sub.count(sub_id)
        orbit_atoms = orbit_atoms_in_primitive * cell_dim ** 3
        per_species = split_to_integers(fracs, orbit_atoms)
        for sp, count in per_species.items():
            element_counts[sp_to_idx[sp]] = count

    if work_dir is None:
        tag = "_".join(species_order)[:40]
        work_dir = Path.cwd() / f"sqs_{tag}"
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    sublattices = sublattices_arg(per_site_sub, fractions_per_orbit,
                                  species_order)

    if not quiet:
        print(f"\n[scraps-wizard] Plan:")
        print(f"  Template       : {template.name}  "
              f"({template.short_desc.split(' — ')[0]})")
        print(f"  Lattice param  : {lattice_param or template.a} A")
        print(f"  Cell           : {cell_dim} x {cell_dim} x {cell_dim} "
              f"= {n_atoms} atoms ({template_atoms} basis per primitive)")
        print(f"  Species        : " + ", ".join(species_order))
        print(f"  ELEMENT counts : " + ", ".join(
            f"{sp}={c}" for sp, c in zip(species_order, element_counts)
        ))
        print(f"  Sublattices    : {len(sublattices)} declared "
              f"({len(template.orbits)} Wyckoff orbits)")
        print(f"  MC budget      : AUTO_BUDGET={budget}  "
              f"(C++ fills MCSTEP / TOTALIter / NESTS / EARLY_STOP)")
        print(f"  MPI ranks      : {ranks}")
        print(f"  Work dir       : {work_dir}")
        print(f"\n[scraps-wizard] Running SCRAPs... ", end="", flush=True)

    import time
    t0 = time.time()
    sqs_atoms = run_sqs_from_atoms(
        primitive,
        species=species_order,
        element_counts=element_counts,
        cell_dim=(cell_dim, cell_dim, cell_dim),
        max_shellnum=2,
        weights=[1.0, 1.0],
        sublattices=sublattices if len(sublattices) > 1 else None,
        auto_budget=budget,
        rng_seed=1,
        ranks=ranks,
        work_dir=work_dir,
    )
    wall = time.time() - t0

    fitness = sqs_atoms.info.get("scraps_final_fitness")
    if not quiet:
        print(f"done in {wall:.1f} s.  Final fitness = "
              f"{fitness if fitness is not None else 'n/a'}.")

    # Export to user-friendly format.
    out_path = work_dir / f"sqs.{output_format}"
    sqs_atoms.write(str(out_path))
    if not quiet:
        print(f"[scraps-wizard] Wrote {out_path}")

    # Quick verification: compute alpha via verify_sro.py for the
    # single-sublattice case (the per-sublattice multi-sub calculator
    # lives in docs/compare_b2_multisub.py and is per-template; for
    # the wizard we just show whatever verify_sro outputs).
    if len(template.orbits) == 1 and not quiet:
        verify = REPO_ROOT / "tools" / "verify_sro.py"
        proc = subprocess.run(
            [sys.executable, str(verify),
             str(work_dir / "SCRAPS.vasp"),
             "--species", ",".join(species_order),
             "--shells", "2"],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode in (0, 1):
            # Trim the table to just the alpha lines.
            lines = proc.stdout.splitlines()
            print(f"\n[scraps-wizard] Per-shell alpha:")
            for line in lines:
                if "Shell" in line or any(sp in line for sp in species_order):
                    print(f"  {line}")
    return WizardResult(
        cell_dim=cell_dim, n_atoms=n_atoms, work_dir=work_dir,
        output_path=out_path, fitness=fitness,
        species_counts={sp: c for sp, c in zip(species_order, element_counts)},
    )


# ----- subcommands ---------------------------------------------------------

def cmd_list_templates(args) -> int:
    print("scraps-wizard built-in structure templates:\n")
    for t in TEMPLATES.values():
        print(f"  {t.name:<20}  {t.short_desc}")
        if t.notes:
            for line in t.notes.splitlines():
                print(f"    {line.rstrip()}")
        print(f"    default a = {t.a:.3f} A, "
              f"{len(t.orbits)} orbit(s)")
        print()
    return 0


def cmd_quick(args) -> int:
    if args.template not in TEMPLATES:
        print(f"Unknown template {args.template!r}. "
              f"Use `list-templates` to see options.", file=sys.stderr)
        return 2
    tmpl = TEMPLATES[args.template]
    composition_args = args.composition or []
    if len(composition_args) != len(tmpl.orbits):
        print(f"Template {tmpl.name!r} has {len(tmpl.orbits)} sublattices "
              f"({', '.join(o.user_name for o in tmpl.orbits)});"
              f" provide one --composition argument per sublattice "
              f"(got {len(composition_args)}).", file=sys.stderr)
        return 2
    fractions_per_orbit = []
    for arg in composition_args:
        try:
            fractions_per_orbit.append(parse_composition(arg))
        except ValueError as e:
            print(f"ERROR parsing --composition {arg!r}: {e}",
                  file=sys.stderr)
            return 2
    try:
        run_pipeline(
            tmpl, fractions_per_orbit,
            target_atoms=args.target_atoms,
            work_dir=args.work_dir,
            lattice_param=args.lattice_param,
            ranks=args.ranks,
            output_format=args.format,
            budget=args.budget,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_from_structure(args) -> int:
    """Read a POSCAR/CIF, run sublattice_detect to find Wyckoff orbits,
    prompt for composition per orbit, then run the pipeline."""
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        print("ERROR: pymatgen required. pip install pymatgen", file=sys.stderr)
        return 2

    struct_path = Path(args.structure)
    if not struct_path.exists():
        print(f"ERROR: file not found: {struct_path}", file=sys.stderr)
        return 2

    print(f"[scraps-wizard] Reading {struct_path}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        struct = Structure.from_file(str(struct_path))
    print(f"  Lattice: {struct.lattice.abc}, "
          f"angles {struct.lattice.angles}")
    print(f"  {len(struct)} sites, species: "
          f"{sorted(set(s.symbol for s in struct.species))}")

    # Detect Wyckoff orbits via sublattice_detect.py
    detect = REPO_ROOT / "tools" / "sublattice_detect.py"
    proc = subprocess.run(
        [sys.executable, str(detect), str(struct_path), "--json"],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        print(f"ERROR: sublattice_detect failed:\n{proc.stderr}",
              file=sys.stderr)
        return 1
    import json
    detection = json.loads(proc.stdout)
    print(f"  Space group: {detection['space_group_symbol']} "
          f"(#{detection['space_group_number']})")
    print(f"  Wyckoff orbits: {detection['n_orbits']}")

    fractions_per_orbit = []
    species_seen: set[str] = set()
    for orbit in detection["orbits"]:
        print(f"\n  Orbit {orbit['orbit_id']}: Wyckoff {orbit['wyckoff_letter']} "
              f"(multiplicity {orbit['multiplicity']}, sites {orbit['basis_indices']})")
        prompt = (f"  Composition for orbit {orbit['orbit_id']} "
                  f"(e.g. Ni:0.5,Co:0.5): ").format()
        if args.compositions and len(args.compositions) > orbit["orbit_id"]:
            raw = args.compositions[orbit["orbit_id"]]
            print(prompt + raw)
        else:
            raw = input(prompt).strip()
        if not raw:
            print(f"  ERROR: empty composition for orbit {orbit['orbit_id']}",
                  file=sys.stderr)
            return 2
        try:
            fracs = parse_composition(raw)
        except ValueError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 2
        overlap = species_seen & set(fracs)
        if overlap:
            print(f"  ERROR: species {overlap} already used on another orbit "
                  f"(disjoint species per sublattice required).", file=sys.stderr)
            return 2
        species_seen.update(fracs)
        fractions_per_orbit.append(fracs)

    # Build a template from the detected structure.
    primitive_orbits = []
    for i, orbit in enumerate(detection["orbits"]):
        first_basis_idx = orbit["basis_indices"][0]
        coords = orbit["frac_coords"][0]
        placeholder = struct.species[first_basis_idx].symbol
        primitive_orbits.append(TemplateOrbit(
            placeholder=placeholder,
            coords=tuple(coords),
            user_name=f"Wyckoff {orbit['wyckoff_letter']}",
            sublattice_id=i,
        ))
    custom = Template(
        name="custom",
        short_desc=f"from {struct_path.name}",
        space_group=detection["space_group_symbol"],
        a=float(min(struct.lattice.abc)),
        orbits=primitive_orbits,
    )
    try:
        run_pipeline(
            custom, fractions_per_orbit,
            target_atoms=args.target_atoms,
            work_dir=args.work_dir,
            lattice_param=float(min(struct.lattice.abc)),
            ranks=args.ranks,
            output_format=args.format,
            budget=args.budget,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_interactive(args) -> int:
    print("\n[scraps-wizard] Welcome. Let's design an SQS together.\n")
    print("What kind of structure do you want?")
    print("  1) Single-sublattice solid solution (FCC, BCC)")
    print("  2) Multi-sublattice ordered structure (B2, rocksalt, Heusler, perovskite)")
    print("  3) Use my own POSCAR/CIF file")
    choice = input("> ").strip()
    if choice == "3":
        path = input("Path to POSCAR/CIF: ").strip()
        ns = argparse.Namespace(
            structure=path, target_atoms=args.target_atoms,
            work_dir=args.work_dir, ranks=args.ranks,
            format=args.format, budget=args.budget,
            compositions=None,
        )
        return cmd_from_structure(ns)

    is_single = (choice == "1")
    if is_single:
        print("\nPick a lattice:")
        print("  1) FCC (Cu/Ag/Au, NiCoCr-type CCAs and HEAs)")
        print("  2) BCC (Fe/Cr, NbMoTaW refractory HEAs)")
        sub = input("> ").strip()
        template_name = {"1": "fcc", "2": "bcc"}.get(sub)
    else:
        print("\nPick a class:")
        multisubs = ["b2", "rocksalt", "heusler-l21", "perovskite-abo3"]
        for k, name in enumerate(multisubs, 1):
            print(f"  {k}) {name}  ({TEMPLATES[name].short_desc.split(' — ')[1]})")
        sub = input("> ").strip()
        try:
            template_name = multisubs[int(sub) - 1]
        except (ValueError, IndexError):
            template_name = None

    if template_name not in TEMPLATES:
        print("Unknown choice. Aborting.", file=sys.stderr)
        return 2
    tmpl = TEMPLATES[template_name]
    print(f"\nUsing template: {tmpl.name}  ({tmpl.short_desc})")
    if tmpl.notes:
        for line in tmpl.notes.splitlines():
            print(f"  Note: {line.rstrip()}")

    fractions_per_orbit = []
    for orb in tmpl.orbits:
        prompt = (f"\nComposition for {orb.user_name} "
                  f"(e.g. Ni:0.5,Co:0.5): ")
        raw = input(prompt).strip()
        try:
            fractions_per_orbit.append(parse_composition(raw))
        except ValueError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 2

    target = input(
        f"\nTarget number of atoms in the supercell [{args.target_atoms}]: "
    ).strip()
    target_atoms = int(target) if target else args.target_atoms

    try:
        run_pipeline(
            tmpl, fractions_per_orbit, target_atoms=target_atoms,
            work_dir=args.work_dir, ranks=args.ranks,
            output_format=args.format, budget=args.budget,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


# ----- CLI -----------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scraps-wizard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared options
    def add_common(sp):
        sp.add_argument("--target-atoms", type=int, default=100,
                        help="Target supercell size (default: 100). "
                             "Actual cell size is the closest "
                             "integer-feasible match.")
        sp.add_argument("--work-dir", type=Path, default=None,
                        help="Output directory (default: ./sqs_<species_tag>)")
        sp.add_argument("--ranks", type=int, default=4,
                        help="MPI ranks for SCRAPs (default 4)")
        sp.add_argument("--format", default="cif",
                        choices=["cif", "vasp", "xyz"],
                        help="Output structure format (default cif)")
        sp.add_argument("--budget", type=int, default=2,
                        choices=[1, 2, 3],
                        help="AUTO_BUDGET profile: 1=basic (fastest), "
                             "2=thorough (default), 3=exhaustive")

    sp = sub.add_parser("quick", help="One-shot SQS from a template")
    sp.add_argument("--template", required=True,
                    help="Template name (run `list-templates` for options)")
    sp.add_argument("--composition", action="append", default=[],
                    help="Composition per sublattice, e.g. Ni:0.5,Co:0.5. "
                         "Repeat for multi-sublattice (one per orbit, "
                         "in template orbit order).")
    sp.add_argument("--lattice-param", type=float, default=None,
                    help="Override the template's default lattice "
                         "parameter (Angstrom)")
    add_common(sp)
    sp.set_defaults(func=cmd_quick)

    sp = sub.add_parser("from-structure",
                        help="Take a POSCAR/CIF, auto-detect sublattices, "
                             "prompt for composition")
    sp.add_argument("structure", help="Path to POSCAR or CIF file")
    sp.add_argument("--composition", dest="compositions", action="append",
                    default=[],
                    help="Optional pre-filled compositions to skip prompts "
                         "(one per orbit in detection order)")
    add_common(sp)
    sp.set_defaults(func=cmd_from_structure)

    sp = sub.add_parser("interactive",
                        help="Step-by-step REPL (best for first-time users)")
    add_common(sp)
    sp.set_defaults(func=cmd_interactive)

    sp = sub.add_parser("list-templates",
                        help="Show the built-in structure templates")
    sp.set_defaults(func=cmd_list_templates)

    return p


def main() -> int:
    args = make_parser().parse_args()
    if not SCRAPS_BIN.exists() and args.cmd != "list-templates":
        print(f"ERROR: scraps binary not found at {SCRAPS_BIN}\n"
              f"Run ./build.sh from the repo root first.", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
