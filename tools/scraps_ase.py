"""ASE interoperability for SCRAPS.

The SCRAPS POSCAR/SCRAPS.vasp format is plain VASP-style and already
parses correctly with `ase.io.read(..., format='vasp')` -- the only
gotcha is that SCRAPS.vasp lacks species labels (atoms are indexed by
`ELEMENT` order from INPUT), so when reading you have to supply the
species list yourself. This module:

  - `read_scraps_output(path, species)` returns an `ase.Atoms` with
    proper species labels for SCRAPS-style POSCARs.
  - `write_scraps_poscar(atoms, path)` writes an ASE Atoms as a
    SCRAPS-compatible POSCAR (lattice + basis only; SCRAPS reads
    species counts from INPUT).
  - `make_scraps_inputs(...)` generates the INPUT + POSCAR pair from
    an ASE Atoms object plus composition.
  - `run_sqs_from_atoms(...)` is the end-to-end wrapper: ASE Atoms in,
    SCRAPS optimized ASE Atoms out.

ASE is a hard dependency of this module; the rest of `tools/` stays
pymatgen-only so the core test suite doesn't grow another dep.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional, Sequence

try:
    import ase
    from ase import Atoms
    from ase.io import read as ase_read, write as ase_write
except ImportError as exc:
    raise ImportError(
        "scraps_ase requires the `ase` package. Install with: pip install ase"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---- I/O ------------------------------------------------------------------

def read_scraps_output(
    path: os.PathLike,
    species: Optional[Sequence[str]] = None,
) -> Atoms:
    """Read a SCRAPS POSCAR/SCRAPS.vasp and return an ASE Atoms object.

    If `species` is provided, it's a list of element symbols indexed by
    SCRAPS's species number (matching `ELEMENT` order in INPUT). The
    function relabels the structure's atoms accordingly: the first
    `ELEMENT[0]` atoms become species[0], the next `ELEMENT[1]` become
    species[1], etc. This is the standard SCRAPS output convention.

    If `species` is None, ASE's built-in parser is used as-is; for
    SCRAPS-style POSCARs without a species line, that yields a
    structure of "X" placeholder atoms.
    """
    path = Path(path)

    # Try ASE's parser first; SCRAPS-style POSCARs lacking a species
    # label line (canonical VASP line 6) make it raise ParseError. We
    # then read the raw text, detect which layout it is, and inject the
    # user-supplied species names so ASE can parse it.
    try:
        atoms = ase_read(str(path), format="vasp")
        # If a species list was provided, ensure we relabel to match.
        if species is not None and len(atoms) > 0:
            # The count groups: derive from the POSCAR; ASE already
            # consumed them, but we re-parse for clarity.
            raw = path.read_text().splitlines()
            counts = _parse_counts_line(raw)
            atoms = _relabel(atoms, species, counts)
        return atoms
    except Exception:
        if species is None:
            raise
        # Fallback: inject species labels into a temp copy and retry.
        raw = path.read_text().splitlines()
        counts = _parse_counts_line(raw)
        if len(counts) != len(species):
            raise ValueError(
                f"Species count mismatch: POSCAR has {len(counts)} species "
                f"groups, got {len(species)} symbols ({species})"
            )
        # Build a canonical-VASP variant with a species line inserted
        # right before the counts line. Detect counts line index first.
        counts_idx = _find_counts_line_index(raw)
        species_line = " ".join(species)
        patched = raw[:counts_idx] + [species_line] + raw[counts_idx:]
        from io import StringIO
        return ase_read(StringIO("\n".join(patched) + "\n"), format="vasp")


def _parse_counts_line(raw: list[str]) -> list[int]:
    """Return the per-species integer counts from a VASP/SCRAPS POSCAR."""
    idx = _find_counts_line_index(raw)
    return [int(x) for x in raw[idx].split()]


def _find_counts_line_index(raw: list[str]) -> int:
    """Counts line is line 6 (0-indexed 5) in SCRAPS-style POSCAR,
    or line 7 (0-indexed 6) when a species line is present at index 5.
    Detect by integer-parseability."""
    for idx in (5, 6):
        if idx >= len(raw):
            continue
        try:
            [int(x) for x in raw[idx].split()]
            return idx
        except ValueError:
            continue
    raise ValueError("could not find per-species counts line in POSCAR")


def _relabel(atoms, species, counts):
    new_symbols: list[str] = []
    for sp, n in zip(species, counts):
        new_symbols.extend([sp] * n)
    if len(new_symbols) != len(atoms):
        raise ValueError(
            f"Relabeling length {len(new_symbols)} != atoms {len(atoms)}"
        )
    atoms.symbols = new_symbols
    return atoms


def write_scraps_poscar(atoms: Atoms, path: os.PathLike) -> None:
    """Write an ASE Atoms object as a SCRAPS-compatible POSCAR.

    SCRAPS' rewritten POSCAR parser (SCRAPs/readInput.cpp::readPOSCAR
    as of v0.2.0+) handles full canonical VASP 5 layout -- species
    label lines, multi-species count groups, Direct/Cartesian keyword,
    scale factors, Selective dynamics. So writing via ASE's standard
    `ase.io.write(..., format="vasp")` would work in principle.

    We still generate the file directly here for two reasons:
      1. Single count group with no species labels matches the existing
         tests/ format that we lock in via pytest, keeping the test
         POSCARs human-diffable.
      2. We write Cartesian positions, side-stepping any future doubt
         about Direct-keyword handling in a downstream toolchain that
         might still embed an older SCRAPS.
    """
    path = Path(path)
    cell = atoms.get_cell()
    cart = atoms.get_positions()  # Cartesian Angstroms
    lines = []
    lines.append("SCRAPS POSCAR generated from ASE Atoms")
    lines.append(" 1.0")
    for row in cell:
        lines.append(f" {row[0]:.16f} {row[1]:.16f} {row[2]:.16f}")
    lines.append(f" {len(atoms)}")
    # We write "Cartesian" to be self-documenting even though SCRAPS
    # ignores the keyword and would treat the numbers the same either way.
    lines.append("Cartesian")
    for r in cart:
        lines.append(f" {r[0]:.16f} {r[1]:.16f} {r[2]:.16f}")
    path.write_text("\n".join(lines) + "\n")


# ---- INPUT generation -----------------------------------------------------

def make_scraps_inputs(
    work_dir: os.PathLike,
    atoms: Atoms,
    element_counts: Sequence[int],
    cell_dim: tuple[int, int, int] = (1, 1, 1),
    *,
    max_shellnum: int = 2,
    weights: Optional[Sequence[float]] = None,
    nests: Optional[int] = None,
    mcstep: Optional[int] = None,
    auto_mcstep: int = 7,
    total_iter: Optional[int] = None,
    early_stop_patience: Optional[int] = None,
    early_stop_tol: float = 1e-6,
    rng_seed: int = 0,
    auto_budget: int = 0,
    sro_targets: Optional[Sequence[tuple[int, int, int, float]]] = None,
    spectator_basis_species: Optional[Sequence[tuple[int, int]]] = None,
    sublattices: Optional[Sequence[tuple[int, Sequence[int], Sequence[int]] | tuple[int, Sequence[int], Sequence[int], Sequence[int]]]] = None,
    figure_cutoffs: Optional[Sequence[float]] = None,
    figure_species: Optional[Sequence[str]] = None,
    figure_objective: str = "weighted_l1",
    figure_wd: float = 0.0,
    figure_wn: float = 1.0,
    figure_wr: float = 1.0,
    figure_tol: float = 1e-3,
    extra_lines: str = "",
) -> tuple[Path, Path]:
    """Write `POSCAR` and `INPUT` files into `work_dir` from ASE Atoms.

    `atoms` is the PRIMITIVE / basis unit cell (will be tiled by
    `cell_dim` inside SCRAPS). `element_counts` lists the per-species
    integer counts; their sum must equal `len(atoms) * prod(cell_dim)`.
    `sro_targets` is an optional list of (i, j, shell, alpha) tuples
    written as `SRO i j shell alpha` lines.

    `sublattices` is an optional list of (id, basis_positions, species_indices)
    tuples for multi-sublattice SQS (v0.4.1+). Each tuple becomes one
    `SUBLATTICE id BASIS b1 b2 ... SPECIES s1 s2 ...` line. Species and
    basis positions must be disjoint across sublattices in legacy pair mode; when omitted,
    SCRAPS uses the default single-sublattice behavior. In figure mode a fourth
    tuple member supplies integer local counts aligned with species_indices.
    Supply it for every group to allow shared species. These counts plus fixed
    spectators must sum to element_counts, and each group must fill its sites.

    `auto_budget` selects a one-keyword MC-budget profile (v0.4.5+):
    1 = basic, 2 = thorough, 3 = exhaustive. When set, the C++ side
    fills in any unset MCSTEP / TOTALIter / NESTS / EARLY_STOP_PATIENCE
    once the supercell size is known. Pass explicit `nests`, `mcstep`,
    `total_iter`, or `early_stop_patience` to override the auto-filled
    defaults selectively. Default 0 = off (the legacy mode where
    explicit values are required).

    `figure_cutoffs` enables the native cluster-function objective (pair,
    triplet, quadruplet diameter cutoffs in Angstrom). Requires icet 4.x and
    `figure_species` in ELEMENT order. It replaces the legacy pair objective;
    Shared species with explicit local counts and fixed spectators are supported;
    SRO targets are not.
    `figure_wd` controls exponential diameter decay, `figure_wn` the order
    penalty, and `figure_wr`/`figure_tol` the ATAT-style exact-match reward.

    Returns `(poscar_path, input_path)`.
    """
    # Apply explicit defaults only when AUTO_BUDGET is off. When
    # AUTO_BUDGET > 0 the C++ side fills any unset (= None) field from
    # the profile, so we omit those lines from INPUT to preserve the
    # "user wants AUTO_BUDGET defaults" signal.
    if auto_budget == 0:
        if nests is None: nests = 24
        if mcstep is None: mcstep = 100
        if total_iter is None: total_iter = 20
        if early_stop_patience is None: early_stop_patience = 3
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    poscar = work_dir / "POSCAR"
    input_path = work_dir / "INPUT"

    expected = len(atoms) * cell_dim[0] * cell_dim[1] * cell_dim[2]
    if sum(element_counts) != expected:
        raise ValueError(
            f"element_counts {tuple(element_counts)} sum to {sum(element_counts)} "
            f"but supercell will have {expected} atoms "
            f"({len(atoms)} basis x {cell_dim})"
        )

    if figure_cutoffs is None and (figure_species is not None or figure_objective != "weighted_l1"
            or figure_wd != 0 or figure_wn != 1 or figure_wr != 1 or figure_tol != 1e-3):
        raise ValueError("figure parameters require figure_cutoffs")
    if any(len(entry) == 4 for entry in (sublattices or [])) and figure_cutoffs is None:
        raise ValueError("explicit sublattice counts require figure_cutoffs")
    if figure_cutoffs is not None:
        if sro_targets:
            raise ValueError("figure mode requires random-alloy targets; SRO targets are unsupported")
        if figure_species is None:
            raise ValueError("figure_species is required in ELEMENT order for figure export")
        if figure_objective not in ("weighted_l1", "atat"):
            raise ValueError("figure_objective must be weighted_l1 or atat")
        import math
        if (not all(math.isfinite(x) for x in (figure_wd, figure_wn, figure_wr, figure_tol))
                or figure_wd < 0 or not 0 < figure_wn <= 1 or figure_wr < 0 or figure_tol < 0):
            raise ValueError("figure weights must be finite: wd/wr/tol >= 0 and 0 < wn <= 1")
        from higher_order import export_figures
        export_figures(work_dir / "FIGURES", atoms, figure_species, element_counts,
                       cell_dim, figure_cutoffs, sublattices=sublattices,
                       spectator_basis_species=spectator_basis_species)

    write_scraps_poscar(atoms, poscar)

    if weights is None:
        weights = [1.0] * max_shellnum
    weights_str = " ".join(f"{w:g}" for w in weights[:max_shellnum])
    element_str = " ".join(str(c) for c in element_counts)
    cell_dim_str = " ".join(str(k) for k in cell_dim)

    body = textwrap.dedent(f"""\
        TYP     {len(element_counts)}
        GITER   1000
        ELEMENT {element_str}
        Cell_DIM        {cell_dim_str}
        MAXSHELLNUM {max_shellnum}
        WEIGHT {weights_str}
        EARLY_STOP_TOL {early_stop_tol:g}
        """)
    # Per-keyword inclusion: omit lines for fields the user left as None
    # so AUTO_BUDGET can fill them in on the C++ side.
    if nests is not None:           body += f"NESTS   {nests}\n"
    if mcstep is not None:          body += f"MCSTEP  {mcstep}\n"
    if total_iter is not None:      body += f"TOTALIter {total_iter}\n"
    if early_stop_patience is not None:
        body += f"EARLY_STOP_PATIENCE {early_stop_patience}\n"
    if auto_mcstep > 0:             body += f"AUTO_MCSTEP {auto_mcstep}\n"
    if auto_budget > 0:             body += f"AUTO_BUDGET {auto_budget}\n"
    if rng_seed != 0:               body += f"RNG_SEED {rng_seed}\n"
    for entry in (spectator_basis_species or []):
        bi, sp = entry
        body += f"SPECTATOR_BASIS_SPECIES {bi} {sp}\n"
    for entry in (sublattices or []):
        if len(entry) not in (3, 4):
            raise ValueError("sublattice entries require id, basis, species, and optional counts")
        sub_id, basis_pos, species_idx = entry[:3]
        basis_str = " ".join(str(b) for b in basis_pos)
        species_str = " ".join(str(s) for s in species_idx)
        local_counts = " COUNTS " + " ".join(str(c) for c in entry[3]) if len(entry) == 4 else ""
        body += f"SUBLATTICE {sub_id} BASIS {basis_str} SPECIES {species_str}{local_counts}\n"
    for entry in (sro_targets or []):
        i, j, shell, alpha = entry
        body += f"SRO {i} {j} {shell} {alpha:g}\n"
    if figure_cutoffs is not None:
        body += (f"FIGURES_FILE FIGURES\nFIGURE_OBJECTIVE {figure_objective}\n"
                 f"FIGURE_WD {figure_wd:.17g}\nFIGURE_WN {figure_wn:.17g}\n"
                 f"FIGURE_WR {figure_wr:.17g}\nFIGURE_TOL {figure_tol:.17g}\n")
    if extra_lines:
        body += extra_lines if extra_lines.endswith("\n") else (extra_lines + "\n")

    input_path.write_text(body)
    return poscar, input_path


# ---- High-level wrapper ---------------------------------------------------

def _find_mpirun() -> Optional[str]:
    if env := os.environ.get("MPIRUN"):
        if Path(env).exists():
            return env
    mpicxx = os.environ.get("MPICXX", "/opt/homebrew/bin/mpic++")
    sibling = Path(mpicxx).parent / "mpirun"
    if sibling.exists():
        return str(sibling)
    return shutil.which("mpirun")


def run_sqs_from_atoms(
    atoms: Atoms,
    species: Sequence[str],
    element_counts: Sequence[int],
    cell_dim: tuple[int, int, int],
    *,
    work_dir: Optional[os.PathLike] = None,
    scraps_bin: Optional[os.PathLike] = None,
    mpirun: Optional[str] = None,
    ranks: int = 4,
    keep_work_dir: bool = True,
    **scraps_kwargs,
) -> Atoms:
    """End-to-end: takes ASE primitive Atoms in, returns SQS Atoms out.

    Generates POSCAR + INPUT, launches `mpirun -np <ranks> ./scraps` in
    a fresh work directory, reads back `SCRAPS.vasp`, relabels with
    `species`, and returns the resulting ASE Atoms.

    `scraps_kwargs` are forwarded to `make_scraps_inputs` (e.g.
    `max_shellnum`, `sro_targets`, `spectator_basis_species`,
    `rng_seed`, ...).
    """
    if work_dir is None:
        work_dir = Path.cwd() / f"scraps_ase_{int(time.time())}"
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if scraps_kwargs.get("figure_cutoffs") is not None:
        if "figure_species" in scraps_kwargs and list(scraps_kwargs["figure_species"]) != list(species):
            raise ValueError("figure_species must match the output species ordering")
        scraps_kwargs["figure_species"] = species
    make_scraps_inputs(work_dir, atoms, element_counts, cell_dim, **scraps_kwargs)

    if scraps_bin is None:
        scraps_bin = REPO_ROOT / "SCRAPs" / "scraps"
    scraps_bin = Path(scraps_bin)
    if not scraps_bin.exists():
        raise FileNotFoundError(
            f"scraps binary not found at {scraps_bin}; run ./build.sh first"
        )
    shutil.copy(scraps_bin, work_dir / "scraps")
    os.chmod(work_dir / "scraps", 0o755)

    if mpirun is None:
        mpirun = _find_mpirun()
    if mpirun is None:
        raise RuntimeError("no mpirun on PATH; set MPIRUN env var or pass mpirun=...")

    log_path = work_dir / "scraps.log"
    with log_path.open("w") as f:
        proc = subprocess.run(
            [mpirun, "-np", str(ranks), "./scraps"],
            cwd=work_dir, stdout=f, stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"scraps failed (rc={proc.returncode}); see {log_path}"
        )

    output = work_dir / "SCRAPS.vasp"
    if not output.exists():
        raise RuntimeError(f"SCRAPS produced no output at {output}; see {log_path}")
    result = read_scraps_output(output, species=species)
    # Stash useful metadata on the returned Atoms via .info
    result.info["scraps_work_dir"] = str(work_dir)
    result.info["scraps_log"] = str(log_path)
    for line in log_path.read_text().splitlines():
        if line.startswith("FINAL fitness"):
            try:
                result.info["scraps_final_fitness"] = float(line.split("=")[-1].strip())
            except Exception:
                pass
            break

    if not keep_work_dir:
        shutil.rmtree(work_dir)
        result.info.pop("scraps_work_dir", None)
        result.info.pop("scraps_log", None)
    return result
