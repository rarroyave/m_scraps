"""Tests for SCRAPS' POSCAR parser (SCRAPs/readInput.cpp::readPOSCAR).

These tests feed pathological POSCAR variants to scraps and assert it
either parses them correctly (when valid) or aborts with a clear
message (when malformed). Pre-rewrite, scraps silently mis-read most
of these, leaving the basis atoms in the wrong positions; the
end-to-end alpha calculation then gave nonsense.

Each test:
  1. Writes a POSCAR variant + minimal INPUT
  2. Runs mpirun -np 1 ./scraps (a single rank is enough; the parser
     runs on every rank anyway)
  3. Checks the SCRAPS.vasp output -- the lattice and basis should
     match what pymatgen reads from the *input* POSCAR
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
SCRAPS_BIN = REPO / "SCRAPs" / "scraps"


def find_mpirun() -> str | None:
    if env := os.environ.get("MPIRUN"):
        if Path(env).exists():
            return env
    mpicxx = os.environ.get("MPICXX", "/opt/homebrew/bin/mpic++")
    sibling = Path(mpicxx).parent / "mpirun"
    if sibling.exists():
        return str(sibling)
    return shutil.which("mpirun")


MPIRUN = find_mpirun()

pytestmark = pytest.mark.skipif(
    not SCRAPS_BIN.exists() or MPIRUN is None,
    reason="scraps binary not built or no mpirun",
)

# Minimal INPUT that runs in well under a second and doesn't care about
# species ordering -- the parser tests just need scraps to read the
# POSCAR, run a tiny optimization, and write SCRAPS.vasp so we can
# inspect the basis positions.
MINIMAL_INPUT = textwrap.dedent("""\
    TYP     2
    GITER   100
    ELEMENT 1 1
    Cell_DIM        1 1 1
    NESTS   2
    MCSTEP  10
    TOTALIter 1
    MAXSHELLNUM 1
    WEIGHT 1
    RNG_SEED 1
    """)


def run_scraps(work_dir: Path, poscar: str, input_text: str = MINIMAL_INPUT,
               ranks: int = 1) -> tuple[int, str]:
    """Run scraps; return (returncode, log_text)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "POSCAR").write_text(poscar)
    (work_dir / "INPUT").write_text(input_text)
    shutil.copy(SCRAPS_BIN, work_dir / "scraps")
    os.chmod(work_dir / "scraps", 0o755)
    log_path = work_dir / "scraps.log"
    with log_path.open("w") as f:
        proc = subprocess.run(
            [MPIRUN, "-np", str(ranks), "./scraps"],
            cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, timeout=60,
        )
    return proc.returncode, log_path.read_text()


def read_lattice_and_basis(path: Path):
    """Return (3x3 lattice in Angstrom, Nx3 Cartesian positions in Angstrom).
    Uses pymatgen as the reference parser."""
    from pymatgen.io.vasp import Poscar
    import warnings
    from pymatgen.io.vasp.inputs import BadPoscarWarning
    warnings.filterwarnings("ignore", category=BadPoscarWarning)
    s = Poscar.from_file(str(path), check_for_potcar=False).structure
    return np.array(s.lattice.matrix), np.array(s.cart_coords)


def assert_supercell_matches_input(work_dir: Path, cell_dim=(1, 1, 1)):
    """The output SCRAPS.vasp should have lattice = cell_dim * input lattice
    and basis positions equal to the input basis tiled over Cell_DIM.

    For Cell_DIM (1, 1, 1) this is just an identity check: input lattice
    and basis match the output lattice and basis (modulo species ordering).
    """
    in_lat, in_basis = read_lattice_and_basis(work_dir / "POSCAR")
    out_lat, out_basis = read_lattice_and_basis(work_dir / "SCRAPS.vasp")

    expected_lat = in_lat * np.array(cell_dim)[:, None]
    assert np.allclose(out_lat, expected_lat, atol=1e-6), (
        f"output lattice mismatch:\n  expected:\n{expected_lat}\n  got:\n{out_lat}"
    )

    # Tile the input basis. Cartesian positions of every tiled image must
    # appear in the output basis (set equality up to FP tolerance).
    tiled = []
    for ki in range(cell_dim[0]):
        for kj in range(cell_dim[1]):
            for kk in range(cell_dim[2]):
                offset = ki * in_lat[0] + kj * in_lat[1] + kk * in_lat[2]
                for b in in_basis:
                    tiled.append(b + offset)
    tiled = np.array(tiled)

    # Match every output atom to a tiled position (allowing reorder).
    matched = np.zeros(len(tiled), dtype=bool)
    for o in out_basis:
        d = np.linalg.norm(tiled - o, axis=1)
        idx = np.argmin(d)
        assert d[idx] < 1e-4, f"output basis atom {o} has no match in tiled input"
        matched[idx] = True
    assert matched.all(), "some tiled input atoms didn't appear in output"


# ---- Pre-fix bugs that should now pass --------------------------------------

def test_unit_lattice_no_species_line(tmp_path):
    """The SCRAPS-internal dialect (unit lattice, no species line, single count).
    This is the only variant the old parser handled; it must still work."""
    poscar = textwrap.dedent("""\
        Cubic test
         1.0
         1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 1.0
         2
        Direct
         0.0 0.0 0.0
         0.5 0.5 0.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    assert_supercell_matches_input(tmp_path)


def test_direct_with_realistic_lattice(tmp_path):
    """REGRESSION: pre-fix the parser ignored the Direct keyword, so basis
    atoms at fractional (0.5, 0.5, 0.5) ended up at Cartesian (0.5, 0.5, 0.5)
    instead of (a/2, a/2, a/2) for a non-unit lattice."""
    poscar = textwrap.dedent("""\
        BCC with a=3.0
         1.0
         3.0 0.0 0.0
         0.0 3.0 0.0
         0.0 0.0 3.0
         2
        Direct
         0.0 0.0 0.0
         0.5 0.5 0.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    # The second atom must be at (1.5, 1.5, 1.5) Cartesian, not (0.5, 0.5, 0.5).
    _, out_basis = read_lattice_and_basis(tmp_path / "SCRAPS.vasp")
    found = False
    for b in out_basis:
        if np.allclose(b, [1.5, 1.5, 1.5], atol=1e-4):
            found = True
            break
    assert found, f"expected basis atom at (1.5,1.5,1.5); got {out_basis}"
    # Definitely should NOT be at (0.5, 0.5, 0.5)
    for b in out_basis:
        assert not np.allclose(b, [0.5, 0.5, 0.5], atol=1e-4), (
            f"basis atom at (0.5,0.5,0.5) means Direct keyword was ignored")


def test_cartesian_with_realistic_lattice(tmp_path):
    """Explicit Cartesian coords should be honored as written."""
    poscar = textwrap.dedent("""\
        BCC Cartesian
         1.0
         3.0 0.0 0.0
         0.0 3.0 0.0
         0.0 0.0 3.0
         2
        Cartesian
         0.0 0.0 0.0
         1.5 1.5 1.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    assert_supercell_matches_input(tmp_path)


def test_positive_scale_factor_multiplier(tmp_path):
    """REGRESSION: scale factor on line 2 was read but not applied.
    Scale 2.0 with unit lattice should produce lattice = (2,2,2)."""
    poscar = textwrap.dedent("""\
        Scale-factor test
         2.0
         1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 1.0
         2
        Direct
         0.0 0.0 0.0
         0.5 0.5 0.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    out_lat, out_basis = read_lattice_and_basis(tmp_path / "SCRAPS.vasp")
    assert np.allclose(out_lat, 2.0 * np.eye(3), atol=1e-6), (
        f"expected lattice = 2*I; got {out_lat}")
    # Direct (0.5, 0.5, 0.5) in a 2x2x2 cell = Cartesian (1, 1, 1)
    found = any(np.allclose(b, [1.0, 1.0, 1.0], atol=1e-4) for b in out_basis)
    assert found, f"expected basis at (1,1,1); got {out_basis}"


def test_negative_scale_factor_target_volume(tmp_path):
    """Scale = -27 means rescale uniformly so |det(lattice)| = 27.
    With base lattice |det| = 1, linear scale = 3."""
    poscar = textwrap.dedent("""\
        Target volume = 27
        -27.0
         1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 1.0
         2
        Direct
         0.0 0.0 0.0
         0.5 0.5 0.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    out_lat, _ = read_lattice_and_basis(tmp_path / "SCRAPS.vasp")
    assert np.isclose(abs(np.linalg.det(out_lat)), 27.0, atol=1e-4), (
        f"expected |det|=27; got |det|={abs(np.linalg.det(out_lat))}")


def test_species_label_line_present(tmp_path):
    """REGRESSION: pre-fix, a species-label line on row 6 caused scraps to
    parse the species names as integers and crash or mis-count atoms."""
    poscar = textwrap.dedent("""\
        With species labels
         1.0
         1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 1.0
         Cu Zn
         1 1
        Direct
         0.0 0.0 0.0
         0.5 0.5 0.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    assert_supercell_matches_input(tmp_path)


def test_multiple_species_count_groups(tmp_path):
    """REGRESSION: pre-fix, `ss >> totalatoms` only took the first integer,
    so a multi-species count line like '1 1' would set totalatoms=1 and
    miss the second atom."""
    poscar = textwrap.dedent("""\
        Two species, two count integers
         1.0
         1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 1.0
         1 1
        Direct
         0.0 0.0 0.0
         0.5 0.5 0.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    # Must have read 2 atoms total.
    _, out_basis = read_lattice_and_basis(tmp_path / "SCRAPS.vasp")
    assert len(out_basis) == 2, f"expected 2 atoms in supercell; got {len(out_basis)}"


def test_selective_dynamics_line(tmp_path):
    """REGRESSION: a 'Selective dynamics' line between counts and coord-type
    used to be read as the coord-type, then the actual 'Direct' line was
    read as a position and crashed parsing."""
    poscar = textwrap.dedent("""\
        Selective dynamics variant
         1.0
         3.0 0.0 0.0
         0.0 3.0 0.0
         0.0 0.0 3.0
         2
        Selective dynamics
        Direct
         0.0 0.0 0.0  T T T
         0.5 0.5 0.5  T T T
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    _, out_basis = read_lattice_and_basis(tmp_path / "SCRAPS.vasp")
    assert len(out_basis) == 2
    found = any(np.allclose(b, [1.5, 1.5, 1.5], atol=1e-4) for b in out_basis)
    assert found, f"expected (1.5,1.5,1.5) basis atom; got {out_basis}"


# ---- Malformed POSCARs should abort with a clear message --------------------

def test_truncated_poscar_aborts(tmp_path):
    poscar = "comment\n 1.0\n 1 0 0\n 0 1 0\n"
    rc, log = run_scraps(tmp_path, poscar)
    assert rc != 0
    assert "POSCAR" in log


def test_bad_coord_type_aborts(tmp_path):
    poscar = textwrap.dedent("""\
        Bad coord-type keyword
         1.0
         1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 1.0
         1
        Magic
         0.0 0.0 0.0
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc != 0
    assert "Direct" in log or "Cartesian" in log


def test_truncated_atoms_aborts(tmp_path):
    poscar = textwrap.dedent("""\
        Claims 5 atoms but only gives 2
         1.0
         1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 1.0
         5
        Direct
         0.0 0.0 0.0
         0.5 0.5 0.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc != 0
    assert "atom" in log.lower() or "POSCAR" in log


# ---- Non-orthogonal lattice (HEB hexagonal-style) ---------------------------

def test_non_orthogonal_lattice_direct(tmp_path):
    """Hexagonal cell with Direct coords: basis atom at fractional (1/3, 2/3, 0)
    must end up at the right Cartesian position after the lattice transform."""
    poscar = textwrap.dedent("""\
        Hexagonal a=3, c=3.3
         1.0
         3.0  0.0           0.0
        -1.5  2.598076211   0.0
         0.0  0.0           3.3
         2
        Direct
         0.0      0.0      0.0
         0.333333 0.666667 0.5
        """)
    rc, log = run_scraps(tmp_path, poscar)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    _, out_basis = read_lattice_and_basis(tmp_path / "SCRAPS.vasp")
    # Expected Cartesian for (1/3, 2/3, 0.5):
    #   x = 1/3 * 3 + 2/3 * (-1.5) = 1 - 1 = 0
    #   y = 1/3 * 0 + 2/3 * 2.598076211 = 1.732050807
    #   z = 0 + 0 + 0.5 * 3.3 = 1.65
    expected = np.array([0.0, 1.732050807, 1.65])
    found = any(np.allclose(b, expected, atol=1e-3) for b in out_basis)
    assert found, f"expected basis atom near {expected}; got {out_basis}"
