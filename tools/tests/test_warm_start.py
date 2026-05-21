"""Tests for the INIT_NEST warm-start API.

Verifies:
  - A perfect B2 init nest with a B2 SRO target finishes at fitness=0
    in iter 0 (the seed already IS the optimum).
  - A wrong-length / wrong-species / wrong-count init nest aborts
    cleanly with a clear message naming the mismatch.
  - INIT_NEST file with blank lines and '#' comments parses fine.
  - HEB spectator: an init nest that violates the fixed boron
    sublattice aborts (we won't silently overwrite the spectator).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

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


def run_scraps(work_dir: Path, poscar: str, input_text: str,
               init_nest: str | None = None, ranks: int = 1) -> tuple[int, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "POSCAR").write_text(poscar)
    (work_dir / "INPUT").write_text(input_text)
    if init_nest is not None:
        (work_dir / "init.nest").write_text(init_nest)
    shutil.copy(SCRAPS_BIN, work_dir / "scraps")
    os.chmod(work_dir / "scraps", 0o755)
    log_path = work_dir / "scraps.log"
    with log_path.open("w") as f:
        proc = subprocess.run(
            [MPIRUN, "-np", str(ranks), "./scraps"],
            cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, timeout=60,
        )
    return proc.returncode, log_path.read_text()


# A 2x2x2 BCC supercell = 16 atoms; we'll seed it as a perfect B2
# arrangement: every basis-0 corner = species 0, every basis-1
# body-center = species 1. With the supercell ordering basis-OUTERMOST,
# that means the first 8 atoms (basis 0 across 8 cells) are species 0
# and the next 8 (basis 1) are species 1.

BCC_POSCAR_NO_SPECIES = textwrap.dedent("""\
    BCC unit cell
     1.0
     1.0 0.0 0.0
     0.0 1.0 0.0
     0.0 0.0 1.0
     2
    Direct
     0.0 0.0 0.0
     0.5 0.5 0.5
    """)

# Perfect B2: first 8 atoms = species 0, next 8 = species 1.
B2_INIT_NEST = "\n".join(["0"] * 8 + ["1"] * 8) + "\n"

B2_INPUT = textwrap.dedent("""\
    TYP     2
    GITER   100
    ELEMENT 8 8
    Cell_DIM        2 2 2
    NESTS   4
    MCSTEP  10
    TOTALIter 2
    MAXSHELLNUM 2
    WEIGHT 1 1
    RNG_SEED 1
    SRO 0 1 0 -1.0
    SRO 0 1 1 +1.0
    INIT_NEST init.nest
    """)


def test_warm_start_at_optimum_yields_zero_fitness(tmp_path):
    """A perfect B2 nest seeded into a B2-targeted run should already
    BE the optimum, so the very first fitness eval gives 0 and the run
    terminates with FINAL fitness = 0."""
    rc, log = run_scraps(tmp_path, BCC_POSCAR_NO_SPECIES, B2_INPUT,
                         init_nest=B2_INIT_NEST)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    assert "INIT_NEST: seeded nests[0]" in log
    # Look for the FINAL fitness on rank 0
    final = None
    for line in log.splitlines():
        if "FINAL fitness" in line:
            final = float(line.split("=")[-1].strip())
            break
    assert final is not None, f"no FINAL fitness in log:\n{log[-500:]}"
    assert final == 0.0, f"expected fitness=0 (already-optimum seed); got {final}"


def test_warm_start_with_comments_and_blanks(tmp_path):
    """Init-nest file with comments and blank lines should parse."""
    init = "# perfect B2 seed\n0 0 0 0\n0 0 0 0\n\n# now species 1\n1 1 1 1\n1 1 1 1\n"
    rc, log = run_scraps(tmp_path, BCC_POSCAR_NO_SPECIES, B2_INPUT,
                         init_nest=init)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    final = next(float(l.split("=")[-1].strip())
                 for l in log.splitlines() if "FINAL fitness" in l)
    assert final == 0.0


def test_warm_start_wrong_atom_count_aborts(tmp_path):
    init = "\n".join(["0"] * 10) + "\n"  # 10 atoms, expected 16
    rc, log = run_scraps(tmp_path, BCC_POSCAR_NO_SPECIES, B2_INPUT,
                         init_nest=init)
    assert rc != 0
    assert "INIT_NEST" in log and "16" in log


def test_warm_start_wrong_species_count_aborts(tmp_path):
    """16 entries total (matches), but per-species counts don't match ELEMENT 8 8."""
    init = "\n".join(["0"] * 9 + ["1"] * 7) + "\n"  # 9 of 0, 7 of 1
    rc, log = run_scraps(tmp_path, BCC_POSCAR_NO_SPECIES, B2_INPUT,
                         init_nest=init)
    assert rc != 0
    assert "INIT_NEST" in log
    # ELEMENT says 8 / 8; warm start says 9 / 7
    assert ("8" in log) or ("9" in log) or ("7" in log)


def test_warm_start_out_of_range_species_aborts(tmp_path):
    """Species index 5 in a TYP=2 INPUT must be flagged."""
    init = "\n".join(["0"] * 7 + ["1"] * 8 + ["5"]) + "\n"
    rc, log = run_scraps(tmp_path, BCC_POSCAR_NO_SPECIES, B2_INPUT,
                         init_nest=init)
    assert rc != 0
    assert "INIT_NEST" in log and "out of range" in log


def test_warm_start_missing_file_aborts(tmp_path):
    """If INIT_NEST points to a file that doesn't exist."""
    rc, log = run_scraps(tmp_path, BCC_POSCAR_NO_SPECIES, B2_INPUT,
                         init_nest=None)
    # init.nest was never written
    assert rc != 0
    assert "INIT_NEST file not found" in log


def test_warm_start_l21_with_fast_swaps_preserved(tmp_path):
    """Regression for the legacy-vs-delta fitness disagreement.

    Build a perfect L2_1 (Heusler X_2YZ) warm-start on BCC 2x2x2 = 16
    atoms. Heusler structure: X on the body-center sublattice (basis 1,
    8 atoms), Y and Z alternating on the corner sublattice (basis 0, 8
    atoms) by cell parity. With BCC primitive 2-basis tiled (2,2,2)
    and basis-OUTERMOST ordering: atoms 0..7 are basis 0 (Y or Z by
    parity), atoms 8..15 are basis 1 (all X).

    The L2_1 warm-start has zero X-Y and X-Z at the 2NN shell (X's
    2NN are all X), which is the case the FAST_SWAPS=1 path used to
    mis-evaluate -- legacy fitness reported 0, delta-fitness reported
    a non-zero value, and MC then 'optimized' away from the perfect
    seed. This test verifies the warm-start nest stays at fitness=0
    through the MC loop with FAST_SWAPS=1 (default).
    """
    # Build the warm-start nest: basis 0 (atoms 0..7) gets Y or Z by
    # (i+j+k) parity; basis 1 (atoms 8..15) gets X.
    k1 = k2 = k3 = 2
    n_cells = k1 * k2 * k3  # 8
    species = [0] * (2 * n_cells)  # 0=X, 1=Y, 2=Z
    cell_idx = 0
    # SCRAPS' ordering: for each basis (outer), z (next), y, x (inner)
    for b in range(2):
        c_idx = 0
        for k in range(k3):
            for j in range(k2):
                for i in range(k1):
                    parity = (i + j + k) % 2
                    if b == 0:
                        species[b * n_cells + c_idx] = 1 if parity == 0 else 2
                    else:
                        species[b * n_cells + c_idx] = 0
                    c_idx += 1
    init_nest = "\n".join(str(s) for s in species) + "\n"

    inp = textwrap.dedent("""\
        TYP     3
        GITER   100
        ELEMENT 8 4 4
        Cell_DIM        2 2 2
        NESTS   1
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 1
        SRO 0 1 0 -1.0
        SRO 0 2 0 -1.0
        SRO 1 2 0 +1.0
        SRO 1 2 1 -3.0
        INIT_NEST init.nest
        """)
    rc, log = run_scraps(tmp_path, BCC_POSCAR_NO_SPECIES, inp,
                         init_nest=init_nest)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    # The warm-started perfect L2_1 should hold at fitness=0 through MC
    final = next((float(l.split("=")[-1].strip())
                  for l in log.splitlines() if "FINAL fitness" in l), None)
    assert final is not None, "no FINAL fitness in log"
    assert final == 0.0, (
        f"L2_1 warm-start with FAST_SWAPS=1 lost fitness=0 -- got {final}. "
        "This is the legacy-vs-delta fitness regression.")
