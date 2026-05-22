"""Tests for the AUTO_BUDGET INPUT keyword (v0.4.5).

AUTO_BUDGET 1/2/3 sets MCSTEP, TOTALIter, NESTS, and
EARLY_STOP_PATIENCE from a per-profile default when the user did NOT
explicitly specify them. Explicit values always win. Provides a one-
keyword "I don't want to tune anything" entry point for first-time
users; the C++ side fills the defaults in once total_atoms is known.

Profiles (chosen for empirical coverage; not theoretically derived):
  - 1 basic:      MCSTEP=8*N,   TOTALIter=15, NESTS=4,  patience=3
  - 2 thorough:   MCSTEP=20*N,  TOTALIter=25, NESTS=8,  patience=5
  - 3 exhaustive: MCSTEP=50*N,  TOTALIter=40, NESTS=16, patience=8
"""
from __future__ import annotations

import os
import shutil
import subprocess
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


FCC_POSCAR = textwrap.dedent("""\
    FCC conventional cell
    1.0
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    4
    Direct
    0.0 0.0 0.0
    0.5 0.5 0.0
    0.5 0.0 0.5
    0.0 0.5 0.5
    """)


def run_scraps(work_dir: Path, input_text: str, ranks: int = 1) -> tuple[int, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "POSCAR").write_text(FCC_POSCAR)
    (work_dir / "INPUT").write_text(input_text)
    shutil.copy(SCRAPS_BIN, work_dir / "scraps")
    os.chmod(work_dir / "scraps", 0o755)
    log_path = work_dir / "scraps.log"
    with log_path.open("w") as f:
        proc = subprocess.run(
            [MPIRUN, "-np", str(ranks), "./scraps"],
            cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, timeout=120,
        )
    return proc.returncode, log_path.read_text()


# ---------------------------------------------------------------------------
# Smoke: AUTO_BUDGET without MCSTEP/TOTALIter/NESTS runs cleanly
# ---------------------------------------------------------------------------

def test_auto_budget_basic_minimal_input(tmp_path):
    """Minimal INPUT: just TYP, ELEMENT, Cell_DIM, MAXSHELLNUM, WEIGHT,
    plus AUTO_BUDGET 1. No MCSTEP, TOTALIter, NESTS. Run should
    complete cleanly with the basic profile."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM 2 2 2
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        AUTO_BUDGET 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed with AUTO_BUDGET 1: {log[-500:]}"
    assert "AUTO_BUDGET=1 (basic)" in log
    # The basic profile sets MCSTEP = 8 * 32 atoms = 256.
    assert "MCSTEP = 256" in log
    assert "TOTALIter = 15" in log
    assert "NESTS = 4" in log
    assert "EARLY_STOP_PATIENCE = 3" in log


def test_auto_budget_thorough_reaches_sqs(tmp_path):
    """AUTO_BUDGET 2 (thorough profile) should reach fit=0 on the
    feasible FCC 50:50 SQS without any manual tuning."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM 2 2 2
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        AUTO_BUDGET 2
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0
    assert "AUTO_BUDGET=2 (thorough)" in log
    fit_line = next(l for l in log.splitlines() if l.startswith("FINAL fitness"))
    fit = float(fit_line.split("=")[-1].strip())
    assert fit == 0.0, f"AUTO_BUDGET 2 didn't reach SQS: fit={fit}"


# ---------------------------------------------------------------------------
# Explicit values override AUTO_BUDGET defaults
# ---------------------------------------------------------------------------

def test_auto_budget_explicit_mcstep_wins(tmp_path):
    """If MCSTEP is given explicitly alongside AUTO_BUDGET, the user
    value wins (only the unset fields get filled in)."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM 2 2 2
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        MCSTEP 999
        AUTO_BUDGET 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0
    # Explicit MCSTEP=999 must NOT be overridden by AUTO_BUDGET 1's
    # default of 8 * 32 = 256. AUTO_BUDGET should still log TOTALIter
    # / NESTS / EARLY_STOP_PATIENCE defaults (those weren't explicit).
    assert "MCSTEP = 256" not in log, (
        f"AUTO_BUDGET overrode an explicit MCSTEP: {log[-500:]}"
    )
    assert "TOTALIter = 15" in log
    assert "NESTS = 4" in log


def test_auto_budget_partial_explicit(tmp_path):
    """Explicit MCSTEP + NESTS; let AUTO_BUDGET fill the rest."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM 2 2 2
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        MCSTEP 100
        NESTS 6
        AUTO_BUDGET 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0
    # Both explicit; AUTO_BUDGET fills only TOTALIter + EARLY_STOP_PATIENCE.
    assert "MCSTEP = 256" not in log
    assert "NESTS = 4" not in log
    assert "TOTALIter = 15" in log
    assert "EARLY_STOP_PATIENCE = 3" in log


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_auto_budget_out_of_range_aborts(tmp_path):
    """AUTO_BUDGET values outside {1, 2, 3} should abort with a clear
    diagnostic."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM 2 2 2
        MAXSHELLNUM 1
        WEIGHT 1
        AUTO_BUDGET 5
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "AUTO_BUDGET = 5" in log
    assert "out of range" in log


def test_no_auto_budget_still_requires_mcstep(tmp_path):
    """Without AUTO_BUDGET, MCSTEP / TOTALIter / NESTS are still
    required (legacy behavior preserved)."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM 2 2 2
        MAXSHELLNUM 1
        WEIGHT 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    # The first missing-required keyword to be checked is NESTS.
    assert "NESTS" in log or "MCSTEP" in log or "TOTALIter" in log
