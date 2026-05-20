"""Tests for SCRAPS' INPUT keyword parser (SCRAPs/readInput.cpp::input).

Pre-hardening, the parser silently accepted typos, missing required
keywords, malformed CELL_DIM, and ELEMENT-vs-TYP mismatches -- the
downstream code then segfaulted, hung, or produced wrong fitness
calculations. After the input() audit, each of these aborts cleanly
with a diagnostic naming the offending keyword.

Each test writes a minimal POSCAR + a deliberately-broken INPUT and
asserts that scraps exits non-zero with the expected error string in
its log.
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

# A valid POSCAR; we keep it constant across all tests since the bugs
# under test are in the INPUT parser, not POSCAR.
GOOD_POSCAR = textwrap.dedent("""\
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

GOOD_INPUT = textwrap.dedent("""\
    TYP     2
    GITER   100
    ELEMENT 8 8
    Cell_DIM        2 2 2
    NESTS   4
    MCSTEP  20
    TOTALIter 2
    MAXSHELLNUM 1
    WEIGHT 1
    """)


def run_scraps(work_dir: Path, input_text: str, poscar: str = GOOD_POSCAR,
               ranks: int = 1) -> tuple[int, str]:
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


# ---- Backward-compat: the GOOD_INPUT must still run ------------------------

def test_good_input_still_runs(tmp_path):
    rc, log = run_scraps(tmp_path, GOOD_INPUT)
    assert rc == 0, f"baseline good INPUT failed: {log[-500:]}"


def test_comments_and_blank_lines_skipped(tmp_path):
    """Lines starting with # and blank lines must be silently skipped,
    NOT trigger an unrecognized-keyword warning."""
    inp = textwrap.dedent("""\
        # This is a comment
        TYP     2

        # Another comment between params
        GITER   100
        ELEMENT 8 8
        Cell_DIM        2 2 2
        NESTS   4
        MCSTEP  20
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"comments-and-blanks INPUT failed: {log[-500:]}"
    assert "unrecognized" not in log.lower(), (
        f"comment/blank line was flagged as unrecognized:\n{log[-500:]}")


# ---- Unrecognized keywords ------------------------------------------------

def test_unknown_keyword_warns_but_runs(tmp_path):
    """Typo in keyword name should produce a warning but not abort -- the
    optimizer can still run with the surrounding valid lines."""
    inp = GOOD_INPUT + "EARLY_STOP_PATEINCE 5\n"  # typo: PATEINCE
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed on warn-only typo: {log[-500:]}"
    assert "unrecognized" in log.lower() and "EARLY_STOP_PATEINCE" in log


# ---- Required-keyword validation ------------------------------------------

@pytest.mark.parametrize("missing_line,expected_kw", [
    ("TYP", "TYP"),
    ("NESTS", "NESTS"),
    ("MCSTEP", "MCSTEP"),
    ("TOTALIter", "TOTALIter"),
    ("MAXSHELLNUM", "MAXSHELLNUM"),
])
def test_missing_required_keyword_aborts(missing_line, expected_kw, tmp_path):
    """Dropping any of the must-have keywords should produce a clear
    abort naming that keyword."""
    inp = "\n".join(
        line for line in GOOD_INPUT.splitlines()
        if not line.strip().startswith(missing_line)
    ) + "\n"
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0, f"expected abort for missing {missing_line}; got rc=0"
    assert expected_kw in log, (
        f"abort log missing the offending keyword name: {log[-500:]}")


def test_element_count_mismatch_aborts(tmp_path):
    """ELEMENT must list exactly TYP integers."""
    inp = textwrap.dedent("""\
        TYP     3
        ELEMENT 8 8
        Cell_DIM        2 2 2
        NESTS   4
        MCSTEP  20
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "ELEMENT" in log and "TYP" in log


def test_cell_dim_must_have_three_entries(tmp_path):
    """Cell_DIM with !=3 integers should abort."""
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 8 8
        Cell_DIM        2 2
        NESTS   4
        MCSTEP  20
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "Cell_DIM" in log


def test_negative_weight_aborts(tmp_path):
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 8 8
        Cell_DIM        2 2 2
        NESTS   4
        MCSTEP  20
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT -1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "WEIGHT" in log


def test_weight_too_few_entries_aborts(tmp_path):
    """WEIGHT must have at least MAXSHELLNUM entries."""
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 8 8
        Cell_DIM        2 2 2
        NESTS   4
        MCSTEP  20
        TOTALIter 2
        MAXSHELLNUM 3
        WEIGHT 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "WEIGHT" in log


def test_cell_dim_zero_aborts(tmp_path):
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 8 8
        Cell_DIM        2 0 2
        NESTS   4
        MCSTEP  20
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "Cell_DIM" in log
