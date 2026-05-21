"""Tests for the SUBLATTICE INPUT keyword (parser + validation only).

This is the first wedge of multi-sublattice SQS support: the API and
its validation. Swap-path integration (per-sublattice randomizeLattice
and within-sublattice swap proposals) lands in a follow-up commit. Here
we lock in:

  - Absence of SUBLATTICE = legacy single-sublattice behavior (no
    banner, no validation tripped, identical optimization trajectory).
  - A single all-encompassing SUBLATTICE block (covering all basis
    positions with all species) parses cleanly and prints a banner. The
    run still uses the legacy flat-shuffle swap path -- behavior should
    be indistinguishable from no-SUBLATTICE in this trivial-partition
    case.
  - Disjointness errors: species in two sublattices, or basis position
    in two sublattices, abort with a clear message.
  - Range errors: species index >= TYP aborts.
  - Structural errors: empty BASIS, empty SPECIES, missing id, or
    tokens before BASIS/SPECIES section keyword.
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
    FCC conventional basis
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
            cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, timeout=60,
        )
    return proc.returncode, log_path.read_text()


def final_fitness(log: str) -> float | None:
    for line in log.splitlines():
        if line.startswith("FINAL fitness"):
            try:
                return float(line.split("=")[-1].strip())
            except ValueError:
                return None
    return None


# ---- Backward compat ------------------------------------------------------

def test_no_sublattice_keyword_runs_legacy(tmp_path):
    """No SUBLATTICE line => no banner, run succeeds, FCC 50:50 SQS
    converges to fit=0 (legacy single-sublattice path is untouched)."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   4
        MCSTEP  100
        TOTALIter 20
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 5
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"baseline failed: {log[-500:]}"
    assert "[SUBLATTICE]" not in log, "banner printed even though no SUBLATTICE keyword"
    assert final_fitness(log) == 0.0


# ---- Trivial-partition SUBLATTICE round-trips ------------------------------

def test_sublattice_full_partition_round_trips(tmp_path):
    """A single SUBLATTICE block covering every basis position with
    every species is structurally a no-op (same partition as the legacy
    path). Until the swap-path integration lands, this must run and
    reach fit=0 just like the no-SUBLATTICE case."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   4
        MCSTEP  100
        TOTALIter 20
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 5
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"sublattice round-trip failed: {log[-500:]}"
    assert "[SUBLATTICE] 1 sublattices declared:" in log
    assert "id=0 basis=[0,1,2,3] species=[0,1]" in log
    assert final_fitness(log) == 0.0


def test_sublattice_two_blocks_round_trip(tmp_path):
    """Two non-overlapping SUBLATTICE blocks parse and pass validation.
    Behavior at the optimizer level isn't yet sublattice-aware (that's
    the next commit) -- this test only locks in the parser + validator
    contract."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        SUBLATTICE 0 BASIS 0 1 SPECIES 0
        SUBLATTICE 1 BASIS 2 3 SPECIES 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"two-block parse failed: {log[-500:]}"
    assert "[SUBLATTICE] 2 sublattices declared:" in log
    assert "id=0 basis=[0,1] species=[0]" in log
    assert "id=1 basis=[2,3] species=[1]" in log


def test_sublattice_section_order_independent(tmp_path):
    """SPECIES <...> BASIS <...> should parse equivalently to BASIS first."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        SUBLATTICE 0 SPECIES 0 1 BASIS 0 1 2 3
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"section-order test failed: {log[-500:]}"
    assert "id=0 basis=[0,1,2,3] species=[0,1]" in log


# ---- Disjointness validation ----------------------------------------------

def test_sublattice_species_overlap_aborts(tmp_path):
    """Same species in two sublattices => abort with a disjointness error."""
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 BASIS 0 1 SPECIES 0 1
        SUBLATTICE 1 BASIS 2 3 SPECIES 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0, f"expected abort on species overlap; got rc=0\n{log[-500:]}"
    assert "species" in log.lower() and "disjoint" in log.lower()


def test_sublattice_basis_overlap_aborts(tmp_path):
    """Same basis position in two sublattices => abort."""
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 BASIS 0 1 2 SPECIES 0
        SUBLATTICE 1 BASIS 2 3   SPECIES 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "basis position 2" in log


def test_sublattice_duplicate_id_aborts(tmp_path):
    """Two SUBLATTICE blocks with the same id => abort."""
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 BASIS 0 1 SPECIES 0
        SUBLATTICE 0 BASIS 2 3 SPECIES 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "id 0" in log and "more than once" in log


# ---- Range / structural validation ----------------------------------------

def test_sublattice_species_out_of_range_aborts(tmp_path):
    """SUBLATTICE referencing species >= TYP => abort with range message."""
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1 5
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "species 5" in log and "out of range" in log


def test_sublattice_empty_basis_aborts(tmp_path):
    """SUBLATTICE with no BASIS section => abort."""
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 SPECIES 0 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "no BASIS" in log


def test_sublattice_empty_species_aborts(tmp_path):
    """SUBLATTICE with no SPECIES section => abort."""
    inp = textwrap.dedent("""\
        TYP     2
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 BASIS 0 1 2 3
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0
    assert "no SPECIES" in log


def test_sublattice_token_before_section_warns_and_ignores(tmp_path):
    """An integer or junk token before BASIS/SPECIES => line ignored
    (warning), but the run still proceeds because no valid sublattice
    was registered."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 99 BASIS 0 1 SPECIES 0
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"expected warn-and-continue; got rc={rc}\n{log[-500:]}"
    assert "before BASIS or SPECIES" in log
    assert "[SUBLATTICE]" not in log  # malformed line not registered


def test_sublattice_non_integer_token_warns_and_ignores(tmp_path):
    """Non-integer value in BASIS/SPECIES list => line ignored."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 BASIS 0 foo 2 SPECIES 0 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0
    assert "expected integer" in log
    assert "[SUBLATTICE]" not in log
