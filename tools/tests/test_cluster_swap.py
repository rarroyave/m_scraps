"""Tests for the K-atom cluster swap move (CLUSTER_SWAP_K).

Verifies:
  - K=1 (default) is unchanged from the existing single-swap behavior.
  - K=2 reaches the same plateau as K=1 on the infeasible 25:50:25
    n=108 case (cluster swap correctness; doesn't break the optimizer).
  - K=2 reaches fit=0 on a feasible SQS (FCC 50:50 n=32).
  - K must be >= 1 (defaults clamp to 1 silently).

Known limitation (documented in commit message): on the L2_1 cold-start
case where single-swap MC plateaus at ~148, K=2 and K=4 also plateau
near the same value. The L2_1 basin is unreachable by *random* K-atom
moves; principled escape would require pattern-aware moves.
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
            cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, timeout=120,
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


def test_cluster_k2_reaches_zero_on_feasible(tmp_path):
    """K=2 should still reach fit=0 on a feasible SQS (FCC 50:50 n=32)."""
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
        CLUSTER_SWAP_K 2
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    assert "CLUSTER_SWAP_K=2" in log
    fit = final_fitness(log)
    assert fit == 0.0, f"K=2 failed on feasible SQS; got {fit}"


def test_cluster_k2_matches_k1_on_infeasible(tmp_path):
    """K=2 should reach the same deterministic √2.5 plateau as K=1 on
    the infeasible 25:50:25 n=108 case (the integer-constraint floor
    is a property of the cell, not the optimizer or move set)."""
    base = textwrap.dedent("""\
        TYP     3
        GITER   100
        ELEMENT 27 54 27
        Cell_DIM        3 3 3
        NESTS   8
        MCSTEP  200
        TOTALIter 10
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 7
        EARLY_STOP_PATIENCE 5
        """)
    rc, log_k1 = run_scraps(tmp_path / "k1", base, ranks=4)
    assert rc == 0
    rc, log_k2 = run_scraps(tmp_path / "k2", base + "CLUSTER_SWAP_K 2\n", ranks=4)
    assert rc == 0
    f1, f2 = final_fitness(log_k1), final_fitness(log_k2)
    assert f1 == pytest.approx(1.58114, rel=1e-3), f"K=1 wrong plateau: {f1}"
    assert f2 == pytest.approx(1.58114, rel=1e-3), f"K=2 wrong plateau: {f2}"


def test_cluster_k_default_is_one(tmp_path):
    """No CLUSTER_SWAP_K line means K=1 -- the printed startup banner
    only fires when K > 1. Verifies default = 1."""
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
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0
    assert "CLUSTER_SWAP_K" not in log, (
        "default K=1 should NOT print the cluster-swap banner")


def test_cluster_k_zero_or_negative_clamps_to_one(tmp_path):
    """CLUSTER_SWAP_K = 0 or negative should be clamped to 1 (no error)."""
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
        CLUSTER_SWAP_K 0
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0
    # No banner because clamped to 1
    assert "CLUSTER_SWAP_K=" not in log
