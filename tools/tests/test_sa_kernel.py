"""Tests for the simulated-annealing MC kernel (MC_KERNEL=sa).

Locks in:
  - Default kernel is cuckoo (backward compat) — covered indirectly by
    every other test.
  - MC_KERNEL=sa reaches fit=0 on a feasible SQS target (proves the
    Metropolis acceptance + temperature schedule actually descend).
  - SA's temperature schedule is logged once per outer iter (proves
    g_sa_current_T is updated each iter).
  - SA reaches the same deterministic plateau as cuckoo on an
    infeasible-SQS case (25:50:25 n=108 plateaus at √2.5 either way).
  - MC_KERNEL=sa with a warm-started already-optimal nest preserves
    fit=0 (the same regression we hit for cuckoo+FAST_SWAPS).
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


def run_scraps(work_dir: Path, input_text: str, ranks: int = 1,
               extra_files: dict[str, str] | None = None) -> tuple[int, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "POSCAR").write_text(FCC_POSCAR)
    (work_dir / "INPUT").write_text(input_text)
    for name, content in (extra_files or {}).items():
        (work_dir / name).write_text(content)
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


# ---------------------------------------------------------------------------

def test_sa_reaches_zero_on_feasible_sqs(tmp_path):
    """MC_KERNEL=sa on a small feasible SQS (FCC 32 atoms 50:50) must
    converge to fitness=0 just like the cuckoo kernel."""
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
        MC_KERNEL sa
        SA_T_INITIAL 1.0
        SA_T_FINAL 1e-3
        EARLY_STOP_PATIENCE 5
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    assert "MC_KERNEL=sa" in log
    fit = final_fitness(log)
    assert fit is not None
    assert fit == 0.0, f"SA failed to reach fit=0 on feasible SQS; got {fit}"


def test_sa_temperature_schedule_is_logged(tmp_path):
    """Each outer iter should print 'SA T = <T>'. Verifies the
    temperature global is being updated. Uses the 25:50:25 n=108
    infeasible plateau so the loop doesn't terminate at iter 0
    (fitness stays at √2.5 > epsi, so we get the full iter count)."""
    inp = textwrap.dedent("""\
        TYP     3
        GITER   100
        ELEMENT 27 54 27
        Cell_DIM        3 3 3
        NESTS   4
        MCSTEP  20
        TOTALIter 5
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 100
        MC_KERNEL sa
        SA_T_INITIAL 1.0
        SA_T_FINAL 0.01
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0
    t_lines = [l for l in log.splitlines() if " SA T = " in l]
    assert len(t_lines) >= 3, f"expected multiple SA T lines, got: {t_lines}"
    import re
    first_t = float(re.search(r"SA T = ([0-9.e+-]+)", t_lines[0]).group(1))
    last_t = float(re.search(r"SA T = ([0-9.e+-]+)", t_lines[-1]).group(1))
    assert first_t == pytest.approx(1.0, rel=1e-3)
    assert last_t < first_t, f"SA T did not cool: first={first_t}, last={last_t}"


def test_sa_plateau_matches_cuckoo_on_infeasible(tmp_path):
    """Infeasible 25:50:25 on FCC k=3 (n=108) plateaus at √2.5 ≈ 1.58114
    with either kernel — the floor is a property of the integer
    constraint, not the optimizer."""
    base = textwrap.dedent("""\
        TYP     3
        GITER   100
        ELEMENT 27 54 27
        Cell_DIM        3 3 3
        NESTS   12
        MCSTEP  500
        TOTALIter 8
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 7
        EARLY_STOP_PATIENCE 5
        """)
    # cuckoo
    rc, log_c = run_scraps(tmp_path / "cuckoo", base)
    assert rc == 0
    fit_c = final_fitness(log_c)
    # sa
    rc, log_s = run_scraps(tmp_path / "sa",
                           base + "MC_KERNEL sa\nSA_T_INITIAL 2.0\nSA_T_FINAL 1e-3\n")
    assert rc == 0
    fit_s = final_fitness(log_s)
    # Both should land on the same √2.5 plateau (within a small tol).
    assert fit_c is not None and fit_s is not None
    assert fit_c == pytest.approx(1.58114, rel=1e-3), (
        f"cuckoo did not reach plateau: {fit_c}")
    assert fit_s == pytest.approx(1.58114, rel=1e-3), (
        f"SA did not reach plateau: {fit_s}")


def test_pt_runs_and_exchanges(tmp_path):
    """MC_KERNEL=pt with 4 MPI ranks: each rank should report its own
    fixed log-spaced T, and exchange attempts should happen in the
    log. Convergence quality is not asserted -- this is a smoke test
    for the PT plumbing.
    """
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   4
        MCSTEP  20
        TOTALIter 6
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 100
        MC_KERNEL pt
        SA_T_INITIAL 10.0
        SA_T_FINAL 0.01
        PT_EXCHANGE_INTERVAL 1
        """)
    rc, log = run_scraps(tmp_path, inp, ranks=4)
    assert rc == 0, f"PT run failed: {log[-500:]}"
    # Each rank should print its own T
    pt_setup = [l for l in log.splitlines() if "MC_KERNEL=pt" in l]
    assert len(pt_setup) == 4, f"expected 4 rank setup lines, got: {pt_setup}"
    # And there should be at least one exchange logged (printed on rank 0 only)
    assert "PT exchange" in log, "no PT exchange events logged"


def test_pt_falls_back_gracefully_on_one_rank(tmp_path):
    """With nproc=1, PT has no partner. The run should still succeed
    (PT exchange step skipped) and produce a valid output."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   1
        MCSTEP  20
        TOTALIter 3
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        MC_KERNEL pt
        SA_T_INITIAL 2.0
        SA_T_FINAL 0.01
        """)
    rc, log = run_scraps(tmp_path, inp, ranks=1)
    assert rc == 0
    # No PT exchange events expected with 1 rank
    assert "PT exchange" not in log
    fit = final_fitness(log)
    assert fit is not None


def test_sa_preserves_warm_start(tmp_path):
    """MC_KERNEL=sa with a perfect B2 warm-start should keep fit=0
    (cool temperature doesn't introduce uphill drift past the floor)."""
    # Perfect B2 on BCC 2x2x2 = 16 atoms: 8 species-0 + 8 species-1
    init_nest = "\n".join(["0"] * 8 + ["1"] * 8) + "\n"
    bcc_poscar = textwrap.dedent("""\
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
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 8 8
        Cell_DIM        2 2 2
        NESTS   2
        MCSTEP  10
        TOTALIter 5
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 1
        SRO 0 1 0 -1.0
        SRO 0 1 1 +1.0
        INIT_NEST init.nest
        MC_KERNEL sa
        SA_T_INITIAL 0.01
        SA_T_FINAL 1e-6
        """)
    work = tmp_path / "warm"
    work.mkdir()
    (work / "POSCAR").write_text(bcc_poscar)
    (work / "INPUT").write_text(inp)
    (work / "init.nest").write_text(init_nest)
    shutil.copy(SCRAPS_BIN, work / "scraps")
    os.chmod(work / "scraps", 0o755)
    log_path = work / "scraps.log"
    with log_path.open("w") as f:
        rc = subprocess.run([MPIRUN, "-np", "1", "./scraps"],
                            cwd=work, stdout=f, stderr=subprocess.STDOUT,
                            timeout=60).returncode
    log = log_path.read_text()
    assert rc == 0, f"scraps failed: {log[-500:]}"
    fit = final_fitness(log)
    assert fit == 0.0, f"SA degraded warm-start B2: got fit={fit}"
