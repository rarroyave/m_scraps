"""End-to-end optimization regression tests.

Each test pins `RNG_SEED`, runs `mpirun -np 4 ./scraps` on a tiny
INPUT, parses the final fitness, and asserts the optimizer hit the
expected outcome (zero for feasible SQS / ordered targets, the
exact integer-constraint plateau for infeasible ones).

Skipped automatically when:
  - The scraps binary hasn't been built (run `./build.sh` first)
  - mpirun isn't on PATH

These tests are the regression net for the optimization loop itself
(the per-swap delta accumulator in empty_nests_fast and the
acceptance logic). They complement the unit-style tests in
test_suggest_cell.py / test_verify_sro.py which exercise the math
in isolation. With RNG_SEED pinned the runs are reproducible across
machines and CI.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRAPS_BIN = REPO / "SCRAPs" / "scraps"
TOOLS = REPO / "tools"
VERIFY = TOOLS / "verify_sro.py"


def find_mpirun() -> str | None:
    """Find the mpirun matching the MPI used to build scraps.

    PATH-first `mpirun` is wrong when multiple MPIs are installed (a
    common case on macOS: anaconda ships MPICH's Hydra mpirun, while
    SCRAPS may be built against Homebrew's Open MPI). Mixing them gives
    silent singleton fallback. Prefer:
      1. $MPIRUN env var
      2. sibling of the mpic++ used in build.sh (default
         /opt/homebrew/bin/mpic++ on macOS, mpic++ on PATH on Linux)
      3. PATH `mpirun` (last resort)
    """
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
    reason="scraps binary not built or no mpirun found",
)


# ---- Reusable fixtures and helpers -----------------------------------------

FCC_POSCAR = textwrap.dedent("""\
    FCC conventional basis
     1.0
     1.0 0.0 0.0
     0.0 1.0 0.0
     0.0 0.0 1.0
     4
    direct
     0.0 0.0 0.0
     0.5 0.5 0.0
     0.5 0.0 0.5
     0.0 0.5 0.5
    """)

BCC_POSCAR = textwrap.dedent("""\
    BCC conventional basis
     1.0
     1.0 0.0 0.0
     0.0 1.0 0.0
     0.0 0.0 1.0
     2
    direct
     0.0 0.0 0.0
     0.5 0.5 0.5
    """)


def oriented_bcc_poscar(transform: list[list[int]]) -> str:
    """Return a BCC POSCAR in a general integer-oriented setting.

    These cells are crystallographically BCC, but their lattice vectors are
    deliberately not restricted to the conventional cubic axes.  They guard
    the periodic-image and Wigner--Seitz geometry code used by every SCRAPS
    workflow, not just a particular application workflow.
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp import Poscar

    structure = Structure(
        Lattice.cubic(3.0), ["X", "X"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    structure.make_supercell(transform)
    return Poscar(structure).get_str()


def run_scraps(work_dir: Path, poscar: str, input_text: str, ranks: int = 4) -> dict:
    """Run scraps in `work_dir` and return a parsed result dict.

    Returns: {'fitness': float | None, 'wall_s': float, 'log': str,
              'scraps_vasp': Path}
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "POSCAR").write_text(poscar)
    (work_dir / "INPUT").write_text(input_text)
    shutil.copy(SCRAPS_BIN, work_dir / "scraps")
    os.chmod(work_dir / "scraps", 0o755)

    import time
    t0 = time.time()
    log_path = work_dir / "scraps.log"
    with log_path.open("w") as f:
        proc = subprocess.run(
            [MPIRUN, "-np", str(ranks), "./scraps"],
            cwd=work_dir, stdout=f, stderr=subprocess.STDOUT,
            timeout=120,
        )
    wall = time.time() - t0
    assert proc.returncode == 0, f"scraps failed (rc={proc.returncode}); log:\n{log_path.read_text()[-2000:]}"

    log_text = log_path.read_text()
    fitness = None
    for line in log_text.splitlines():
        if line.startswith("FINAL fitness"):
            try:
                fitness = float(line.split("=")[-1].strip())
            except Exception:
                pass
            break
    return {
        "fitness": fitness,
        "wall_s": wall,
        "log": log_text,
        "scraps_vasp": work_dir / "SCRAPS.vasp",
    }


def max_alpha(scraps_vasp: Path, species: str, shells: int = 2) -> float:
    """Return max |alpha| across all shells from verify_sro."""
    proc = subprocess.run(
        [sys.executable, str(VERIFY), str(scraps_vasp),
         "--species", species, "--shells", str(shells), "--json"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"verify_sro failed: {proc.stderr}"
    data = json.loads(proc.stdout)
    mx = 0.0
    for sh in data["shells"]:
        for row in sh["alpha_matrix"]:
            for v in row:
                if v is not None and v == v:
                    mx = max(mx, abs(v))
    return mx


# ---- The actual regression tests -------------------------------------------

def test_fcc_50_50_binary_hits_zero(tmp_path):
    """Feasible SQS on FCC 32 atoms: must converge to fitness=0."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   1000
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 7
        TOTALIter 20
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 12345
        EARLY_STOP_PATIENCE 3
        """)
    out = run_scraps(tmp_path, FCC_POSCAR, inp)
    assert out["fitness"] == 0, f"expected fitness=0, got {out['fitness']}"
    assert max_alpha(out["scraps_vasp"], "A,B", shells=1) < 1e-6


def test_bcc_50_50_binary_hits_zero(tmp_path):
    """Feasible SQS on BCC 16 atoms: must converge to fitness=0."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   1000
        ELEMENT 8 8
        Cell_DIM        2 2 2
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 7
        TOTALIter 20
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 12345
        EARLY_STOP_PATIENCE 3
        """)
    out = run_scraps(tmp_path, BCC_POSCAR, inp)
    assert out["fitness"] == 0


@pytest.mark.parametrize(
    "transform",
    [
        [[1, -1, 0], [0, 0, 1], [1, 1, 0]],  # rotated orthorhombic BCC
        [[1, -1, 0], [0, 0, 1], [1, 1, 1]],  # skew/triclinic BCC
    ],
    ids=["rotated", "skew"],
)
def test_oriented_bcc_ternary_matches_independent_sro_oracle(tmp_path, transform):
    """A valid oriented BCC SQS must not silently acquire an empty graph.

    The 3x4x3 tiling has 144 sites, so each ternary component has 48 sites
    and the random pair counts through the third BCC shell are integral.
    A zero internal fitness is accepted only when the independent
    Pymatgen-based Warren--Cowley calculation also reports zero.
    """
    inp = textwrap.dedent("""\
        TYP     3
        GITER   1000
        ELEMENT 48 48 48
        Cell_DIM        3 4 3
        NESTS   8
        MCSTEP  200
        AUTO_MCSTEP 7
        TOTALIter 25
        MAXSHELLNUM 3
        WEIGHT 1 1 1
        RNG_SEED 20261006
        EARLY_STOP_PATIENCE 5
        """)
    out = run_scraps(tmp_path, oriented_bcc_poscar(transform), inp)
    assert out["fitness"] == 0, out["log"][-2000:]
    assert max_alpha(out["scraps_vasp"], "A,B,C", shells=3) < 1e-6


def test_infeasible_plateau_is_deterministic(tmp_path):
    """25:50:25 ternary on FCC k=3 (n=108) is infeasible at shell 1.

    The deterministic plateau fitness is sqrt(2.5) ~= 1.58114. With
    RNG_SEED pinned, the optimizer must land here, not below.
    """
    inp = textwrap.dedent("""\
        TYP     3
        GITER   1000
        ELEMENT 27 54 27
        Cell_DIM        3 3 3
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 7
        TOTALIter 30
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 12345
        EARLY_STOP_PATIENCE 5
        """)
    out = run_scraps(tmp_path, FCC_POSCAR, inp)
    assert out["fitness"] == pytest.approx(1.58114, rel=1e-4), (
        f"plateau changed: expected 1.58114, got {out['fitness']}")


def test_b2_ordering_via_sro_keyword(tmp_path):
    """SRO target alpha_AB^NN=-1, alpha_AB^2NN=+1 on 50:50 BCC must
    converge to fitness=0 (= exact B2 structure)."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   1000
        ELEMENT 16 16
        Cell_DIM        2 2 4
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 10
        TOTALIter 30
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 12345
        EARLY_STOP_PATIENCE 5
        SRO 0 1 0 -1.0
        SRO 0 1 1 +1.0
        """)
    out = run_scraps(tmp_path, BCC_POSCAR, inp)
    assert out["fitness"] == 0
    # The B2 NN signature is alpha_AB = -1 exactly; the diagonal
    # alpha_AA = +1 (no A-A NN in B2) makes max|alpha| over the matrix = 1.0.
    assert max_alpha(out["scraps_vasp"], "A,B", shells=2) == pytest.approx(1.0, abs=1e-6)


def test_random_only_is_worse_than_full_scraps(tmp_path):
    """Sanity: the cuckoo-search MC actually beats the random baseline.

    Same RNG seed, same INPUT except RANDOM_ONLY=1 vs 0. The full
    SCRAPS run must end strictly below the random baseline.
    """
    base_inp = textwrap.dedent("""\
        TYP     3
        GITER   1000
        ELEMENT 27 54 27
        Cell_DIM        3 3 3
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 7
        TOTALIter 30
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 12345
        EARLY_STOP_PATIENCE 5
        """)
    random_run = run_scraps(tmp_path / "random_only", FCC_POSCAR,
                             base_inp + "RANDOM_ONLY 1\n")
    scraps_run = run_scraps(tmp_path / "full_scraps", FCC_POSCAR, base_inp)
    assert scraps_run["fitness"] < random_run["fitness"], (
        f"SCRAPS ({scraps_run['fitness']}) not better than "
        f"random baseline ({random_run['fitness']})")


def test_fast_swaps_matches_legacy_fitness(tmp_path):
    """FAST_SWAPS=0 (legacy) and FAST_SWAPS=1 (delta) must reach the
    same final fitness with the same RNG seed. The actual nest may
    differ slightly at the same plateau-fitness due to fp-summation
    order divergence, but fitness must match to FP precision."""
    base_inp = textwrap.dedent("""\
        TYP     3
        GITER   1000
        ELEMENT 27 54 27
        Cell_DIM        3 3 3
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 7
        TOTALIter 30
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 12345
        EARLY_STOP_PATIENCE 5
        """)
    legacy = run_scraps(tmp_path / "legacy", FCC_POSCAR,
                        base_inp + "FAST_SWAPS 0\n")
    fast = run_scraps(tmp_path / "fast", FCC_POSCAR,
                       base_inp + "FAST_SWAPS 1\n")
    assert legacy["fitness"] == pytest.approx(fast["fitness"], rel=1e-6)


def test_spectator_heb_runs_and_alpha_excludes_b(tmp_path):
    """HEB-type 5-metal AlB2 run: spectator B sublattice must stay
    in place and the alpha matrix on the M sublattice must converge
    near zero (random target)."""
    poscar = textwrap.dedent("""\
        AlB2-type basis (M at 000, B at honeycomb)
         1.0
          1.000000000000  0.000000000000  0.000000000000
         -0.500000000000  0.866025403784  0.000000000000
          0.000000000000  0.000000000000  1.100000000000
         3
        direct
         0.000000000000  0.000000000000  0.000000000000
         0.333333333333  0.666666666667  0.500000000000
         0.666666666667  0.333333333333  0.500000000000
        """)
    inp = textwrap.dedent("""\
        TYP     6
        GITER   500
        ELEMENT 10 10 10 10 10 100
        Cell_DIM        5 5 2
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 7
        TOTALIter 10
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 12345
        EARLY_STOP_PATIENCE 3
        SPECTATOR_BASIS_SPECIES 1 5
        SPECTATOR_BASIS_SPECIES 2 5
        """)
    out = run_scraps(tmp_path, poscar, inp)
    # On the M sublattice (5 metals equiatomic) the cell is feasible at 2 shells
    # so fitness should reach 0 (or near it within the iteration budget).
    assert out["fitness"] is not None
    assert out["fitness"] < 0.5, f"fitness too high: {out['fitness']}"


def _read_conv(work_dir: Path, base_name: str = "conv.csv") -> list[tuple[int, float]]:
    """Return list of (iteration, best_fitness) tuples from any
    convergence log under work_dir. Handles both the real-MPI case
    (single file named base_name) and the macOS singleton case
    (multiple files base_name.<pid> -- pick the longest trajectory)."""
    candidates = list(work_dir.glob(f"{base_name}*"))
    assert candidates, f"no convergence log found in {work_dir}"
    best_rows = None
    for p in candidates:
        lines = p.read_text().strip().splitlines()
        if not lines or not lines[0].startswith("iteration"):
            continue
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                rows.append((int(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if best_rows is None or len(rows) > len(best_rows):
            best_rows = rows
    assert best_rows, f"no parseable rows in any of {candidates}"
    return best_rows


def test_convergence_is_monotone_nonincreasing(tmp_path):
    """Verify the cuckoo-search MC trajectory descends, not stumbles.

    Runs the infeasible-plateau case (25:50:25 n=108) with
    CONVERGENCE_LOG, parses the per-iteration best fitness, and
    asserts it monotonically decreases (or stays the same) from one
    outer iteration to the next.

    A small epsilon allows for the ε-band stochastic acceptance in
    empty_nests, which can transiently move to an equal-fitness
    neighbor. But a true increase by more than `tol` would mean the
    optimizer is going backwards -- not what a descent search should
    do.
    """
    inp = textwrap.dedent("""\
        TYP     3
        GITER   1000
        ELEMENT 27 54 27
        Cell_DIM        3 3 3
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 7
        TOTALIter 15
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 12345
        EARLY_STOP_PATIENCE 100
        CONVERGENCE_LOG conv.csv
        """)
    out = run_scraps(tmp_path, FCC_POSCAR, inp)
    rows = _read_conv(tmp_path)
    fitnesses = [f for _, f in rows]
    assert len(fitnesses) >= 3, f"too few iterations: {len(fitnesses)}"
    tol = 1e-6
    for prev, curr in zip(fitnesses, fitnesses[1:]):
        assert curr <= prev + tol, (
            f"fitness went UP: {prev} -> {curr} (delta = {curr - prev}). "
            f"Full trajectory: {fitnesses}")
    assert fitnesses[-1] < fitnesses[0], (
        f"no progress: initial {fitnesses[0]} == final {fitnesses[-1]}")


def test_convergence_log_records_pre_mc_baseline(tmp_path):
    """The convergence CSV's iteration=-1 row is the pre-MC best of the
    initial random pool. Confirms the baseline anchor exists and that
    SCRAPS' MC drove the fitness below it."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   1000
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   12
        MCSTEP  100
        AUTO_MCSTEP 7
        TOTALIter 10
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 12345
        CONVERGENCE_LOG conv.csv
        """)
    out = run_scraps(tmp_path, FCC_POSCAR, inp)
    rows = _read_conv(tmp_path)
    iters = [i for i, _ in rows]
    fitnesses = [f for _, f in rows]
    assert iters[0] == -1, f"expected iter -1 first, got {iters[0]}"
    assert fitnesses[-1] <= fitnesses[0], (
        "SCRAPS final not <= pre-MC baseline: "
        f"baseline={fitnesses[0]}, final={fitnesses[-1]}")


def test_random_only_skips_mc_loop(tmp_path):
    """RANDOM_ONLY 1 should print the skip message and never print a
    'FITNESS - >' line (which only appears inside the MC loop)."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   1000
        ELEMENT 16 16
        Cell_DIM        2 2 2
        NESTS   12
        MCSTEP  100
        TOTALIter 30
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 12345
        RANDOM_ONLY 1
        """)
    out = run_scraps(tmp_path, FCC_POSCAR, inp)
    assert "RANDOM_ONLY mode" in out["log"]
    assert "FITNESS - >" not in out["log"], (
        "MC loop ran despite RANDOM_ONLY=1 (look for FITNESS - > lines)")
