"""Tests for the same-species SRO auto-expansion via Warren-Cowley sum rule.

SCRAPs's fitness sums over (i, j) with i != j only (the independent
degrees of freedom; alpha_ii is determined by alpha_ij for j != i via
the sum rule sum_j c_j alpha_ij = 0). Through v0.4.11, a `SRO i i shell
alpha` line in INPUT had no effect because the optimizer never touched
the diagonal. v0.4.12 auto-expands such lines via the symmetric split:

    alpha_ij = -c_i * alpha_ii / (1 - c_i)  for every j != i

so the user can write the natural literature form (e.g.
`SRO 2 2 0 -0.10` for alpha(Cr-Cr) at 1NN = -0.10) and SCRAPs
internally drives the right cross-species targets.
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
    sib = Path(mpicxx).parent / "mpirun"
    if sib.exists():
        return str(sib)
    return shutil.which("mpirun")


MPIRUN = find_mpirun()
pytestmark = pytest.mark.skipif(
    not SCRAPS_BIN.exists() or MPIRUN is None,
    reason="scraps binary not built or mpirun missing",
)


FCC_POSCAR = textwrap.dedent("""\
    FCC primitive
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
# Expansion mechanics: banner is printed; correct beta value is used
# ---------------------------------------------------------------------------

def test_same_species_sro_expands_with_banner(tmp_path):
    """`SRO 2 2 0 -0.10` on equiatomic NiCoCr (c_i = 1/3) should
    expand to two cross-species lines at +0.05 each via
    beta = -c_i * alpha_ii / (1 - c_i) = -(1/3)*(-0.1)/(2/3) = +0.05."""
    inp = textwrap.dedent("""\
        TYP     3
        GITER   100
        ELEMENT 36 36 36
        Cell_DIM 3 3 3
        NESTS 4
        MCSTEP 50
        TOTALIter 2
        MAXSHELLNUM 2
        WEIGHT 1 1
        SRO 2 2 0 -0.10
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    # Expansion banner
    assert "[SRO] expanding same-species SRO 2 2 0 -0.1" in log
    assert "via Warren-Cowley sum rule" in log
    # beta = 0.05
    assert "alpha(i,j) = 0.05" in log
    # The two expanded lines should appear
    assert "SRO 2 0 0 0.05" in log
    assert "SRO 2 1 0 0.05" in log


def test_same_species_sro_non_equiatomic_correct_beta(tmp_path):
    """For asymmetric composition (Ni:54, Co:27, Cr:27 = c=0.5/0.25/0.25):
    SRO 0 0 0 -0.10 -> beta = -0.5*(-0.10)/(1-0.5) = +0.10.
    SRO 2 2 0 -0.10 -> beta = -0.25*(-0.10)/(1-0.25) = +0.0333..."""
    inp = textwrap.dedent("""\
        TYP     3
        GITER   100
        ELEMENT 54 27 27
        Cell_DIM 3 3 3
        NESTS 4
        MCSTEP 50
        TOTALIter 2
        MAXSHELLNUM 2
        WEIGHT 1 1
        SRO 0 0 0 -0.10
        SRO 2 2 0 -0.10
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    # Ni (c=0.5) expansion: beta = +0.10
    assert "expanding same-species SRO 0 0 0 -0.1" in log
    assert "alpha(i,j) = 0.1" in log
    # Cr (c=0.25) expansion: beta = +0.0333...
    assert "expanding same-species SRO 2 2 0 -0.1" in log
    assert ("alpha(i,j) = 0.0333333" in log
            or "alpha(i,j) = 0.033333" in log)


# ---------------------------------------------------------------------------
# Edge cases: bad fractions get a warning, don't crash
# ---------------------------------------------------------------------------

def test_same_species_sro_pure_species_warns(tmp_path):
    """If a species has fraction == 1 (pure), the sum-rule denominator
    is zero; we can't expand. Print a warning and skip the line
    (don't crash, don't apply it)."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 108 0
        Cell_DIM 3 3 3
        NESTS 2
        MCSTEP 50
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SRO 0 0 0 -0.10
        """)
    # Note: ELEMENT 108 0 means species 1 has zero count, species 0
    # is pure. The same-species line for species 0 should warn (c=1).
    rc, log = run_scraps(tmp_path, inp)
    # We don't assert on returncode -- the optimization may be trivial
    # or hit a different abort path. Just check the warning fired.
    assert "Warning: SRO same-species line" in log
    assert "species 0 has fraction 1" in log
    assert "must be in (0, 1)" in log


# ---------------------------------------------------------------------------
# Convergence: same-species SRO actually drives the target value
# ---------------------------------------------------------------------------

def test_same_species_sro_drives_target(tmp_path):
    """End-to-end: `SRO 2 2 0 -0.10` should make the achieved
    alpha(Cr-Cr) at 1NN reach -0.10 (within integer-feasibility
    tolerance)."""
    pytest.importorskip("pymatgen")
    inp = textwrap.dedent("""\
        TYP     3
        GITER   100
        ELEMENT 36 36 36
        Cell_DIM 3 3 3
        NESTS 8
        MCSTEP 500
        TOTALIter 15
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 42
        EARLY_STOP_PATIENCE 5
        SRO 2 2 0 -0.10
        """)
    rc, log = run_scraps(tmp_path, inp, ranks=4)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    # Verify the achieved alpha matrix
    verify = REPO / "tools" / "verify_sro.py"
    import sys
    proc = subprocess.run(
        [sys.executable, str(verify),
         str(tmp_path / "SCRAPS.vasp"),
         "--species", "Ni,Co,Cr",
         "--shells", "1", "--json"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode in (0, 1), proc.stderr
    import json
    data = json.loads(proc.stdout)
    sh0 = next(s for s in data["shells"] if s["index"] == 0)
    alpha_CrCr = sh0["alpha_matrix"][2][2]
    alpha_CrNi = sh0["alpha_matrix"][2][0]
    alpha_CrCo = sh0["alpha_matrix"][2][1]
    # Cr-Cr should hit -0.10 within 0.02 (integer-feasibility tolerance)
    assert abs(alpha_CrCr - (-0.10)) < 0.02, (
        f"alpha(Cr-Cr) at 1NN = {alpha_CrCr:+.4f}, target -0.10 "
        f"(deviation {abs(alpha_CrCr + 0.10):.4f})"
    )
    # And the expanded cross-species pairs should hit ~+0.05
    assert abs(alpha_CrNi - 0.05) < 0.02, (
        f"alpha(Cr-Ni) at 1NN = {alpha_CrNi:+.4f}, target +0.05"
    )
    assert abs(alpha_CrCo - 0.05) < 0.02, (
        f"alpha(Cr-Co) at 1NN = {alpha_CrCo:+.4f}, target +0.05"
    )
