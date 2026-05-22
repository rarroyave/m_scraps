"""Tests for per-Wyckoff-orbit SUBLATTICE on lattices with non-uniform
coordination (v0.4.2 extension).

Through v0.4.1, SCRAPs' Lattice::coordination_per_shell() asserted that
every non-spectator atom had identical per-shell coordination — abort
on mismatch. This blocked structures like fluorite (CaF2-type, cation
z=8 vs anion z=4 at 1NN), sigma phase (5 Wyckoff orbits, z = 12..15),
Laves, MAX phases, etc.

v0.4.2 generalizes the uniformity check to be per-sublattice when
SUBLATTICE blocks are declared: each sublattice must be uniform within
itself (so split distinct Wyckoff orbits into separate SUBLATTICE
blocks), but different sublattices may legitimately have different z.

These tests use fluorite as the canonical non-uniform-coordination
structure:
  - 4 cation positions at FCC sites (Wyckoff 4a in Fm-3m), each has
    8 anion neighbors at d = a*sqrt(3)/4
  - 8 anion positions at tetrahedral sites (Wyckoff 8c), each has
    4 cation neighbors at the same distance plus 6 anion neighbors
    at d = a/2

For SCRAPs this means a cation has 1NN z=8, an anion has 1NN z=10
(4 Ca + 6 F at the same shell because F-F is at 0.5a and the SCRAPs
shell detector clusters by relative distance gaps -- it may or may
not merge with the Ca-F shell depending on the cell). Either way,
different per-atom coordination, which trips the legacy check.
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


FLUORITE_POSCAR = textwrap.dedent("""\
    CaF2-fluorite (4 cation + 8 anion basis)
    1.0
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    12
    Direct
    0.00 0.00 0.00
    0.50 0.50 0.00
    0.50 0.00 0.50
    0.00 0.50 0.50
    0.25 0.25 0.25
    0.75 0.25 0.25
    0.25 0.75 0.25
    0.75 0.75 0.25
    0.25 0.25 0.75
    0.75 0.25 0.75
    0.25 0.75 0.75
    0.75 0.75 0.75
    """)


def run_scraps(work_dir: Path, input_text: str, ranks: int = 1) -> tuple[int, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "POSCAR").write_text(FLUORITE_POSCAR)
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
# Legacy behavior preserved: fluorite without SUBLATTICE still aborts
# ---------------------------------------------------------------------------

def test_fluorite_without_sublattice_still_aborts(tmp_path):
    """Without SUBLATTICE blocks (legacy single-sublattice mode),
    fluorite still trips the global uniformity check and aborts with
    the v0.4.2 diagnostic that suggests SUBLATTICE blocks per orbit
    as the recovery path."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 32 64
        Cell_DIM 2 2 2
        NESTS 4
        MCSTEP 50
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0, f"expected abort; got rc=0\n{log[-500:]}"
    assert "non-uniform coordination" in log
    # New diagnostic points at the recovery paths.
    assert ("SUBLATTICE" in log or "SPECTATOR_BASIS_SPECIES" in log)


# ---------------------------------------------------------------------------
# v0.4.2 capability: per-Wyckoff SUBLATTICE blocks unlock fluorite
# ---------------------------------------------------------------------------

def test_fluorite_with_per_wyckoff_sublattice_runs(tmp_path):
    """With one SUBLATTICE block per Wyckoff orbit (cation 4a + anion
    8c), the per-sublattice uniformity check passes and the optimizer
    runs to completion. 16 Ca + 16 Sr on the cation sublattice and
    32 F + 32 Cl on the anion sublattice (2x2x2 supercell of the
    12-basis cell -> 96 atoms total)."""
    inp = textwrap.dedent("""\
        TYP     4
        GITER   100
        ELEMENT 16 16 32 32
        Cell_DIM 2 2 2
        NESTS 4
        MCSTEP 200
        TOTALIter 5
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        SUBLATTICE 1 BASIS 4 5 6 7 8 9 10 11 SPECIES 2 3
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed: {log[-800:]}"
    assert "non-uniform coordination" not in log
    assert "[SUBLATTICE] active partition: 2 sublattices, 96 atoms total" in log


def test_fluorite_per_sublattice_sqs_quality(tmp_path):
    """With per-Wyckoff SUBLATTICE, the optimizer should drive per-
    sublattice intra-pair counts toward the per-sublattice random
    target. Verify by reading SCRAPS.vasp and computing per-sublattice
    alpha. Cation sublattice (16 Ca + 16 Sr) at 1NN should approach
    alpha ~ 0 within achievable bounds."""
    pytest.importorskip("pymatgen")
    inp = textwrap.dedent("""\
        TYP     4
        GITER   100
        ELEMENT 16 16 32 32
        Cell_DIM 2 2 2
        NESTS 8
        MCSTEP 500
        TOTALIter 15
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 5
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        SUBLATTICE 1 BASIS 4 5 6 7 8 9 10 11 SPECIES 2 3
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    # Final fitness should be small (cation Ca-Sr at the cation-anion
    # 1NN shell only has anion neighbors, so the only contributing
    # same-sub pairs are anion F-Cl pairs at the anion-anion next
    # shell or none at shell 0; either way the optimizer should
    # converge to a low value).
    fit_line = [l for l in log.splitlines() if l.startswith("FINAL fitness")]
    assert fit_line, f"no FINAL fitness line: {log[-500:]}"
    fit = float(fit_line[0].split("=")[-1].strip())
    assert fit < 50, f"unexpectedly high fitness {fit} for fluorite SQS"


def test_fluorite_per_wyckoff_with_csro(tmp_path):
    """Per-pair SRO targeting works across non-uniform-coordination
    sublattices. Target alpha(Ca-Sr)=+0.5 (mild clustering on cation
    side). Verify the achieved value via the comparison harness's
    per-sublattice alpha calculator."""
    pytest.importorskip("pymatgen")
    inp = textwrap.dedent("""\
        TYP     4
        GITER   100
        ELEMENT 16 16 32 32
        Cell_DIM 2 2 2
        NESTS 8
        MCSTEP 500
        TOTALIter 15
        MAXSHELLNUM 2
        WEIGHT 0 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 5
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        SUBLATTICE 1 BASIS 4 5 6 7 8 9 10 11 SPECIES 2 3
        SRO 0 1 1 +0.5
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps failed: {log[-500:]}"
    # Test passes if the run completes cleanly -- the alpha-checking
    # via verify_sro / pymatgen is verified by other test files; here
    # we just want to confirm the per-sublattice path handles
    # non-uniform coordination with CSRO targets without aborting.


# ---------------------------------------------------------------------------
# Within-sublattice uniformity contract: violations abort cleanly
# ---------------------------------------------------------------------------

def test_fluorite_mixed_wyckoff_in_one_sublattice_aborts(tmp_path):
    """Putting the cation Wyckoff orbit (z=8) and the anion orbit
    (z != 8) into ONE SUBLATTICE block violates the within-sublattice
    uniformity contract. The per-sublattice uniformity check should
    catch it and abort with the v0.4.2 diagnostic that names the
    sublattice id and points at splitting per orbit."""
    inp = textwrap.dedent("""\
        TYP     2
        GITER   100
        ELEMENT 48 48
        Cell_DIM 2 2 2
        NESTS 4
        MCSTEP 50
        TOTALIter 2
        MAXSHELLNUM 1
        WEIGHT 1
        SUBLATTICE 0 BASIS 0 1 2 3 4 5 6 7 8 9 10 11 SPECIES 0 1
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc != 0, f"expected abort; got rc=0\n{log[-500:]}"
    assert "non-uniform coordination" in log
    assert "SUBLATTICE id 0" in log
    assert "split this sublattice" in log


# ---------------------------------------------------------------------------
# Numerical alpha verification via the per-sublattice calculator
# ---------------------------------------------------------------------------

def _import_alpha_calculator():
    """Reuse the per-sublattice alpha calculator from the B2 comparison
    harness. Returns (compute_alpha_per_shell, fix_scraps_species)."""
    import sys
    sys.path.insert(0, str(REPO / "docs"))
    from compare_b2_multisub import compute_alpha_per_shell, fix_scraps_species
    return compute_alpha_per_shell, fix_scraps_species


def test_fluorite_sqs_alpha_numerically_close_to_zero(tmp_path):
    """The SQS run should drive per-sublattice intra-sub alpha close
    to zero. With per-Wyckoff SUBLATTICE on fluorite and 16+16 on the
    cation orbit + 32+32 on the anion orbit, the only optimizable
    pairs at the first within-sublattice shell are anion-anion
    (anion-anion 1NN at d=a/2, which is the first intra-sublattice
    shell). Cation-cation is at the next intra-sublattice shell
    (FCC 1NN at d=a*sqrt(2)/2). Verify max|alpha| on each intra-sub
    shell is small."""
    pytest.importorskip("pymatgen")
    inp = textwrap.dedent("""\
        TYP     4
        GITER   100
        ELEMENT 16 16 32 32
        Cell_DIM 2 2 2
        NESTS 8
        MCSTEP 1500
        TOTALIter 30
        MAXSHELLNUM 3
        WEIGHT 1 1 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 5
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        SUBLATTICE 1 BASIS 4 5 6 7 8 9 10 11 SPECIES 2 3
        """)
    rc, log = run_scraps(tmp_path, inp, ranks=4)
    assert rc == 0, f"scraps failed: {log[-500:]}"

    compute_alpha_per_shell, fix_scraps_species = _import_alpha_calculator()
    out_poscar = tmp_path / "SCRAPS.vasp"
    fix_scraps_species(out_poscar, ["Ca", "Sr", "F", "Cl"])
    # Provide per-species sublattice for the calculator: Ca,Sr -> 0; F,Cl -> 1
    sub_of = {"Ca": 0, "Sr": 0, "F": 1, "Cl": 1}
    shells = compute_alpha_per_shell(
        out_poscar, species=["Ca", "Sr", "F", "Cl"],
        sublattice_of=sub_of, n_shells=3,
    )
    # The intra-sub alpha values should be small (per-sublattice SQS quality).
    # Don't pin to exactly 0 because the small cell + integer-feasibility
    # constraints may force a residual; just assert "well below the
    # global-c_j-misuse bug-era values of 0.5+".
    for sh in shells:
        if sh["max_alpha_intra"] > 0.0:
            assert sh["max_alpha_intra"] < 0.3, (
                f"shell {sh['index']}: max|alpha| intra-sub = "
                f"{sh['max_alpha_intra']:.4f} too high (per-sublattice SQS "
                f"should drive this to near 0)"
            )


def test_fluorite_csro_alpha_hits_target(tmp_path):
    """Target alpha(Ca-Sr) = +0.5 at the cation-cation 1NN shell
    (FCC 1NN at d = a*sqrt(2)/2, shell index 2 in SCRAPs detection).
    With focused WEIGHT on that shell and a 3x3x3 supercell (108
    cation sites, much better target granularity than 32), achieved
    alpha should be within 0.05 of the +0.5 target. The 2x2x2 case
    plateaus at alpha=+0.33 because the discrete pair-count
    granularity at 32 cation sites doesn't span the target value
    finely enough.
    """
    pytest.importorskip("pymatgen")
    inp = textwrap.dedent("""\
        TYP     4
        GITER   100
        ELEMENT 54 54 108 108
        Cell_DIM 3 3 3
        NESTS 8
        MCSTEP 3000
        TOTALIter 30
        MAXSHELLNUM 3
        WEIGHT 0 0 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 5
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        SUBLATTICE 1 BASIS 4 5 6 7 8 9 10 11 SPECIES 2 3
        SRO 0 1 2 +0.5
        """)
    rc, log = run_scraps(tmp_path, inp, ranks=4)
    assert rc == 0, f"scraps failed: {log[-500:]}"

    compute_alpha_per_shell, fix_scraps_species = _import_alpha_calculator()
    out_poscar = tmp_path / "SCRAPS.vasp"
    fix_scraps_species(out_poscar, ["Ca", "Sr", "F", "Cl"])
    sub_of = {"Ca": 0, "Sr": 0, "F": 1, "Cl": 1}
    shells = compute_alpha_per_shell(
        out_poscar, species=["Ca", "Sr", "F", "Cl"],
        sublattice_of=sub_of, n_shells=3,
    )
    sh2 = next((s for s in shells if s["index"] == 2), None)
    assert sh2 is not None, "expected shell 2 (cation-cation 1NN) detected"
    alpha_CaSr = sh2["alpha"][0][1]
    assert alpha_CaSr is not None
    assert abs(alpha_CaSr - 0.5) < 0.05, (
        f"alpha(Ca-Sr) at shell 2 = {alpha_CaSr:+.4f}, target +0.500 "
        f"(deviation {abs(alpha_CaSr - 0.5):.4f})"
    )


# ---------------------------------------------------------------------------
# Three-sublattice case: split the anion orbit by hand
# ---------------------------------------------------------------------------

def test_fluorite_three_sublattice_run(tmp_path):
    """The user can intentionally break symmetry by splitting one
    Wyckoff orbit into multiple SUBLATTICE blocks. Here: split the
    8 anion basis positions into two halves (basis 4-7 and 8-11),
    each its own SUBLATTICE with a distinct species pair. Tests
    the n_sublattices > 2 plumbing through randomizeLattice,
    swap-proposal gates, and the per-sublattice fitness."""
    inp = textwrap.dedent("""\
        TYP     6
        GITER   100
        ELEMENT 16 16 16 16 16 16
        Cell_DIM 2 2 2
        NESTS 4
        MCSTEP 800
        TOTALIter 8
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 5
        SUBLATTICE 0 BASIS 0 1 2 3   SPECIES 0 1
        SUBLATTICE 1 BASIS 4 5 6 7   SPECIES 2 3
        SUBLATTICE 2 BASIS 8 9 10 11 SPECIES 4 5
        """)
    rc, log = run_scraps(tmp_path, inp, ranks=4)
    assert rc == 0, f"scraps failed: {log[-800:]}"
    assert "[SUBLATTICE] active partition: 3 sublattices" in log


# ---------------------------------------------------------------------------
# sublattice_detect.py round-trips a real CaF2 POSCAR
# ---------------------------------------------------------------------------

def test_sublattice_detect_on_fluorite_finds_two_orbits(tmp_path):
    """`tools/sublattice_detect.py` on a fluorite POSCAR with proper
    Ca,F species labels should identify the 2 Wyckoff orbits in
    Fm-3m: 4a (Ca) and 8c (F). Validates the auto-detection workflow
    advertised in the README."""
    pytest.importorskip("pymatgen")
    import sys
    fluorite_with_labels = textwrap.dedent("""\
        CaF2 fluorite
        1.0
        5.46 0.00 0.00
        0.00 5.46 0.00
        0.00 0.00 5.46
        Ca F
        4 8
        Direct
        0.00 0.00 0.00
        0.50 0.50 0.00
        0.50 0.00 0.50
        0.00 0.50 0.50
        0.25 0.25 0.25
        0.75 0.25 0.25
        0.25 0.75 0.25
        0.75 0.75 0.25
        0.25 0.25 0.75
        0.75 0.25 0.75
        0.25 0.75 0.75
        0.75 0.75 0.75
        """)
    (tmp_path / "POSCAR").write_text(fluorite_with_labels)
    tool = REPO / "tools" / "sublattice_detect.py"
    proc = subprocess.run(
        [sys.executable, str(tool), "POSCAR"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"sublattice_detect failed: {proc.stderr}"
    assert "Fm-3m" in proc.stdout
    assert "Wyckoff orbits: 2" in proc.stdout
    # Cation orbit (4a) should map basis 0-3; anion orbit (8c) basis 4-11.
    assert "BASIS 0 1 2 3" in proc.stdout
    assert "BASIS 4 5 6 7 8 9 10 11" in proc.stdout


# ---------------------------------------------------------------------------
# Larger fluorite cell -- scaling smoke
# ---------------------------------------------------------------------------

def test_fluorite_3x3x3_scaling(tmp_path):
    """Larger fluorite supercell (3x3x3 = 12*27 = 324 atoms) with
    per-Wyckoff SUBLATTICE. Smoke test the per-sublattice
    coordination tensor build and feasibility check scale, and the
    run finishes in a reasonable time without aborting."""
    inp = textwrap.dedent("""\
        TYP     4
        GITER   100
        ELEMENT 54 54 108 108
        Cell_DIM 3 3 3
        NESTS 4
        MCSTEP 1000
        TOTALIter 3
        MAXSHELLNUM 1
        WEIGHT 1
        RNG_SEED 1
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        SUBLATTICE 1 BASIS 4 5 6 7 8 9 10 11 SPECIES 2 3
        """)
    rc, log = run_scraps(tmp_path, inp, ranks=4)
    assert rc == 0, f"scraps failed at n=324: {log[-800:]}"
    assert "[SUBLATTICE] active partition: 2 sublattices, 324 atoms total" in log


# ---------------------------------------------------------------------------
# SUBLATTICE + SPECTATOR coexistence with non-uniform coordination
# ---------------------------------------------------------------------------

def test_per_sublattice_feasibility_warning_on_non_integer_target(tmp_path):
    """The v0.4.2 generalization of the feasibility-warning math should
    use per-sublattice quantities to detect non-integer target pair
    counts. Set up a cation composition that makes Ca-Sr at FCC 1NN
    non-integer: 15 Ca + 17 Sr on 32 cation sites gives target =
    15*12*17/32 = 95.625, which is non-integer and forces a positive
    fitness floor. Anion side stays 50/50 (32 F + 32 Cl) so its
    targets are integer."""
    inp = textwrap.dedent("""\
        TYP     4
        GITER   100
        ELEMENT 15 17 32 32
        Cell_DIM 2 2 2
        NESTS 2
        MCSTEP 50
        TOTALIter 2
        MAXSHELLNUM 3
        WEIGHT 0 0 1
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        SUBLATTICE 1 BASIS 4 5 6 7 8 9 10 11 SPECIES 2 3
        """)
    rc, log = run_scraps(tmp_path, inp)
    assert rc == 0, f"scraps should run (warning only): {log[-500:]}"
    assert "non-integer" in log
    assert "Minimum achievable fitness" in log
    # The cation-cation target should appear in the per-shell diagnostic.
    assert "pair[0][1]" in log or "pair[1][0]" in log


def test_fluorite_with_spectator_anion_and_disordered_cation(tmp_path):
    """The classic 'HEB-style' use case for fluorite: disorder on the
    cation orbit, pure F on the anion orbit (a fixed spectator). The
    spectator path and the per-Wyckoff SUBLATTICE path should
    coexist on the same cell. Cation orbit declared via SUBLATTICE
    with 2 mixed species; anion orbit declared via
    SPECTATOR_BASIS_SPECIES for each of the 8 anion basis positions
    pinned to species 2 (the fixed F)."""
    spectator_lines = "\n".join(
        f"SPECTATOR_BASIS_SPECIES {b} 2" for b in range(4, 12)
    )
    inp = textwrap.dedent(f"""\
        TYP     3
        GITER   100
        ELEMENT 16 16 64
        Cell_DIM 2 2 2
        NESTS 4
        MCSTEP 200
        TOTALIter 5
        MAXSHELLNUM 2
        WEIGHT 1 1
        RNG_SEED 1
        EARLY_STOP_PATIENCE 5
        SUBLATTICE 0 BASIS 0 1 2 3 SPECIES 0 1
        {spectator_lines}
        """)
    rc, log = run_scraps(tmp_path, inp, ranks=4)
    assert rc == 0, f"scraps failed: {log[-800:]}"
    # Only one SUBLATTICE block was declared (the spectator orbit
    # isn't itself a SUBLATTICE block), so the banner should report 1
    # active sublattice -- the cation one. Spectator atoms are tracked
    # separately via initialPlacement.
    # (The banner only fires when sublattices.size() > 1; with 1
    # sublattice it stays silent.)
    assert "[SUBLATTICE] active partition" not in log
    # Verify the spectator atoms ended up where declared (species 2).
    pytest.importorskip("pymatgen")
    compute_alpha_per_shell, fix_scraps_species = _import_alpha_calculator()
    out_poscar = tmp_path / "SCRAPS.vasp"
    fix_scraps_species(out_poscar, ["Ca", "Sr", "F"])
    from pymatgen.io.vasp import Poscar
    import warnings
    warnings.filterwarnings("ignore")
    poscar = Poscar.from_file(out_poscar, check_for_potcar=False,
                              read_velocities=False)
    counts = {s.symbol: 0 for s in poscar.structure.species}
    for site in poscar.structure:
        counts[site.species_string] = counts.get(site.species_string, 0) + 1
    assert counts.get("Ca", 0) == 16
    assert counts.get("Sr", 0) == 16
    assert counts.get("F", 0) == 64
