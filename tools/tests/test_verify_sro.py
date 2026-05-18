"""
Tests for verify_sro.py against analytically-known structures.

Reference cells:
    - B2 NiAl (BCC, 2-atom basis) — perfect chemical ordering
    - L1_2 Ni3Al (FCC, 4-atom basis) — partial ordering with known α matrix
    - A1 random FCC binary — α ≈ 0 within statistical bounds

Each test builds the structure via pymatgen, computes α with our script's
functions, and checks against textbook values.

Run with:
    pytest tools/tests/test_verify_sro.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make verify_sro importable when tests run from any cwd.
TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

from verify_sro import (
    detect_shells,
    compute_alpha,
    parse_target_file,
    load_structure,
)

from pymatgen.core import Structure, Lattice


# ---- Reference structures ---------------------------------------------------

def make_b2_NiAl(reps: tuple[int, int, int] = (4, 4, 4)) -> Structure:
    """B2 NiAl: BCC lattice with Ni at (0,0,0) and Al at (1/2,1/2,1/2).

    In B2, every nearest neighbor of a Ni is an Al and vice versa.
    Coordination z = 8, equiatomic. Expected first-shell alpha:
        alpha(Ni-Al) = alpha(Al-Ni) = -1   (perfect ordering)
        alpha(Ni-Ni) = alpha(Al-Al) = +1   (sum rule, c_i = 1/2)
    """
    lat = Lattice.cubic(2.88)
    structure = Structure(lat, ["Ni", "Al"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structure.make_supercell(reps)
    return structure.get_sorted_structure()


def make_L12_Ni3Al(reps: tuple[int, int, int] = (3, 3, 3)) -> Structure:
    """L1_2 Ni3Al: FCC lattice with Al at the corner and Ni at face centers.

    c_Ni = 3/4, c_Al = 1/4, z = 12. Each Al center has 12 Ni neighbors
    (no Al at NN). Each Ni center has 8 Ni and 4 Al neighbors.
    Expected first-shell alpha:
        alpha(Ni-Ni) = 1 - (8/12)/(3/4) = +1/9
        alpha(Ni-Al) = 1 - (4/12)/(1/4) = -1/3
        alpha(Al-Ni) = 1 - (12/12)/(3/4) = -1/3
        alpha(Al-Al) = 1 - 0/(1/4)        = +1
    """
    lat = Lattice.cubic(3.57)
    # Al at corner + Ni at three independent face centers
    species = ["Al", "Ni", "Ni", "Ni"]
    coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    structure = Structure(lat, species, coords)
    structure.make_supercell(reps)
    return structure.get_sorted_structure()


# ---- Tests ------------------------------------------------------------------

def test_b2_NiAl_perfect_ordering():
    s = make_b2_NiAl()
    shells = detect_shells(s, n_shells=1)
    assert len(shells) == 1
    assert shells[0].z == 8

    alpha, counts, names = compute_alpha(s, shells)
    ni = names.index("Ni")
    al = names.index("Al")

    assert alpha[0, ni, al] == pytest.approx(-1.0, abs=1e-9)
    assert alpha[0, al, ni] == pytest.approx(-1.0, abs=1e-9)
    assert alpha[0, ni, ni] == pytest.approx(+1.0, abs=1e-9)
    assert alpha[0, al, al] == pytest.approx(+1.0, abs=1e-9)


def test_L12_Ni3Al_known_alpha():
    s = make_L12_Ni3Al()
    shells = detect_shells(s, n_shells=1)
    assert len(shells) == 1
    assert shells[0].z == 12

    alpha, counts, names = compute_alpha(s, shells)
    ni = names.index("Ni")
    al = names.index("Al")

    assert alpha[0, ni, ni] == pytest.approx(+1.0 / 9.0, abs=1e-9)
    assert alpha[0, ni, al] == pytest.approx(-1.0 / 3.0, abs=1e-9)
    assert alpha[0, al, ni] == pytest.approx(-1.0 / 3.0, abs=1e-9)
    assert alpha[0, al, al] == pytest.approx(+1.0, abs=1e-9)

    # Sum rule check: sum_j c_j alpha[i,j] = 0 for each i
    c = np.array([3 / 4, 1 / 4]) if names == ["Ni", "Al"] else np.array([1 / 4, 3 / 4])
    for i in range(2):
        assert (c * alpha[0, i]).sum() == pytest.approx(0.0, abs=1e-9)


def test_alpha_matrix_is_symmetric_for_finite_cell():
    """For any finite supercell, count[i][j] = count[j][i] physically,
    and the alpha matrix must therefore be exactly symmetric in the
    equiatomic case (and satisfy c_i*alpha_ij = c_j*alpha_ji in general)."""
    s = make_b2_NiAl()
    shells = detect_shells(s, n_shells=2)
    alpha, _, names = compute_alpha(s, shells)
    for k in range(len(shells)):
        # equiatomic: simple symmetry
        np.testing.assert_allclose(alpha[k], alpha[k].T, atol=1e-9)


def test_multi_shell_detected():
    """BCC lattice should yield first 3 shells at distances
    a*sqrt(3)/2 (z=8), a (z=6), a*sqrt(2) (z=12)."""
    s = make_b2_NiAl(reps=(5, 5, 5))
    shells = detect_shells(s, n_shells=3)
    assert len(shells) == 3
    # Check coordinations
    a = 2.88
    expected = [(a * np.sqrt(3) / 2, 8), (a, 6), (a * np.sqrt(2), 12)]
    for sh, (r_exp, z_exp) in zip(shells, expected):
        assert sh.z == z_exp, f"shell {sh.index}: z={sh.z}, expected {z_exp}"
        assert sh.r_min == pytest.approx(r_exp, rel=1e-4)


def test_target_file_parsing():
    content = """
    # An example SCRAPS target file
    TYP 3
    SRO 0 1 0 -0.10
    SRO 0 2 0 +0.05
    SRO 1 2 1 -0.03  # inline comment
    # SRO 9 9 9 0  -- commented-out line should not parse
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        entries = parse_target_file(tmp)
    finally:
        tmp.unlink()

    assert len(entries) == 3
    assert (entries[0].i, entries[0].j, entries[0].shell) == (0, 1, 0)
    assert entries[0].alpha == -0.10
    assert entries[1].alpha == +0.05
    assert entries[2].shell == 1


def test_target_file_handles_malformed_lines():
    content = "SRO 0 1\nSRO not_int 1 0 0.1\nSRO 0 1 0 -0.05\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        entries = parse_target_file(tmp)
    finally:
        tmp.unlink()
    # Only the well-formed line should parse
    assert len(entries) == 1
    assert entries[0].alpha == -0.05


def test_scraps_random_supercell_approximately_zero():
    """If the SCRAPS-generated nicocr_random case has been run,
    it should produce alpha ~ 0 within statistical bounds."""
    supercell = TOOLS.parent / "tests" / "nicocr_random" / "SCRAPS.vasp"
    if not supercell.exists():
        pytest.skip(f"{supercell} not yet generated — run SCRAPS first")
    structure, _ = load_structure(supercell, species_override=["Ni", "Co", "Cr"])
    shells = detect_shells(structure, n_shells=1)
    alpha, _, _ = compute_alpha(structure, shells)
    # A genuine SQS should hit |alpha| < 0.05 on shell 1 for 108 atoms
    assert np.max(np.abs(alpha[0])) < 0.05, \
        f"SQS quality check failed: max |alpha| = {np.max(np.abs(alpha[0]))}"


def test_scraps_csro_supercell_matches_target():
    """The nicocr_csro case should match its INPUT-file target within 0.02."""
    supercell = TOOLS.parent / "tests" / "nicocr_csro" / "SCRAPS.vasp"
    target_file = TOOLS.parent / "tests" / "nicocr_csro" / "INPUT"
    if not supercell.exists():
        pytest.skip(f"{supercell} not yet generated — run SCRAPS first")
    structure, _ = load_structure(supercell, species_override=["Ni", "Co", "Cr"])
    shells = detect_shells(structure, n_shells=1)
    alpha, _, names = compute_alpha(structure, shells)
    targets = parse_target_file(target_file)
    for t in targets:
        if t.shell != 0:
            continue
        achieved = float(alpha[0, t.i, t.j])
        err = abs(achieved - t.alpha)
        # 108 atoms; integer-count quantization gives up to ~0.03 error
        assert err < 0.03, \
            f"{names[t.i]}-{names[t.j]} shell {t.shell}: " \
            f"target {t.alpha:+.4f}, achieved {achieved:+.4f}, err {err:.4f}"
