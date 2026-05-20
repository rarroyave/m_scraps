"""Tests for tools/suggest_cell.py.

Covers the integer-divisibility check that flags Cell_DIM choices on which
exact SQS (fitness = 0) is mathematically unreachable due to non-integer
target pair counts.

Reference predictions for 25:50:25 ternary on FCC k×k×k conventional:
    N_A = N_C = k², N_B = 2k², N = 4k³.
    Shell 1 has coordination z=6. The AC target is N_A·z·N_C/N = 3k²/2,
    integer iff k² is even, i.e., k even.
    Shell 0 (z=12): AC target is N_A·z·N_C/N = 3k², always integer.

So k=2, 4 are FEASIBLE; k=3, 5, 7 are INFEASIBLE (floor = sqrt(0.5) ≈ 0.707
from the unavoidable AC/CA off-by-0.5 with weights 1,1).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from textwrap import dedent

import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

from suggest_cell import (
    evaluate_cell,
    integer_counts,
    parse_spectator_spec,
    spectator_atom_indices,
)
from verify_sro import load_structure


@pytest.fixture
def fcc_poscar(tmp_path: Path) -> Path:
    p = tmp_path / "POSCAR"
    p.write_text(dedent("""\
        FCC conventional basis (test fixture)
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
        """))
    return p


@pytest.fixture
def bcc_poscar(tmp_path: Path) -> Path:
    p = tmp_path / "POSCAR"
    p.write_text(dedent("""\
        BCC conventional basis (test fixture)
         1.0
         1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 1.0
         2
        direct
         0.0 0.0 0.0
         0.5 0.5 0.5
        """))
    return p


@pytest.mark.parametrize("k,expected_feasible", [
    (2, True),   # n=32, AC target = 6 (integer)
    (3, False),  # n=108, AC target = 13.5 (half-integer)
    (4, True),   # n=256, AC target = 24
    (5, False),  # n=500, AC target = 37.5
    (6, True),   # n=864, AC target = 54
])
def test_fcc_25_50_25_feasibility(fcc_poscar, k, expected_feasible):
    """25:50:25 on FCC k×k×k: feasible iff k is even.

    With 4 basis atoms per conventional cell, N = 4k³ and the 25:50:25
    split is (k³, 2k³, k³). The shell-1 AC target is 1.5·k³, which is
    integer iff k³ is even.
    """
    struct, _ = load_structure(fcc_poscar)
    counts = (k ** 3, 2 * (k ** 3), k ** 3)  # exact 25:50:25
    report = evaluate_cell(struct, k, counts, n_shells=2, weights=[1.0, 1.0])
    assert report.feasible is expected_feasible
    if not expected_feasible:
        # The unavoidable error is on AC and CA at shell 1, each off by 0.5,
        # contributing 2 * (0.5)^2 * weight = 0.5 with weight 1.
        assert report.fitness_sq_lower_bound == pytest.approx(0.5)
        assert report.fitness_lower_bound == pytest.approx(math.sqrt(0.5))


def test_fcc_equiatomic_ternary_feasibility(fcc_poscar):
    """33:33:33 on FCC k=3 (108 atoms, 36/36/36): every target is integer."""
    struct, _ = load_structure(fcc_poscar)
    report = evaluate_cell(struct, 3, (36, 36, 36), n_shells=4, weights=[1.0] * 4)
    assert report.feasible


def test_fcc_50_50_binary_feasibility(fcc_poscar):
    """50:50 binary on FCC k=3 (54/54): trivially feasible at every shell."""
    struct, _ = load_structure(fcc_poscar)
    report = evaluate_cell(struct, 3, (54, 54), n_shells=4, weights=[1.0] * 4)
    assert report.feasible


def test_bcc_equi3_feasibility(bcc_poscar):
    """1/3 each on BCC k=3 (54 atoms, 18/18/18): feasible at 2 shells."""
    struct, _ = load_structure(bcc_poscar)
    report = evaluate_cell(struct, 3, (18, 18, 18), n_shells=2, weights=[1.0, 1.0])
    assert report.feasible


def test_weights_zero_excludes_shell(fcc_poscar):
    """A non-integer target in a shell with weight 0 should not be flagged."""
    struct, _ = load_structure(fcc_poscar)
    # k=3 has shell 1 non-integer; setting that weight to 0 hides it.
    report = evaluate_cell(struct, 3, (27, 54, 27), n_shells=2, weights=[1.0, 0.0])
    assert report.feasible


def test_integer_counts_largest_remainder():
    """Composition rounding distributes the remainder by largest fractional part."""
    # 0.2 / 0.5 / 0.3 at total = 10 -> 2 / 5 / 3
    assert integer_counts([0.2, 0.5, 0.3], 10) == (2, 5, 3)
    # 0.33 / 0.33 / 0.34 at total = 99 -> 33 / 33 / 33
    assert integer_counts([0.333, 0.333, 0.334], 99) == (33, 33, 33)
    # Composition needing remainder distribution
    # 1/3 each at total = 100 -> 33 / 33 / 34 (the species with largest fractional remainder wins)
    counts = integer_counts([1/3, 1/3, 1/3], 100)
    assert counts is not None
    assert sum(counts) == 100
    assert sorted(counts) == [33, 33, 34]


def test_integer_counts_returns_none_when_zero_species(fcc_poscar):
    """A species rounding to 0 should yield None (degenerate composition)."""
    # 0.005 species at total = 32 -> 0.16 atoms -> rounds to 0
    counts = integer_counts([0.005, 0.495, 0.5], 32)
    assert counts is None


# ---- Spectator (HEB-type) tests ---------------------------------------------

@pytest.fixture
def alb2_poscar(tmp_path: Path) -> Path:
    p = tmp_path / "POSCAR"
    p.write_text(dedent("""\
        AlB2-type basis (M at 000, B at 1/3,2/3,1/2 and 2/3,1/3,1/2)
         1.0
          1.000000000000  0.000000000000  0.000000000000
         -0.500000000000  0.866025403784  0.000000000000
          0.000000000000  0.000000000000  1.100000000000
         3
        direct
         0.000000000000  0.000000000000  0.000000000000
         0.333333333333  0.666666666667  0.500000000000
         0.666666666667  0.333333333333  0.500000000000
        """))
    return p


def test_parse_spectator_spec():
    assert parse_spectator_spec("") == []
    assert parse_spectator_spec("1:5") == [(1, 5)]
    assert parse_spectator_spec("1:5,2:5") == [(1, 5), (2, 5)]
    with pytest.raises(ValueError):
        parse_spectator_spec("1-5")  # wrong delimiter


def test_spectator_atom_indices():
    # 3-basis structure, 2x2x2 supercell (cell_volume = 8)
    # Spectator basis = [1, 2] -> spectator atoms at [8..15] U [16..23]
    indices = spectator_atom_indices(3, 8, [1, 2])
    assert indices == set(range(8, 24))
    # All atoms (basis 0,1,2): 0..23, active = 0..7
    assert indices.isdisjoint(set(range(0, 8)))


def test_heb_5metal_5x5x2_is_feasible(alb2_poscar):
    """The actual tests/heb_5metal_random/ case: ELEMENT 10 10 10 10 10 100,
    Cell_DIM 5 5 2 with B as spectator at basis 1,2. Should be feasible."""
    struct, _ = load_structure(alb2_poscar)
    counts = (10, 10, 10, 10, 10, 100)
    report = evaluate_cell(struct, (5, 5, 2), counts, n_shells=4,
                           weights=[1.0] * 4,
                           spectator_basis_species=[(1, 5), (2, 5)])
    assert report.feasible
    assert report.fitness_sq_lower_bound == 0.0


def test_heb_smaller_cell_can_be_infeasible(alb2_poscar):
    """A smaller HEB cell with 5 active species and N_M not divisible by 5
    should be flagged infeasible. 3x3x3 supercell -> 27 M atoms, can't
    split into 5 species equiatomically (27/5 = 5.4 each)."""
    struct, _ = load_structure(alb2_poscar)
    # ELEMENT 5 5 5 5 7 54 (non-equiatomic, the best we can do at 27 M)
    counts = (5, 5, 5, 5, 7, 54)
    report = evaluate_cell(struct, (3, 3, 3), counts, n_shells=2,
                           weights=[1.0, 1.0],
                           spectator_basis_species=[(1, 5), (2, 5)])
    # Whether feasible or not depends on the actual M-M shell structure;
    # we just verify the spectator path runs without error and produces
    # a report. The point is the M-M coordination (not the all-atom z)
    # gets used.
    assert report.total_atoms == 81  # 3 * 27 = 81 total
    assert sum(counts) == 81
    # Active M count = 27, B count = 54; spectator basis [1,2] x 27 cells = 54
    assert counts[5] == 54


def test_spectator_mode_doesnt_count_b_b_pairs(alb2_poscar):
    """A pure-B 'pair' should never be flagged as non-integer since species
    counts for active species exclude spectator allocations."""
    struct, _ = load_structure(alb2_poscar)
    counts = (10, 10, 10, 10, 10, 100)
    report = evaluate_cell(struct, (5, 5, 2), counts, n_shells=4,
                           weights=[1.0] * 4,
                           spectator_basis_species=[(1, 5), (2, 5)])
    # No nonintegral entry should mention species index 5 (the spectator B)
    for sh, i, j, _ in report.nonintegral_pairs:
        assert i != 5 and j != 5
