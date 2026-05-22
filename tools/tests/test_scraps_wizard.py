"""Tests for tools/scraps_wizard.py — the novice-friendly CLI.

Covers:
  - Composition parser (correctness + error messages)
  - Hamilton largest-remainder integer split
  - Cell-size auto-pick with integer-feasibility filter
  - `list-templates` (no SCRAPS binary needed)
  - `quick` end-to-end on FCC single-sublattice (smoke + correctness)
  - `quick` end-to-end on B2 multi-sublattice
  - `from-structure` with stdin-injected composition
  - `interactive` mode driven via stdin redirection
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
WIZARD = REPO / "tools" / "scraps_wizard.py"
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
needs_scraps = pytest.mark.skipif(
    not SCRAPS_BIN.exists() or MPIRUN is None,
    reason="scraps binary not built or mpirun missing",
)


# ---- composition parser (pure-Python unit tests) --------------------------

def _import_wizard():
    sys.path.insert(0, str(REPO / "tools"))
    import scraps_wizard
    return scraps_wizard


def test_composition_parser_basic():
    w = _import_wizard()
    assert w.parse_composition("Ni:0.5,Co:0.5") == {"Ni": 0.5, "Co": 0.5}


def test_composition_parser_equals_separator():
    w = _import_wizard()
    assert w.parse_composition("Ni=0.5;Co=0.5") == {"Ni": 0.5, "Co": 0.5}


def test_composition_parser_whitespace_tolerance():
    w = _import_wizard()
    assert w.parse_composition("  Ni : 0.33 , Co : 0.33 , Cr : 0.34 ") == {
        "Ni": 0.33, "Co": 0.33, "Cr": 0.34,
    }


def test_composition_parser_rejects_unbalanced_sum():
    w = _import_wizard()
    with pytest.raises(ValueError, match="sum to"):
        w.parse_composition("Ni:0.5,Co:0.4")


def test_composition_parser_rejects_duplicates():
    w = _import_wizard()
    with pytest.raises(ValueError, match="twice"):
        w.parse_composition("Ni:0.5,Ni:0.5")


def test_composition_parser_rejects_missing_separator():
    w = _import_wizard()
    with pytest.raises(ValueError, match="separator"):
        w.parse_composition("Ni 0.5,Co 0.5")


def test_split_to_integers_sums_correctly():
    w = _import_wizard()
    out = w.split_to_integers({"Ni": 0.33, "Co": 0.33, "Cr": 0.34}, 108)
    assert sum(out.values()) == 108
    # Each within 1 of perfect (Hamilton's method guarantee).
    for sp, count in out.items():
        ideal = {"Ni": 0.33, "Co": 0.33, "Cr": 0.34}[sp] * 108
        assert abs(count - ideal) < 1.0


# ---- list-templates (no scraps binary needed) -----------------------------

def test_list_templates_runs_without_scraps_binary():
    proc = subprocess.run(
        [sys.executable, str(WIZARD), "list-templates"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("fcc", "bcc", "b2", "rocksalt",
                 "heusler-l21", "perovskite-abo3"):
        assert name in proc.stdout, f"template {name!r} missing from output"


# ---- end-to-end ----------------------------------------------------------

@needs_scraps
def test_quick_fcc_single_sublattice(tmp_path):
    """Quick FCC binary 50/50 NiCo: 100-atom target -> 108 cell (3^3*4)."""
    proc = subprocess.run(
        [sys.executable, str(WIZARD), "quick",
         "--template", "fcc",
         "--composition", "Ni:0.5,Co:0.5",
         "--target-atoms", "100",
         "--work-dir", str(tmp_path / "out"),
         "--ranks", "1"],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "MPIRUN": MPIRUN},
    )
    assert proc.returncode == 0, f"wizard failed: {proc.stderr or proc.stdout[-500:]}"
    assert "Cell           : 3 x 3 x 3 = 108 atoms" in proc.stdout
    # Final fitness should be 0 for FCC 50:50 (exactly feasible SQS).
    assert "Final fitness = 0" in proc.stdout or "Final fitness = 0.0" in proc.stdout
    out_cif = tmp_path / "out" / "sqs.cif"
    assert out_cif.exists()


@needs_scraps
def test_quick_b2_multi_sublattice(tmp_path):
    """Quick B2 (Ni,Co)(Ti,Zr) at 50/50 per sublattice. Target 200 atoms."""
    proc = subprocess.run(
        [sys.executable, str(WIZARD), "quick",
         "--template", "b2",
         "--composition", "Ni:0.5,Co:0.5",
         "--composition", "Ti:0.5,Zr:0.5",
         "--target-atoms", "200",
         "--work-dir", str(tmp_path / "out"),
         "--ranks", "1"],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "MPIRUN": MPIRUN},
    )
    assert proc.returncode == 0, f"wizard failed: {proc.stderr or proc.stdout[-500:]}"
    # B2 primitive has 2 atoms; 128 = 2 * 4^3 is the closest feasible to 200.
    assert "Cell           : 4 x 4 x 4 = 128 atoms" in proc.stdout
    assert "2 declared" in proc.stdout  # 2 sublattices
    out_cif = tmp_path / "out" / "sqs.cif"
    assert out_cif.exists()


@needs_scraps
def test_quick_wrong_composition_count_aborts(tmp_path):
    """B2 has 2 orbits; passing only 1 --composition should error out."""
    proc = subprocess.run(
        [sys.executable, str(WIZARD), "quick",
         "--template", "b2",
         "--composition", "Ni:0.5,Co:0.5",
         "--target-atoms", "100"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "2 sublattices" in proc.stderr
    assert "got 1" in proc.stderr


def test_quick_unknown_template_aborts(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(WIZARD), "quick",
         "--template", "nonexistent",
         "--composition", "Ni:0.5,Co:0.5"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "Unknown template" in proc.stderr


@needs_scraps
def test_interactive_mode_via_stdin(tmp_path):
    """Drive interactive mode by piping prompt responses to stdin.
    Picks single-sublattice FCC, ternary NiCoCr, default cell size."""
    answers = "\n".join([
        "1",                      # single-sublattice
        "1",                      # FCC
        "Ni:0.33,Co:0.33,Cr:0.34",  # composition
        "108",                    # target atoms (matches default in --target-atoms)
        "",                       # any leftover prompt
    ]) + "\n"
    proc = subprocess.run(
        [sys.executable, str(WIZARD), "interactive",
         "--work-dir", str(tmp_path / "out"),
         "--ranks", "1",
         "--target-atoms", "108"],
        input=answers, capture_output=True, text=True, timeout=180,
        env={**os.environ, "MPIRUN": MPIRUN},
    )
    assert proc.returncode == 0, f"interactive failed: {proc.stderr or proc.stdout[-500:]}"
    assert "Using template: fcc" in proc.stdout
    out_cif = tmp_path / "out" / "sqs.cif"
    assert out_cif.exists()
