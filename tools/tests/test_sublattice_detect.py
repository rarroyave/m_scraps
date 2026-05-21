"""Tests for tools/sublattice_detect.py (Wyckoff-orbit detection on POSCARs).

Locks in the orbit count and Wyckoff labels for a few canonical structures
where the answer is unambiguous, plus the CLI surface (JSON, --write).

Fixtures (all conventional cubic cells):
  - FCC, 1 species, 4 atoms  -> 1 orbit (Wyckoff 4a in Fm-3m)
  - Rocksalt NaCl, 2 species, 8 atoms  -> 2 orbits (4a Na + 4b Cl)
  - Full Heusler Cu2MnAl, 3 species, 16 atoms  -> 3 orbits (4a Mn + 4b Al + 8c Cu)
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
TOOL = REPO / "tools" / "sublattice_detect.py"

pytest.importorskip("pymatgen")
pytest.importorskip("spglib")


# ---- Fixtures -------------------------------------------------------------

FCC_POSCAR = textwrap.dedent("""\
    FCC Cu conventional cell
     1.0
     3.615 0.000 0.000
     0.000 3.615 0.000
     0.000 0.000 3.615
    Cu
     4
    Direct
     0.0 0.0 0.0
     0.5 0.5 0.0
     0.5 0.0 0.5
     0.0 0.5 0.5
    """)

NACL_POSCAR = textwrap.dedent("""\
    NaCl rocksalt conventional cell
     1.0
     5.640 0.000 0.000
     0.000 5.640 0.000
     0.000 0.000 5.640
    Na Cl
     4 4
    Direct
     0.0 0.0 0.0
     0.5 0.5 0.0
     0.5 0.0 0.5
     0.0 0.5 0.5
     0.5 0.5 0.5
     0.0 0.0 0.5
     0.0 0.5 0.0
     0.5 0.0 0.0
    """)

# Full Heusler Cu2MnAl in the L2_1 conventional cell. Mn at 4a, Al at 4b,
# Cu at 8c. Lattice parameter ~5.95 Å (Cu2MnAl experimental).
HEUSLER_POSCAR = textwrap.dedent("""\
    Cu2MnAl full Heusler L2_1
     1.0
     5.949 0.000 0.000
     0.000 5.949 0.000
     0.000 0.000 5.949
    Mn Al Cu
     4 4 8
    Direct
     0.00 0.00 0.00
     0.00 0.50 0.50
     0.50 0.00 0.50
     0.50 0.50 0.00
     0.50 0.50 0.50
     0.50 0.00 0.00
     0.00 0.50 0.00
     0.00 0.00 0.50
     0.25 0.25 0.25
     0.25 0.25 0.75
     0.25 0.75 0.25
     0.75 0.25 0.25
     0.75 0.75 0.75
     0.75 0.75 0.25
     0.75 0.25 0.75
     0.25 0.75 0.75
    """)


def run_tool(args: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(TOOL)] + args,
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---- Orbit counts ---------------------------------------------------------

def test_fcc_one_orbit(tmp_path):
    (tmp_path / "POSCAR").write_text(FCC_POSCAR)
    rc, out, err = run_tool(["POSCAR"], tmp_path)
    assert rc == 0, f"tool failed: {err}"
    assert "Space group: Fm-3m" in out
    assert "Wyckoff orbits: 1" in out
    assert "only one orbit detected" in out  # advisory note


def test_rocksalt_two_orbits(tmp_path):
    (tmp_path / "POSCAR").write_text(NACL_POSCAR)
    rc, out, err = run_tool(["POSCAR"], tmp_path)
    assert rc == 0, f"tool failed: {err}"
    assert "Space group: Fm-3m" in out
    assert "Wyckoff orbits: 2" in out
    # Each draft line should appear exactly once with the right basis indices.
    assert "SUBLATTICE 0 BASIS 0 1 2 3 SPECIES <fill in>" in out
    assert "SUBLATTICE 1 BASIS 4 5 6 7 SPECIES <fill in>" in out


def test_heusler_three_orbits(tmp_path):
    (tmp_path / "POSCAR").write_text(HEUSLER_POSCAR)
    rc, out, err = run_tool(["POSCAR"], tmp_path)
    assert rc == 0, f"tool failed: {err}"
    assert "Fm-3m" in out
    assert "Wyckoff orbits: 3" in out
    # Multiplicities: 4 (Mn at 4a), 4 (Al at 4b), 8 (Cu at 8c). Order in
    # the printed table follows first appearance in the POSCAR.
    lines = out.splitlines()
    mult_lines = [l for l in lines if l.lstrip().startswith(("0  ", "1  ", "2  "))]
    # Extract the multiplicity column (third whitespace-separated token).
    mults = []
    for l in mult_lines:
        toks = l.split()
        if len(toks) >= 3:
            try:
                mults.append(int(toks[2]))
            except ValueError:
                pass
    assert sorted(mults) == [4, 4, 8], f"expected [4,4,8] multiplicities; got {mults}"


# ---- JSON output ----------------------------------------------------------

def test_json_round_trips(tmp_path):
    (tmp_path / "POSCAR").write_text(NACL_POSCAR)
    rc, out, err = run_tool(["POSCAR", "--json"], tmp_path)
    assert rc == 0
    payload = json.loads(out)
    assert payload["space_group_symbol"] == "Fm-3m"
    assert payload["space_group_number"] == 225
    assert payload["n_basis"] == 8
    assert payload["n_orbits"] == 2
    assert len(payload["orbits"]) == 2
    orb0, orb1 = payload["orbits"]
    assert orb0["basis_indices"] == [0, 1, 2, 3]
    assert orb1["basis_indices"] == [4, 5, 6, 7]
    assert orb0["multiplicity"] == 4
    assert orb1["multiplicity"] == 4
    assert orb0["wyckoff_letter"] == "a"
    assert orb1["wyckoff_letter"] == "b"
    assert "SUBLATTICE 0 BASIS 0 1 2 3" in orb0["draft_sublattice_line"]


# ---- --write behavior ------------------------------------------------------

def test_write_appends_to_input(tmp_path):
    (tmp_path / "POSCAR").write_text(NACL_POSCAR)
    inp = tmp_path / "INPUT"
    inp.write_text("TYP 2\nELEMENT 4 4\n")
    rc, out, err = run_tool(["POSCAR", "--write", "INPUT"], tmp_path)
    assert rc == 0, f"tool failed: {err}"
    final = inp.read_text()
    assert "TYP 2" in final and "ELEMENT 4 4" in final  # original preserved
    assert "SUBLATTICE 0 BASIS 0 1 2 3 SPECIES <fill in>" in final
    assert "SUBLATTICE 1 BASIS 4 5 6 7 SPECIES <fill in>" in final
    assert "# Sublattice detection" in final
    assert "Wrote draft SUBLATTICE lines" in err


def test_write_refuses_existing_sublattice(tmp_path):
    """If INPUT already has SUBLATTICE lines, --write should refuse
    (forcing the user to either delete them or pass --overwrite)."""
    (tmp_path / "POSCAR").write_text(NACL_POSCAR)
    inp = tmp_path / "INPUT"
    inp.write_text("TYP 2\nSUBLATTICE 0 BASIS 0 1 SPECIES 0\n")
    rc, out, err = run_tool(["POSCAR", "--write", "INPUT"], tmp_path)
    assert rc != 0
    assert "already contains SUBLATTICE" in err
    # Original file untouched
    assert inp.read_text() == "TYP 2\nSUBLATTICE 0 BASIS 0 1 SPECIES 0\n"


def test_write_overwrite_replaces_existing(tmp_path):
    (tmp_path / "POSCAR").write_text(NACL_POSCAR)
    inp = tmp_path / "INPUT"
    inp.write_text("TYP 2\nSUBLATTICE 99 BASIS 0 SPECIES 0\nELEMENT 4 4\n")
    rc, out, err = run_tool(
        ["POSCAR", "--write", "INPUT", "--overwrite"], tmp_path
    )
    assert rc == 0, f"tool failed: {err}"
    final = inp.read_text()
    assert "SUBLATTICE 99" not in final  # old line stripped
    assert "SUBLATTICE 0 BASIS 0 1 2 3" in final
    assert "TYP 2" in final and "ELEMENT 4 4" in final  # non-SUBLATTICE preserved


# ---- symprec behavior -----------------------------------------------------

def test_symprec_loose_keeps_single_orbit_for_perturbed_fcc(tmp_path):
    """A very slightly perturbed FCC (sub-millishell noise) should still
    look like a single orbit under a loose symprec, and split into more
    orbits under a strict symprec. Lock in both behaviors."""
    perturbed = textwrap.dedent("""\
        Perturbed FCC
         1.0
         3.615 0.000 0.000
         0.000 3.615 0.000
         0.000 0.000 3.615
        Cu
         4
        Direct
         0.0000 0.0000 0.0000
         0.5001 0.5001 0.0000
         0.5000 0.0000 0.5000
         0.0000 0.5000 0.5000
        """)
    (tmp_path / "POSCAR").write_text(perturbed)

    # Loose: 1e-2 absorbs the ~1e-4 noise -> 1 orbit.
    rc, out, err = run_tool(["POSCAR", "--symprec", "1e-2"], tmp_path)
    assert rc == 0
    assert "Wyckoff orbits: 1" in out

    # Strict: 1e-6 sees the noise -> more than 1 orbit (the perturbed
    # site is no longer symmetry-equivalent to the others).
    rc, out, err = run_tool(["POSCAR", "--symprec", "1e-6"], tmp_path)
    assert rc == 0
    # Don't pin the exact count -- spglib's strict mode can return 2 or
    # more depending on how the perturbation reduces symmetry. Just
    # assert it's more than 1.
    import re
    m = re.search(r"Wyckoff orbits:\s*(\d+)", out)
    assert m is not None
    assert int(m.group(1)) > 1
