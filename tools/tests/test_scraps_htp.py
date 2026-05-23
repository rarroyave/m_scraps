"""Tests for tools/scraps_htp.py — composition-space SQS sweeps.

Covers:
  - Composition enumeration: linear, simplex (combinatorics + edge cases)
  - CSV / JSON composition list loaders
  - composition_tag stability and filesystem safety
  - linear sweep end-to-end on FCC NiCo (binary, 3 compositions)
  - simplex sweep end-to-end on FCC NiCoCr (ternary, resolution 2 = 6 comps)
  - Resume behavior (re-running skips existing outputs)
  - Multi-sublattice --fixed argument parsing
  - Manifest schema (success/skipped/failed counts, per-result fields)
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
HTP = REPO / "tools" / "scraps_htp.py"
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


def _import_htp():
    sys.path.insert(0, str(REPO / "tools"))
    import scraps_htp
    return scraps_htp


# ---- composition enumeration ---------------------------------------------

def test_enumerate_simplex_ternary_resolution_5():
    h = _import_htp()
    comps = list(h.enumerate_simplex(3, 5))
    # C(5+3-1, 3-1) = C(7, 2) = 21
    assert len(comps) == 21
    # Every tuple sums to 5
    for c in comps:
        assert sum(c) == 5
    # Includes the three corners
    assert (5, 0, 0) in comps
    assert (0, 5, 0) in comps
    assert (0, 0, 5) in comps


def test_enumerate_simplex_quaternary_resolution_3():
    h = _import_htp()
    comps = list(h.enumerate_simplex(4, 3))
    # C(3+4-1, 4-1) = C(6, 3) = 20
    assert len(comps) == 20
    for c in comps:
        assert sum(c) == 3


def test_linear_compositions_inclusive_endpoints():
    h = _import_htp()
    out = h.linear_compositions(["Ni", "Co"], 0.0, 1.0, 0.25)
    # 0.0, 0.25, 0.5, 0.75, 1.0 = 5 points
    assert len(out) == 5
    assert out[0] == {"Ni": 0.0, "Co": 1.0}
    assert out[-1] == {"Ni": 1.0, "Co": 0.0}


def test_linear_compositions_rejects_non_binary():
    h = _import_htp()
    with pytest.raises(ValueError, match="exactly 2 species"):
        h.linear_compositions(["Ni", "Co", "Cr"], 0.0, 1.0, 0.5)


def test_simplex_compositions_fractions_sum_to_one():
    h = _import_htp()
    comps = h.simplex_compositions(["Ni", "Co", "Cr"], 10)
    assert len(comps) == 66  # C(12, 2)
    for c in comps:
        assert abs(sum(c.values()) - 1.0) < 1e-9


def test_composition_tag_is_stable_and_filesystem_safe():
    h = _import_htp()
    tag = h.composition_tag({"Ni": 0.5, "Co": 0.5})
    assert tag == "Ni0.50_Co0.50"
    # No slashes, no spaces, no shell metacharacters
    assert "/" not in tag and " " not in tag
    # Ternary
    assert h.composition_tag(
        {"Ni": 0.33, "Co": 0.33, "Cr": 0.34}
    ) == "Ni0.33_Co0.33_Cr0.34"


# ---- CSV / JSON loaders ---------------------------------------------------

def test_list_from_csv(tmp_path):
    h = _import_htp()
    path = tmp_path / "comps.csv"
    path.write_text("Ni,Co\n0.5,0.5\n0.7,0.3\n0.3,0.7\n")
    comps = h.list_compositions_from_csv(path)
    assert comps == [
        {"Ni": 0.5, "Co": 0.5},
        {"Ni": 0.7, "Co": 0.3},
        {"Ni": 0.3, "Co": 0.7},
    ]


def test_list_from_json(tmp_path):
    h = _import_htp()
    path = tmp_path / "comps.json"
    path.write_text(json.dumps([
        {"Ni": 0.5, "Co": 0.5},
        {"Ni": 0.2, "Co": 0.8},
    ]))
    comps = h.list_compositions_from_json(path)
    assert len(comps) == 2
    assert comps[0]["Co"] == 0.5


# ---- End-to-end sweeps ---------------------------------------------------

@needs_scraps
def test_linear_sweep_fcc_nico_three_compositions(tmp_path):
    """Binary FCC NiCo at 50% resolution -> 3 compositions
    (pure Ni, 50/50, pure Co). All should succeed; manifest aggregates."""
    out = tmp_path / "linear_sweep"
    proc = subprocess.run(
        [sys.executable, str(HTP), "linear",
         "--template", "fcc",
         "--species", "Ni,Co",
         "--range", "0:1",
         "--step", "0.5",
         "--target-atoms", "32",
         "--ranks-per-worker", "1",
         "--output-dir", str(out)],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "MPIRUN": MPIRUN},
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sweep_type"] == "linear"
    assert manifest["n_compositions"] == 3
    assert manifest["n_success"] == 3
    assert manifest["n_failed"] == 0
    tags = sorted(r["tag"] for r in manifest["results"])
    assert tags == ["Ni0.00_Co1.00", "Ni0.50_Co0.50", "Ni1.00_Co0.00"]
    # Each composition has an sqs.cif output
    for r in manifest["results"]:
        cif = out / r["tag"] / "sqs.cif"
        assert cif.exists(), f"missing {cif}"
    # Pure compositions have max_alpha = 0 (one species; no SRO possible).
    # Mixed composition should be near 0 (FCC 50:50 = exact SQS at small cell).


@needs_scraps
def test_simplex_sweep_fcc_nicocr_resolution_2(tmp_path):
    """Ternary FCC NiCoCr at resolution 2 -> 6 compositions on the
    coarsest non-trivial simplex grid."""
    out = tmp_path / "simplex_sweep"
    proc = subprocess.run(
        [sys.executable, str(HTP), "simplex",
         "--template", "fcc",
         "--species", "Ni,Co,Cr",
         "--resolution", "2",
         "--target-atoms", "32",
         "--ranks-per-worker", "1",
         "--output-dir", str(out)],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "MPIRUN": MPIRUN},
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    manifest = json.loads((out / "manifest.json").read_text())
    # C(2+3-1, 3-1) = C(4, 2) = 6
    assert manifest["n_compositions"] == 6
    assert manifest["sweep_type"] == "simplex"
    # All should succeed for a small FCC NiCoCr sweep
    assert manifest["n_failed"] == 0


@needs_scraps
def test_sweep_resume_skips_existing(tmp_path):
    """Re-running the same sweep should skip compositions whose
    sqs.cif already exists. The second run's manifest should report
    n_skipped > 0."""
    out = tmp_path / "resume_sweep"
    args = [
        sys.executable, str(HTP), "linear",
        "--template", "fcc",
        "--species", "Ni,Co",
        "--range", "0:1",
        "--step", "0.5",
        "--target-atoms", "32",
        "--ranks-per-worker", "1",
        "--output-dir", str(out),
    ]
    env = {**os.environ, "MPIRUN": MPIRUN}
    # First run
    proc1 = subprocess.run(args, capture_output=True, text=True,
                           timeout=300, env=env)
    assert proc1.returncode == 0
    m1 = json.loads((out / "manifest.json").read_text())
    assert m1["n_success"] == 3
    # Second run -> all skipped
    proc2 = subprocess.run(args, capture_output=True, text=True,
                           timeout=300, env=env)
    assert proc2.returncode == 0
    m2 = json.loads((out / "manifest.json").read_text())
    assert m2["n_skipped"] == 3
    assert m2["n_success"] == 0


# ---- argument parsing ----------------------------------------------------

def test_fixed_argument_parsing_perovskite():
    h = _import_htp()
    # Mock template orbits (perovskite-like, 3 orbits)
    orbits = ["A", "B", "O"]
    # Sweep orbit 0 (A site), fix orbit 1 (Ti) and orbit 2 (O)
    import argparse
    args = argparse.Namespace(
        fixed=["1:Ti:1.0", "2:O:1.0"],
    )
    swept_idx, fixed = h.parse_fixed_compositions(args, orbits)
    assert swept_idx == 0
    assert fixed == [{"Ti": 1.0}, {"O": 1.0}]


def test_fixed_argument_requires_one_unfixed_orbit():
    h = _import_htp()
    orbits = ["A", "B", "O"]
    import argparse
    args = argparse.Namespace(fixed=["1:Ti:1.0"])  # missing orbit 2
    with pytest.raises(SystemExit, match="Exactly one orbit"):
        h.parse_fixed_compositions(args, orbits)


# ---- equiatomic-not-on-grid warning ---------------------------------------

def test_simplex_warns_when_resolution_doesnt_divide_n_species(tmp_path):
    """For a 5-component sweep at resolution 4, the equiatomic
    (1/5 each) composition is not on the simplex grid because the
    grid points sum to 4. The warning should fire and suggest
    resolution 5 (the nearest multiple of 5).

    Uses --template _nonexistent_ so the sweep aborts immediately
    after _dispatch_sweep's template check -- we test the warning
    output, not the SCRAPs runner."""
    proc = subprocess.run(
        [sys.executable, str(HTP), "simplex",
         "--template", "_nonexistent_template_",
         "--species", "Co,Cr,Fe,Mn,Ni",
         "--resolution", "4",
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=30,
    )
    # Warning fires before the template-existence check
    assert "equiatomic" in proc.stderr
    assert "n_species (5)" in proc.stderr
    assert "--resolution 5" in proc.stderr
    # Template error also fires (= proves we exited before any sweep work)
    assert "Unknown template" in proc.stderr or "unknown template" in proc.stderr
    assert proc.returncode != 0


def test_simplex_no_warning_when_resolution_divides_n_species(tmp_path):
    """5 species at resolution 5 -> equiatomic IS on the grid;
    no warning. Same trick: nonexistent template to skip the sweep."""
    proc = subprocess.run(
        [sys.executable, str(HTP), "simplex",
         "--template", "_nonexistent_template_",
         "--species", "Co,Cr,Fe,Mn,Ni",
         "--resolution", "5",
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=30,
    )
    assert "equiatomic" not in proc.stderr
    assert "does not divide" not in proc.stderr
    # Template error still fires
    assert proc.returncode != 0
