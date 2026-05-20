"""Tests for the ASE interop module (tools/scraps_ase.py).

Covers I/O round-tripping (POSCAR <-> ASE Atoms with proper species
relabeling), INPUT generation from ASE primitives, and the end-to-end
`run_sqs_from_atoms` wrapper.

Skipped when:
  - ase isn't installed
  - scraps binary not built or mpirun missing (for the end-to-end test)
"""
from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.io import read as ase_read

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

from scraps_ase import (
    read_scraps_output,
    write_scraps_poscar,
    make_scraps_inputs,
    run_sqs_from_atoms,
)

REPO = TOOLS.parent
SCRAPS_BIN = REPO / "SCRAPs" / "scraps"


# ---- Fixtures --------------------------------------------------------------

@pytest.fixture
def fcc_primitive() -> Atoms:
    """FCC conventional 4-atom basis (same as our test POSCARs)."""
    return Atoms(
        symbols=["X"] * 4,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        pbc=True,
    )


@pytest.fixture
def bcc_primitive() -> Atoms:
    return Atoms(
        symbols=["X", "X"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        pbc=True,
    )


# ---- I/O tests -------------------------------------------------------------

def test_read_scraps_output_relabels_with_species(tmp_path):
    """SCRAPS.vasp atoms get relabeled to match the user-provided species
    list, in the order that ELEMENT counts dictate."""
    # Hand-craft a SCRAPS-style POSCAR with 2 species
    poscar = tmp_path / "SCRAPS.vasp"
    poscar.write_text(textwrap.dedent("""\
        Test cell
         1.0
         2.0 0.0 0.0
         0.0 2.0 0.0
         0.0 0.0 2.0
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
        """))
    atoms = read_scraps_output(poscar, species=["Ni", "Co"])
    assert len(atoms) == 8
    symbols = list(atoms.symbols)
    assert symbols[:4] == ["Ni"] * 4
    assert symbols[4:] == ["Co"] * 4


def test_write_scraps_poscar_roundtrips(fcc_primitive, tmp_path):
    """Write an ASE Atoms as POSCAR, read it back with ASE: lattice and
    fractional positions match to FP precision."""
    poscar = tmp_path / "POSCAR"
    write_scraps_poscar(fcc_primitive, poscar)
    rt = ase_read(poscar, format="vasp")
    assert len(rt) == len(fcc_primitive)
    import numpy as np
    assert np.allclose(rt.cell, fcc_primitive.cell)
    assert np.allclose(rt.get_scaled_positions(), fcc_primitive.get_scaled_positions())


def test_make_scraps_inputs_writes_expected_files(fcc_primitive, tmp_path):
    poscar, inp = make_scraps_inputs(
        tmp_path, fcc_primitive,
        element_counts=[16, 16],
        cell_dim=(2, 2, 2),
        max_shellnum=1,
        weights=[1.0],
        rng_seed=42,
    )
    assert poscar.exists() and inp.exists()
    text = inp.read_text()
    assert "TYP     2" in text
    assert "ELEMENT 16 16" in text
    assert "Cell_DIM        2 2 2" in text
    assert "RNG_SEED 42" in text


def test_make_scraps_inputs_with_sro_targets(bcc_primitive, tmp_path):
    """SRO targets get written as `SRO i j shell alpha` lines."""
    _, inp = make_scraps_inputs(
        tmp_path, bcc_primitive,
        element_counts=[64, 64],
        cell_dim=(4, 4, 4),
        max_shellnum=2,
        sro_targets=[(0, 1, 0, -1.0), (0, 1, 1, +1.0)],
    )
    text = inp.read_text()
    assert "SRO 0 1 0 -1" in text
    assert "SRO 0 1 1 1" in text


def test_make_scraps_inputs_rejects_mismatched_counts(fcc_primitive, tmp_path):
    """Helpful error when element_counts don't sum to the expected supercell size."""
    with pytest.raises(ValueError, match="sum to"):
        make_scraps_inputs(
            tmp_path, fcc_primitive,
            element_counts=[10, 10],  # sum=20, but 4*8=32 expected
            cell_dim=(2, 2, 2),
        )


def test_make_scraps_inputs_writes_spectator_lines(tmp_path):
    """HEB-style spectator basis spec gets written as
    `SPECTATOR_BASIS_SPECIES bi sp` lines."""
    # AlB2-type basis
    alb2 = Atoms(
        symbols=["X"] * 3,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.333333, 0.666667, 0.5],
            [0.666667, 0.333333, 0.5],
        ],
        cell=[[1, 0, 0], [-0.5, 0.866025, 0], [0, 0, 1.1]],
        pbc=True,
    )
    _, inp = make_scraps_inputs(
        tmp_path, alb2,
        element_counts=[10, 10, 10, 10, 10, 100],
        cell_dim=(5, 5, 2),
        max_shellnum=2,
        spectator_basis_species=[(1, 5), (2, 5)],
    )
    text = inp.read_text()
    assert "SPECTATOR_BASIS_SPECIES 1 5" in text
    assert "SPECTATOR_BASIS_SPECIES 2 5" in text


# ---- End-to-end ------------------------------------------------------------

@pytest.mark.skipif(
    not SCRAPS_BIN.exists() or shutil.which("mpirun") is None,
    reason="scraps binary not built or no mpirun",
)
def test_run_sqs_from_atoms_end_to_end(fcc_primitive, tmp_path):
    """Real ASE -> SCRAPS -> ASE round trip: run a small SQS and verify
    we get back a sensibly populated ASE Atoms object."""
    result = run_sqs_from_atoms(
        fcc_primitive,
        species=["Ni", "Co"],
        element_counts=[16, 16],
        cell_dim=(2, 2, 2),
        work_dir=tmp_path / "run",
        max_shellnum=1,
        weights=[1.0],
        nests=12,
        mcstep=100,
        auto_mcstep=7,
        total_iter=20,
        early_stop_patience=3,
        rng_seed=12345,
    )
    # Returned Atoms should have 32 atoms (2x2x2 * 4 basis)
    assert len(result) == 32
    # Half should be Ni, half Co
    syms = list(result.symbols)
    assert syms.count("Ni") == 16
    assert syms.count("Co") == 16
    # Should have hit fitness=0 on this feasible cell
    assert result.info.get("scraps_final_fitness") == 0


@pytest.mark.skipif(
    not SCRAPS_BIN.exists() or shutil.which("mpirun") is None,
    reason="scraps binary not built or no mpirun",
)
def test_run_sqs_from_atoms_with_sro_target_b2(bcc_primitive, tmp_path):
    """End-to-end B2 target: ASE BCC primitive in, B2-ordered Atoms out."""
    # Use real element symbols so ASE's read can relabel cleanly
    # (Cu/Zn is the canonical B2 binary).
    result = run_sqs_from_atoms(
        bcc_primitive,
        species=["Cu", "Zn"],
        element_counts=[16, 16],
        cell_dim=(2, 2, 4),
        work_dir=tmp_path / "b2",
        max_shellnum=2,
        weights=[1, 1],
        nests=12,
        mcstep=100,
        auto_mcstep=10,
        total_iter=20,
        early_stop_patience=5,
        rng_seed=12345,
        sro_targets=[(0, 1, 0, -1.0), (0, 1, 1, +1.0)],
    )
    assert len(result) == 32  # 2 basis atoms * 2x2x4 = 32
    syms = list(result.symbols)
    assert syms.count("Cu") == 16 and syms.count("Zn") == 16
    assert result.info.get("scraps_final_fitness") == 0
