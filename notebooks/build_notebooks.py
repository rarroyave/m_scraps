"""Generate the three tutorial notebooks in this directory.

Run: python3 build_notebooks.py

This builds clean, well-formatted .ipynb files programmatically rather than
trying to maintain raw JSON. Each notebook is a list of (cell_type, source)
tuples passed through nbformat.

After generating, execute end-to-end with:
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 *.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return new_markdown_cell(text.strip() + "\n")


def code(text: str) -> dict:
    return new_code_cell(text.strip() + "\n")


def write_notebook(name: str, cells: list[dict]) -> Path:
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    path = HERE / name
    nbformat.write(nb, path)
    return path


# ---------------------------------------------------------------------------
# Notebook 1: your first SCRAPS run
# ---------------------------------------------------------------------------

nb1 = [
    md("""
# 1. Your first SCRAPS run

The shortest path from "I have an ASE primitive cell" to "I have a special
quasi-random structure (SQS) supercell" using the `scraps_ase` wrapper.

We'll build a 108-atom FCC NiCoCr SQS — the classic equiatomic CCA test case.
"""),
    md("""
## Setup

`tools/scraps_ase.py` ships with the repo. We add `tools/` to the path so
imports work without an install step.
"""),
    code("""
import sys
from pathlib import Path
REPO = Path.cwd().parent  # this notebook lives in notebooks/
sys.path.insert(0, str(REPO / "tools"))

from ase.build import bulk
from scraps_ase import run_sqs_from_atoms
"""),
    md("""
## Build the primitive cell

ASE's `bulk` helper gives us a 4-atom conventional FCC cell. SCRAPS will tile
this by `Cell_DIM` to form the supercell.
"""),
    code("""
primitive = bulk("Cu", "fcc", a=3.56, cubic=True)
print(primitive)
print("Primitive atoms:", len(primitive))
"""),
    md("""
## Run SCRAPS

`run_sqs_from_atoms` writes the POSCAR + INPUT, finds the right `mpirun`,
launches scraps, and reads back `SCRAPS.vasp` as an ASE Atoms object with
proper species labels.

`element_counts=[36, 36, 36]` ⇒ 108 atoms total (= 4 basis × 3³ cells).
`max_shellnum=2` weights the first two coordination shells equally. `rng_seed`
makes the run reproducible across machines.
"""),
    code("""
sqs = run_sqs_from_atoms(
    primitive,
    species=["Ni", "Co", "Cr"],
    element_counts=[36, 36, 36],
    cell_dim=(3, 3, 3),
    max_shellnum=2,
    weights=[1, 1],
    rng_seed=42,
    work_dir=Path.cwd() / "nb1_first_run",
)
print("Atoms:", len(sqs))
print("Symbols:", sqs.symbols)
print("Final fitness:", sqs.info.get("scraps_final_fitness"))
"""),
    md("""
## Verify with the independent SRO calculator

`verify_sro.py` reads `SCRAPS.vasp` and computes the Warren-Cowley α matrix
directly via pymatgen — no shared code with the C++ optimizer, so it's a true
cross-check. For an SQS we want α ≈ 0 at every entry of every weighted shell.
"""),
    code("""
import subprocess
verify = REPO / "tools" / "verify_sro.py"
work_dir = sqs.info["scraps_work_dir"]
result = subprocess.run(
    [sys.executable, str(verify),
     str(Path(work_dir) / "SCRAPS.vasp"),
     "--species", "Ni,Co,Cr", "--shells", "2"],
    capture_output=True, text=True,
)
print(result.stdout)
"""),
    md("""
## Use the output

`sqs` is a standard ASE Atoms object — save in any format ASE supports.
"""),
    code("""
out = Path.cwd() / "ni_co_cr_sqs.cif"
sqs.write(str(out))
print(f"Wrote {out}")
print(f"Lattice: {sqs.cell.cellpar()}")
"""),
    md("""
That's it — a 108-atom NiCoCr SQS in a few lines. The next notebook covers
targeting **non-zero** SRO values (literature CSRO patterns); the third covers
multi-sublattice HEB structures.
"""),
]


# ---------------------------------------------------------------------------
# Notebook 2: targeting CSRO from literature
# ---------------------------------------------------------------------------

nb2 = [
    md("""
# 2. Targeting CSRO from literature

Special quasi-random structures (α=0) are useful, but real alloys often have
**chemical short-range order** — measurably non-zero α values at specific
species pairs and shells, often determined by APT, neutron scattering, or DFT.

The upstream-extending `SRO` INPUT keyword in `scraps-perpair` lets you target
those values directly. This notebook reproduces a literature-style CSRO
pattern on NiCoCr and confirms the structure achieves it.
"""),
    md("""
## A literature-style CSRO pattern

NiCoCr is the canonical CSRO testbed. A representative pattern from the
literature is:

- α(Ni-Cr) at NN ≈ −0.10 (Cr-Cr repulsion drives Ni-Cr to prefer adjacency)
- α(Co-Cr) at NN ≈ −0.05 (weaker preference)
- α(Ni-Co) at NN ≈ 0 (effectively random)

The sum-rule constraint $\\sum_j c_j \\alpha_{ij} = 0$ implies α(Cr-Cr) at NN
ends up positive (Cr-Cr clustering avoided), which we don't have to target
explicitly — it emerges.
"""),
    code("""
import sys
from pathlib import Path
REPO = Path.cwd().parent
sys.path.insert(0, str(REPO / "tools"))

from ase.build import bulk
from scraps_ase import run_sqs_from_atoms

primitive = bulk("Cu", "fcc", a=3.56, cubic=True)

# (species_i, species_j, shell_index, alpha_target)
sro_targets = [
    (0, 2, 0, -0.10),  # Ni-Cr at NN
    (1, 2, 0, -0.05),  # Co-Cr at NN
    (0, 1, 0,  0.00),  # Ni-Co at NN
]

csro = run_sqs_from_atoms(
    primitive,
    species=["Ni", "Co", "Cr"],
    element_counts=[36, 36, 36],
    cell_dim=(3, 3, 3),
    max_shellnum=2,
    weights=[2.0, 1.0],   # weight NN twice as much as 2NN
    sro_targets=sro_targets,
    rng_seed=42,
    work_dir=Path.cwd() / "nb2_csro",
)
print("Final fitness:", csro.info["scraps_final_fitness"])
"""),
    md("""
## Verify the achieved alphas

`verify_sro.py` can be pointed at the INPUT file via `--target` and report
the per-pair pass/fail against a tolerance — useful for CI gates.
"""),
    code("""
import subprocess
verify = REPO / "tools" / "verify_sro.py"
work_dir = csro.info["scraps_work_dir"]
result = subprocess.run(
    [sys.executable, str(verify),
     str(Path(work_dir) / "SCRAPS.vasp"),
     "--species", "Ni,Co,Cr", "--shells", "2"],
    capture_output=True, text=True,
)
print(result.stdout)
"""),
    md("""
## Compare to a pure random SQS

The same composition with `sro_targets=[]` gives the SQS baseline. Notice how
α(Ni-Cr) at shell 0 drops to ~0 (random) while in the CSRO run it sits near
the −0.10 target.
"""),
    code("""
sqs = run_sqs_from_atoms(
    primitive,
    species=["Ni", "Co", "Cr"],
    element_counts=[36, 36, 36],
    cell_dim=(3, 3, 3),
    max_shellnum=2,
    weights=[1, 1],
    rng_seed=42,
    work_dir=Path.cwd() / "nb2_sqs_baseline",
)
result = subprocess.run(
    [sys.executable, str(verify),
     str(Path(sqs.info["scraps_work_dir"]) / "SCRAPS.vasp"),
     "--species", "Ni,Co,Cr", "--shells", "2"],
    capture_output=True, text=True,
)
print(result.stdout)
"""),
    md("""
The contrast — same composition, same cell, different SRO signatures — is the
unique value of the per-pair SRO API. Upstream SCRAPS only supported a single
scalar α target applied to every species pair, so this kind of targeted
ordering wasn't expressible.

The classical-ordered tests in `tests/{b2,l12,l10,d03,l11,l21}_*` use the
same machinery with α targets at the maximum (±1, ±1/3) bounds.
"""),
]


# ---------------------------------------------------------------------------
# Notebook 3: HEB structures end to end
# ---------------------------------------------------------------------------

nb3 = [
    md("""
# 3. HEB structures end to end

High-entropy diborides (HEB) like (Hf,Zr,Ti,Ta,Nb)B₂ have an AlB₂-type
structure — a chemically disordered metal sublattice (M) and a chemically
fixed boron sublattice (the **spectator**). SCRAPS only optimizes the M
sublattice; pairs involving boron are excluded from the fitness.

This notebook builds the canonical 5-metal 150-atom test case used in our
benchmarks.
"""),
    md("""
## Build the AlB₂ primitive cell

3-atom basis: one M position at (0, 0, 0), two B positions at the honeycomb
sites. We give them placeholder species; the actual species are assigned via
`ELEMENT` in INPUT.
"""),
    code("""
import sys
from pathlib import Path
REPO = Path.cwd().parent
sys.path.insert(0, str(REPO / "tools"))

import numpy as np
from ase import Atoms
from scraps_ase import run_sqs_from_atoms

# AlB2-type cell with hexagonal lattice and c/a = 1.1
a = 3.10  # rough HEB average lattice parameter (Å)
c = 1.1 * a
cell = [
    [a, 0, 0],
    [-0.5 * a, np.sqrt(3) / 2 * a, 0],
    [0, 0, c],
]
primitive = Atoms(
    symbols=["X"] * 3,
    scaled_positions=[
        [0.0, 0.0, 0.0],          # M (basis 0)
        [1/3, 2/3, 0.5],          # B (basis 1)
        [2/3, 1/3, 0.5],          # B (basis 2)
    ],
    cell=cell,
    pbc=True,
)
print(primitive)
"""),
    md("""
## Find the right cell size with `suggest_cell.py`

For a 5-metal equiatomic target on the M sublattice with ~150 total atoms,
the closest feasible cell is `Cell_DIM 5 5 2 = 150 atoms` (50 M + 100 B).
We pass `--spectator-basis-species "1:5,2:5"` so the feasibility math uses
M-only coordination.
"""),
    code("""
import subprocess
work = Path.cwd() / "nb3_heb"
work.mkdir(exist_ok=True)
primitive.write(str(work / "POSCAR_primitive.vasp"))

suggest = REPO / "tools" / "suggest_cell.py"
result = subprocess.run(
    [sys.executable, str(suggest),
     str(work / "POSCAR_primitive.vasp"),
     "--composition", "0.2,0.2,0.2,0.2,0.2",
     "--target-atoms", "150",
     "--shells", "2",
     "--spectator-basis-species", "1:5,2:5",
     "--non-cubic"],
    capture_output=True, text=True,
)
print(result.stdout)
"""),
    md("""
## Run SCRAPS

Pass `spectator_basis_species=[(1, 5), (2, 5)]` to declare basis positions 1
and 2 as boron (species index 5 in `ELEMENT`). `species` lists the 6 element
symbols in `ELEMENT` order; the last must match the spectator species index.

**Important subtlety for HEB:** SCRAPS' `MAXSHELLNUM` counts *geometric*
shells over all atoms. In AlB₂ the first two shells are B-B and M-B
distances — they trivially contribute zero to the fitness because the
M-only filter excludes them. The first shell that actually carries M-M
pairs is **shell 2** (in-plane M-M at distance `a`), so we need at least
`max_shellnum=4` to optimize the first few M-M shells. With
`weights=[0, 0, 1, 1]` we explicitly ignore the B-only shells and put all
weight on the M-M ones.
"""),
    code("""
heb = run_sqs_from_atoms(
    primitive,
    species=["Hf", "Zr", "Ti", "Ta", "Nb", "B"],
    element_counts=[10, 10, 10, 10, 10, 100],
    cell_dim=(5, 5, 2),
    max_shellnum=4,
    weights=[0, 0, 1, 1],     # ignore the B-B / M-B shells, weight M-M
    spectator_basis_species=[(1, 5), (2, 5)],
    rng_seed=42,
    work_dir=work / "run",
)
print("Atoms:", len(heb))
print("Symbol counts:", {s: list(heb.symbols).count(s)
                         for s in set(heb.symbols)})
print("Final fitness:", heb.info["scraps_final_fitness"])
"""),
    md("""
## Verify with spectator-aware `verify_sro.py`

`--spectator-species B` tells the verifier to compute the α matrix on the M
sublattice only — boron pairs are excluded. Note: SCRAPS' `MAXSHELLNUM` and
`verify_sro --shells` count *geometric* shells over all atoms, so in AlB₂ the
first couple of shells are B-B and M-B distances; the M-M shells start at
shell 3-4. We pass `--shells 6` here to actually see the M-M α matrix
populate.
"""),
    code("""
verify = REPO / "tools" / "verify_sro.py"
work_dir = heb.info["scraps_work_dir"]
result = subprocess.run(
    [sys.executable, str(verify),
     str(Path(work_dir) / "SCRAPS.vasp"),
     "--species", "Hf,Zr,Ti,Ta,Nb,B",
     "--shells", "6",
     "--spectator-species", "B"],
    capture_output=True, text=True,
)
print(result.stdout)
"""),
    md("""
Same recipe applies for any one-spectator HEB-class structure (other
borides, carbides like the (Hf,Zr,Ti,Ta,Nb)C high-entropy carbides, or
fluorite-like high-entropy oxides where one cation sublattice is the
designed one and the anion sublattice is the spectator).

For truly multi-sublattice cases where multiple sublattices have independent
designed compositions (σ-phase, Heusler with independent X₂/Y/Z compositions),
SCRAPS aborts with a guidance message pointing to ATAT's `mcsqs`. See
`docs/mcsqs_comparison.md` for the comparison.
"""),
]


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name, cells in [
        ("01_first_scraps_run.ipynb", nb1),
        ("02_targeting_csro.ipynb", nb2),
        ("03_heb_end_to_end.ipynb", nb3),
    ]:
        path = write_notebook(name, cells)
        print(f"Wrote {path}")
