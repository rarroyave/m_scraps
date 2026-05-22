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

For structures with **two or more chemically-disordered sublattices**
(rocksalt HEA-carbides where the metal sublattice is disordered AND
the carbon sublattice is also disordered, double perovskites with
independent A and B site mixing, Heusler L2₁ with independent X/Y/Z
compositions), see **notebook 4** — v0.4.1's `SUBLATTICE` keyword and
the `sublattices=` parameter on `run_sqs_from_atoms` handle these
cases directly.
"""),
]


# ---------------------------------------------------------------------------
# Notebook 4: multi-sublattice SQS (v0.4.1)
# ---------------------------------------------------------------------------

nb4 = [
    md("""
# 4. Multi-sublattice SQS — (Ni,Co)(Ti,Zr) B2

The previous notebooks all treated structures with one chemically-disordered
sublattice (NiCoCr FCC, NiCoCr CSRO, HEB with spectator boron). v0.4.1 adds
support for structures with **two or more independent disordered sublattices**.

This notebook walks through:

1. The `SUBLATTICE` INPUT keyword and the `sublattices=` parameter on
   `run_sqs_from_atoms`.
2. A B2-type (Ni,Co)(Ti,Zr) cell at 50/50 on each sublattice, targeting an
   SQS where Ni-Co and Ti-Zr are independently random.
3. A CSRO variant — Ni-Co perfectly ordered on the metal sublattice,
   Ti-Zr 50%-clustered on the anion sublattice — to show the per-pair α
   API working across sublattices simultaneously.
4. Verification via a per-sublattice α calculator (cannot use
   `verify_sro.py` with `--spectator-species` here because there's no
   spectator).

For Wyckoff-orbit-based sublattice detection from an arbitrary POSCAR, the
`tools/sublattice_detect.py` helper is the right place to start; this
notebook constructs sublattices by hand for the simple B2 case.
"""),
    md("""
## Setup
"""),
    code("""
import sys, subprocess, json, math, warnings
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import bulk

REPO = Path.cwd().parent
sys.path.insert(0, str(REPO / "tools"))
from scraps_ase import run_sqs_from_atoms

warnings.filterwarnings("ignore")
"""),
    md("""
## Build the B2 primitive

The B2 (CsCl-type) cell has two basis positions: M at (0,0,0) and X at
(½,½,½). The species at each site is just a placeholder for SCRAPS, which
overrides via `ELEMENT` — we use "Ni" and "Ti" here, but SCRAPS treats them
as basis-position labels, not chemistry.
"""),
    code("""
primitive = Atoms(
    symbols=["Ni", "Ti"],
    scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
    cell=[(3.0, 0, 0), (0, 3.0, 0), (0, 0, 3.0)],
    pbc=True,
)
print(primitive)
"""),
    md("""
## Pattern 1: SQS — all sublattices independently random

Per-sublattice SQS means α=0 inside each sublattice — Ni and Co are
randomly distributed on the M sublattice, Ti and Zr are randomly
distributed on the X sublattice. The `SUBLATTICE` keyword expresses this
partition; SCRAPS' swap-proposal sites are gated so a Ni atom can only
swap with a Co atom (same sublattice), never with a Ti or Zr.

At 4×4×4 (128 atoms), this is feasible: 32 Ni + 32 Co on the M
sublattice (64 sites), 32 Ti + 32 Zr on X (64 sites).
"""),
    code("""
sqs = run_sqs_from_atoms(
    primitive,
    species=["Ni", "Co", "Ti", "Zr"],
    element_counts=[32, 32, 32, 32],
    cell_dim=(4, 4, 4),
    max_shellnum=5,
    weights=[0, 1, 1, 0, 1],   # shell 0 = M-X 1NN (cross-sub, deterministic);
                                # 1,2,4 are intra-sub; 3 is cross-sub 4NN
    sublattices=[
        (0, [0], [0, 1]),       # M sublattice: basis 0, species Ni and Co
        (1, [1], [2, 3]),       # X sublattice: basis 1, species Ti and Zr
    ],
    rng_seed=1,
    early_stop_patience=5,
    work_dir=Path.cwd() / "nb4_sqs",
)
print(f"Final fitness: {sqs.info['scraps_final_fitness']:.4f}")
"""),
    md("""
## Verify per-sublattice α

`verify_sro.py` computes α against the **global** c_j = N_j/N, which is
misleading for multi-sublattice (it would report all cross-sublattice
α values as ±1 trivially). For multi-sublattice we want per-sublattice
α: c_j^τ = N_j / N_τ where τ is j's sublattice.

The B2 comparison script `docs/compare_b2_multisub.py` includes a
self-contained calculator we can import.
"""),
    code("""
sys.path.insert(0, str(REPO / "docs"))
from compare_b2_multisub import compute_alpha_per_shell, SUBLATTICE_OF, fix_scraps_species

out_poscar = Path(sqs.info["scraps_work_dir"]) / "SCRAPS.vasp"
fix_scraps_species(out_poscar, ["Ni", "Co", "Ti", "Zr"])

shells = compute_alpha_per_shell(
    out_poscar, species=["Ni", "Co", "Ti", "Zr"],
    sublattice_of=SUBLATTICE_OF, n_shells=4
)
for sh in shells:
    print(f"shell {sh['index']}: r=[{sh['r_min']:.3f}, {sh['r_max']:.3f}]"
          f"  max|α| intra-sub = {sh['max_alpha_intra']:.4f}"
          f"  max|α| inter-sub = {sh['max_alpha_inter']:.4f}")
"""),
    md("""
The intra-sublattice α values should be ~0 at every intra shell (1, 2, 4 if
detected) — confirming we have per-sublattice SQS quality. The 1NN
cross-sublattice α reflects the natural M-X coordination structure and is
determined by composition (not optimizable).

## Pattern 2: CSRO — Ni-Co ordered on M, Ti-Zr clustered on X

Per-pair α targets work across sublattices simultaneously. Here we target:

- α(Ni-Co) at 2NN intra (M sublattice) = -1.0   → perfect alternation
- α(Ti-Zr) at 2NN intra (X sublattice) = +0.5   → moderate clustering

Set WEIGHT only on shell 1 — perfect Ni-Co alternation inherently forces
non-zero α at shells 2 and 4 (the 3D bipartite pattern induces correlations
at all distances), so weighting those shells too would prevent the optimizer
from hitting α=-1 at shell 1.
"""),
    code("""
csro = run_sqs_from_atoms(
    primitive,
    species=["Ni", "Co", "Ti", "Zr"],
    element_counts=[32, 32, 32, 32],
    cell_dim=(4, 4, 4),
    max_shellnum=5,
    weights=[0, 1, 0, 0, 0],   # focus on the targeted shell only
    sublattices=[
        (0, [0], [0, 1]),
        (1, [1], [2, 3]),
    ],
    sro_targets=[
        (0, 1, 1, -1.0),   # Ni-Co at 2NN: perfect ordering
        (2, 3, 1, +0.5),   # Ti-Zr at 2NN: 50%% clustering
    ],
    rng_seed=1,
    early_stop_patience=5,
    work_dir=Path.cwd() / "nb4_csro",
)
print(f"Final fitness: {csro.info['scraps_final_fitness']:.4f}")
"""),
    code("""
out_poscar = Path(csro.info["scraps_work_dir"]) / "SCRAPS.vasp"
fix_scraps_species(out_poscar, ["Ni", "Co", "Ti", "Zr"])
shells = compute_alpha_per_shell(
    out_poscar, species=["Ni", "Co", "Ti", "Zr"],
    sublattice_of=SUBLATTICE_OF, n_shells=4
)
# Inspect the (Ni, Co) and (Ti, Zr) alphas at shell 1 (2NN intra-sub).
sh1 = next(s for s in shells if s["index"] == 1)
print(f"α(Ni-Co) @ 2NN  = {sh1['alpha'][0][1]:+.4f}   (target -1.000)")
print(f"α(Ti-Zr) @ 2NN  = {sh1['alpha'][2][3]:+.4f}   (target +0.500)")
"""),
    md("""
SCRAPS should hit both targets to within numerical precision in a couple
of seconds. The per-pair SRO targeting works **independently per
sublattice** — α(Ni-Co) on M can be -1 while α(Ti-Zr) on X is +0.5; the
fitness function correctly uses per-sublattice c_j^τ for each pair.

## When to use Wyckoff-orbit auto-detection

For more complex structures (rocksalt HEA-carbides, full Heusler with
3 Wyckoff orbits, double perovskites, spinels), figuring out the
basis-position-to-sublattice mapping by hand becomes error-prone.
`tools/sublattice_detect.py` reads a POSCAR via pymatgen, runs Wyckoff
analysis via spglib, and emits draft `SUBLATTICE` blocks you paste into
INPUT. Two basis positions in the same Wyckoff orbit share identical
pair-correlation environments under symmetry, so the orbit partition is
the right default.

```
python3 tools/sublattice_detect.py POSCAR --write INPUT
```

emits one `SUBLATTICE` line per orbit with `<fill in>` SPECIES
placeholders that you edit by hand.

## Validation

The full mcsqs comparison for (Ni,Co)(Ti,Zr) B2 across all four patterns
(SQS / ordered / clustered / mixed) at N=4 and N=6 is documented in
`docs/mcsqs_comparison.md` under the **Multi-sublattice** section.
Summary: SCRAPS converges to exact targets in <1.3 s at both cell sizes;
mcsqs has no direct per-pair-α API and would require correlation-
engineering for an equivalent CSRO comparison.
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
        ("04_multi_sublattice.ipynb", nb4),
    ]:
        path = write_notebook(name, cells)
        print(f"Wrote {path}")
