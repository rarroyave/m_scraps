# scraps-perpair

A fork of [DuaneDJohnson/Hybrid-Cuckoo-Search](https://github.com/DuaneDJohnson/Hybrid-Cuckoo-Search)
(SCRAPS — Super-Cell Random APproximates) extended to support
**per-pair, per-shell Warren-Cowley α targets** instead of a single hard-coded
scalar applied uniformly to every species pair.

## What's new vs upstream

| Feature | Upstream | This fork |
| --- | --- | --- |
| Random / SQS-style supercells | ✓ | ✓ (backward-compatible) |
| Scalar α target per shell | requires source edit | requires source edit |
| **Per-pair α_μν target per shell** | not supported | **supported via `SRO` INPUT keyword** |
| `printtoscreen()` α output | uses incorrect formula (off by ~N_i, asymmetric) | correct Warren-Cowley, exactly symmetric |
| Uninitialized `totalatoms` in `readPOSCAR` (upstream issue #3) | open | fixed |
| `WEIGHTS` mismatch claims to stop but doesn't | open | aborts cleanly via `MPI_Abort` |

See the commit messages on `main` for the bug-by-bug breakdown.

## Install and build

Requires a working MPI compiler, Eigen headers, and a C++17 (or newer)
compiler. The canonical checkout is `~/src/tools/scraps-perpair`. On macOS with
Homebrew:

```bash
brew install eigen open-mpi
./install.sh
```

The installer discovers the Homebrew prefix, selects matching `mpic++` and
`mpirun` binaries, builds the C++ core, creates an isolated Python environment
from `tools/requirements-lock.txt`, and installs stable commands under
`~/.local/bin`: `scraps-wizard`, `scraps-run`, `scraps-verify`, and
`scraps-htp`. Set `MPICXX`, `MPIRUN`, or `EIGEN_INC` explicitly to override
auto-detection. This avoids accidentally pairing Anaconda's MPICH launcher with
a Homebrew Open MPI build.

Output binary: `SCRAPs/scraps`.

Run the complete regression suite with `./scripts/test.sh -q`. MPI needs local
socket access, so restricted execution sandboxes may need to permit local MPI
communication. Store production sweeps and optimizer outputs outside Git under
`~/runs/scraps-perpair` and run long jobs durably on the Mac Studio.

### Launching: use the matching `mpirun`

PATH-resolved `mpirun` is wrong when multiple MPI installations are
present (a common case on macOS: anaconda ships MPICH's Hydra `mpirun`
and ends up in `$PATH` ahead of Homebrew's Open MPI). Launching an
Open-MPI-built `scraps` with MPICH's Hydra `mpirun` (or vice versa)
silently falls back to **singleton mode**: each process believes it is
the lone rank 0, the four "ranks" can't communicate, each does an
independent search, and only one rank's output survives. SCRAPS now
detects this and prints a loud warning at startup, but the cleanest
fix is to launch with the matching mpirun:

```bash
/opt/homebrew/bin/mpirun -np 4 SCRAPs/scraps     # macOS Homebrew
```

The installed launchers, `tools/run_sqs.py`, and the pytest suite auto-resolve the right
`mpirun` by looking for it next to the `mpic++` from `build.sh`
(controlled by the `MPICXX` env var) before falling back to PATH.

## INPUT file format

All upstream keywords are unchanged. New directives:

```
SRO  <species_i>  <species_j>  <shell_index>  <alpha_target>

AUTO_MCSTEP          <k>          # MCSTEP = max(50, k * N) once N is known
EARLY_STOP_PATIENCE  <n>          # stop after n outer iters w/o improvement
EARLY_STOP_TOL       <tol>        # improvement threshold (default 1e-6)
SPECTATOR_BASIS_SPECIES  <bi>  <s>   # see HEB section below
```

`SRO`: species indices are 0-based and follow the order in `ELEMENT`.
Shell index is 0-based (0 = nearest-neighbor). Lines may appear in any
order; entries are symmetrized automatically (α_ij = α_ji). Pairs not
mentioned default to α = 0.

`AUTO_MCSTEP`: convenient way to scale MCSTEP with supercell size; the
factor `k=7` is what we use in the cell-size benchmarks. Overrides any
explicit `MCSTEP` line when set to a positive integer.

`EARLY_STOP_PATIENCE`: stop the optimization once the global best
fitness has not improved by more than `EARLY_STOP_TOL` for that many
consecutive outer iterations. Default 0 = disabled (run to TOTALIter).
Use `patience=3` for typical SQS runs; `patience=5` for harder cases
where you want a few extra iterations to confirm a plateau.

`MC_KERNEL`, `SA_T_INITIAL`, `SA_T_FINAL`: MC kernel selector.
- `MC_KERNEL cuckoo` (default) uses the upstream-derived greedy +
  ε-band acceptance inside a cuckoo-search population.
- `MC_KERNEL sa` uses simulated annealing with Metropolis acceptance
  (`exp(-ΔE/T)`) inside the same outer cuckoo-search loop. The
  temperature decays exponentially from `SA_T_INITIAL` (default 2.0)
  to `SA_T_FINAL` (default 1e-3) across the `TOTALIter` outer
  iterations. Useful for benchmarking and for problems where you
  want the "standard" SA contract; on the test cases we measured
  (SQS, B2/L1₂/L1₀/D0₃/L1₁/L2₁ ordered targets, cell sizes ≤ 500)
  cuckoo and SA reach the same final fitness — neither escapes the
  L2₁ cold-start plateau, which appears to require cluster moves
  or parallel tempering.

`FAST_SWAPS`, `TARGETED_SWAPS`, `TARGETED_SWAP_PROB`: MC-loop tuning.
- `FAST_SWAPS 1` (default) uses the delta-fitness MC path: maintain
  `pairGr` as state across swaps and apply an O(z) delta per proposal
  instead of recomputing the full O(N·z) fitness. 2-18× speedup on
  loop-bound cases, identical convergence to legacy. Set 0 for the
  legacy full-recompute path (kept for A/B comparison).
- `TARGETED_SWAPS 1` biases swap proposals toward the species pair
  (i, j, shell k) with the largest residual², instead of always
  uniform-random atom pairs. Off by default: on the test cases we
  measured (binary/ternary SQS, B2/L1₂/L1₀/D0₃/L1₁ ordered targets,
  cell sizes up to n=500) uniform-random already converges fast and
  the bias adds overhead without speedup. May help on very large
  supercells or complex multi-pair SRO targets where uniform random
  struggles to find improving swaps. `TARGETED_SWAP_PROB <0.0-1.0>`
  controls the bias level (default 0.7 = 70% targeted, 30% uniform
  for exploration).

`RNG_SEED`, `RANDOM_ONLY`, `CONVERGENCE_LOG`: reproducibility and
ablation knobs.
- `RNG_SEED <n>` sets the RNG; per-rank seeds are derived as `n XOR
  (rank * phi64)` so multi-MPI runs are decorrelated yet reproducible.
  Default 0 = use `std::random_device` (non-reproducible).
- `RANDOM_ONLY 1` skips the MC loop entirely and returns the best of
  the initial random nest pool — useful for measuring how much the
  cuckoo-search MC is actually improving over pure random search.
- `CONVERGENCE_LOG <path>` writes per-iteration best-fitness on rank 0
  as a CSV (`iteration, best_fitness, wall_seconds`). Iteration `-1`
  is the pre-MC random-pool best.

See `docs/baseline_vs_scraps.sh` for a runnable comparison that pins
the same seed for RANDOM_ONLY and full SCRAPS and reports the gap.
Our own runs show: SCRAPS reduces fitness by ~12× vs random baseline
on infeasible-SQS plateaus and finds ordered targets (B2, L1_2, etc.)
that are mathematically unreachable by random search at any budget.

### Example — NiCoCr with literature CSRO pattern

```
TYP     3
GITER   1000
ELEMENT 36 36 36
Cell_DIM        3 3 3
NESTS   24
MCSTEP  750
TOTALIter 10
MAXSHELLNUM 2
WEIGHT 2.0 1.0 1.0

# Species: 0=Ni, 1=Co, 2=Cr
SRO 0 2 0 -0.10
SRO 1 2 0 -0.05
```

This produces a 108-atom FCC supercell with α(Ni-Cr) ≈ −0.10, α(Co-Cr) ≈
−0.05, α(Ni-Co) ≈ 0, and α(Cr-Cr) ≈ +0.15 (which emerges naturally from
the sum rule Σ_j c_j α_ij = 0).

## Picking Cell_DIM: feasibility and a one-shot wrapper

For SQS targets (α=0) the expected pair count `N_i · z_k · N_j / N`
must be integer for every weighted shell, or the optimizer plateaus at
a non-zero fitness floor (see `docs/mcsqs_comparison.md` for the full
explanation). Two tools help:

```bash
# Tell me the best Cell_DIM near 200 atoms for 35:35:30 ternary
tools/suggest_cell.py POSCAR --composition 0.35,0.35,0.30 \
    --target-atoms 200 --shells 2 --non-cubic

# One-shot: pick cell, write INPUT, run scraps with early stopping
tools/run_sqs.py POSCAR --composition 0.35,0.35,0.30 \
    --target-atoms 200 --shells 2 --patience 3
```

`run_sqs.py` composes `suggest_cell.py` with the new INPUT heuristics
(`AUTO_MCSTEP`, `EARLY_STOP_PATIENCE`) so that "give me an SQS at this
composition" is a single command — no manual MCSTEP or TOTALIter
tuning. SCRAPS' main loop also prints a warning at startup if the
chosen `(ELEMENT, Cell_DIM, MAXSHELLNUM)` makes any target
non-integer, naming the offending pair and the floor.

## Quick start (one-command CLI)

For first-time users, `tools/scraps_wizard.py` orchestrates the whole
workflow behind a single command:

```bash
# Single-sublattice equiatomic NiCoCr SQS, target ~100 atoms
python3 tools/scraps_wizard.py quick --template fcc \
    --composition Ni:0.33,Co:0.33,Cr:0.34 --target-atoms 100

# Rocksalt HEA-carbide (Mo,Nb,Ta,W)C
python3 tools/scraps_wizard.py quick --template rocksalt \
    --composition Mo:0.25,Nb:0.25,Ta:0.25,W:0.25 \
    --composition C:1.0

# I have my own POSCAR/CIF (e.g. from Materials Project):
python3 tools/scraps_wizard.py from-structure my_perovskite.cif

# Don't know where to start? Step-by-step prompts:
python3 tools/scraps_wizard.py interactive

# See built-in templates:
python3 tools/scraps_wizard.py list-templates
```

The wizard picks an integer-feasible cell from `--target-atoms`, sets
`AUTO_BUDGET=2` (the "thorough" MC profile — no manual tuning),
applies multi-sublattice `SUBLATTICE` blocks when needed, runs SCRAPs,
and exports the result to CIF (or VASP / XYZ via `--format`). Built-in
templates: `fcc`, `bcc`, `b2`, `rocksalt`, `heusler-l21`,
`perovskite-abo3`. For any non-templated structure, `from-structure`
calls `sublattice_detect.py` to auto-identify Wyckoff orbits.

## High-throughput sweeps (HEA composition space)

For HTP exploration — generating many SQS at once across a composition
grid — `tools/scraps_htp.py` builds on the wizard pipeline:

```bash
# Binary alloy: 11 SQS from pure Ni to pure Co
python3 tools/scraps_htp.py linear --template fcc --species Ni,Co \
    --range 0:1 --step 0.1 --target-atoms 100 --output-dir nico_sweep

# Ternary NiCoCr on the Gibbs simplex at resolution 5 (21 compositions)
python3 tools/scraps_htp.py simplex --template fcc --species Ni,Co,Cr \
    --resolution 5 --target-atoms 100 --output-dir nicocr_sweep

# Custom list from CSV (header row = species, each row = composition)
python3 tools/scraps_htp.py list --template fcc \
    --compositions my_comps.csv --output-dir custom_sweep

# Multi-sublattice perovskite, sweep A-site, fix B and O
python3 tools/scraps_htp.py simplex --template perovskite-abo3 \
    --species Ba,Sr,Ca --resolution 4 --fixed 1:Ti:1.0 --fixed 2:O:1.0
```

Each composition gets its own subdir (e.g. `Ni0.50_Co0.50/`) with the
POSCAR, INPUT, SCRAPS.vasp, sqs.cif, and a per-composition `meta.json`;
the top-level `manifest.json` summarizes everything (cell sizes,
fitness, max\|α\|, wall times). `--workers N` runs N compositions in
parallel; the runner is resumable (re-running skips compositions whose
`sqs.cif` already exists), so a long sweep can be interrupted with
Ctrl-C and resumed later.

## Tutorials

Five end-to-end Jupyter notebooks under `notebooks/`:

1. **`01_first_scraps_run.ipynb`** — shortest path from "I have an ASE
   primitive cell" to "I have a NiCoCr SQS supercell".
2. **`02_targeting_csro.ipynb`** — per-pair SRO targets reproducing a
   literature-style CSRO pattern on NiCoCr, with a random-SQS baseline
   to compare against.
3. **`03_heb_end_to_end.ipynb`** — AlB₂-type high-entropy diboride with
   the boron spectator sublattice, using `suggest_cell.py` to pick the
   cell and `verify_sro.py --spectator-species B` to validate.
4. **`04_multi_sublattice.ipynb`** — (Ni,Co)(Ti,Zr) B2 SQS + CSRO,
   plus `sublattice_detect.py` end-to-end on a Heusler L2₁ POSCAR.
5. **`05_troubleshooting.ipynb`** — common failure modes with
   diagnoses (integer-feasibility floors, multi-shell weighting,
   multi-sub uniformity checks, SCRAPs-vs-mcsqs differences).

See `notebooks/README.md` for a use-case → notebook map. Generated
programmatically from `notebooks/build_notebooks.py`; regenerate with
`python3 notebooks/build_notebooks.py` and `jupyter nbconvert
--execute --inplace notebooks/*.ipynb`.

## ASE interoperability

For users in the ASE ecosystem (`ase`-based workflows, MD codes,
DFT frontends, etc.), `tools/scraps_ase.py` is a thin wrapper that
lets you stay in ASE Atoms-land end to end:

```python
from ase.build import bulk
from scraps_ase import run_sqs_from_atoms

# Start with an ASE primitive cell (any way you like; ase.build, ase.io.read, ...)
primitive = bulk("Cu", "fcc", a=3.6, cubic=True)

# Run SCRAPS and get a SQS as an ASE Atoms back
sqs = run_sqs_from_atoms(
    primitive,
    species=["Ni", "Co", "Cr"],         # symbols matching ELEMENT order
    element_counts=[36, 36, 36],        # exact integer occupation
    cell_dim=(3, 3, 3),                 # supercell tiling
    max_shellnum=2, weights=[1, 1],
    rng_seed=42,
)
print(sqs)                              # standard ASE Atoms
sqs.write("ni_co_cr_sqs.cif")           # ASE I/O
print(sqs.info["scraps_final_fitness"]) # 0 -> exact SQS

# For ordered targets, pass sro_targets=[(i, j, shell, alpha), ...]:
b2 = run_sqs_from_atoms(
    bulk("Cu", "bcc", a=3.0, cubic=True),
    species=["Cu", "Zn"],
    element_counts=[16, 16],
    cell_dim=(2, 2, 4),
    max_shellnum=2,
    sro_targets=[(0, 1, 0, -1.0), (0, 1, 1, +1.0)],  # B2/CsCl
)
```

Two lower-level helpers are also exported:
- `read_scraps_output(path, species=[...])` -- read SCRAPS.vasp into
  an ASE Atoms with proper species relabeling (SCRAPS' output POSCAR
  lacks a species line, so you supply the list).
- `make_scraps_inputs(work_dir, atoms, element_counts, cell_dim, ...)`
  -- generate POSCAR + INPUT from an ASE Atoms primitive; useful when
  you want to launch scraps yourself (e.g. from a job script) instead
  of via the wrapper.

## Verification

`tools/verify_sro.py` computes the Warren-Cowley α matrix directly from
any `SCRAPS.vasp` using pymatgen — independent of SCRAPS's own
accounting, so it serves as a cross-check. Supports multi-shell
computation, target-file comparison with pass/fail tolerance, JSON
output for CI, and POSCAR formats including species symbol lines,
selective dynamics, and non-orthogonal cells.

```bash
pip install pymatgen
python3 tools/verify_sro.py SCRAPS.vasp --species Ni,Co,Cr
python3 tools/verify_sro.py SCRAPS.vasp --species Ni,Co,Cr --target INPUT
```

See [`tools/README.md`](tools/README.md) for the full CLI reference.

After the printtoscreen fix, the SCRAPS log and `verify_sro.py` agree
to floating-point precision.

## Tests

```bash
pytest tools/tests/                  # exercise verifier against known structures
```

Reference cells with analytically-known α matrices (B2 NiAl, L1₂ Ni₃Al)
plus SCRAPS round-trip checks.

The `tests/` directory also ships three case files (INPUT + POSCAR only
— run each in its own directory to produce `SCRAPS.vasp`):

- `tests/bcc_binary/` — binary A_27 B_27 in 3×3×3 BCC, no SRO target.
  Reproduces upstream's `test_bcc` case as a regression check.
- `tests/nicocr_random/` — equiatomic FCC NiCoCr, no SRO. Produces an
  exact SQS (every α = 0.0000).
- `tests/nicocr_csro/` — same composition with the literature-pattern
  CSRO above.
- `tests/fluorite_unsupported/` — **intentionally rejected case.**
  Fluorite-like 96-atom binary supercell with non-uniform sublattice
  coordination (cation z=8, anion z=4). Should exit with code 1 in
  ~1 second after the multi-sublattice diagnostic; verifies the
  detection in `Lattice::coordination_per_shell()` is wired correctly.
  See `tests/fluorite_unsupported/README.md`.
- `tests/heb_5metal_random/` — AlB₂-type 150-atom supercell of
  (Hf,Zr,Ti,Ta,Nb)B₂ with boron declared as the spectator sublattice.
  Optimizer finds an exact SQS on the M sublattice (α = 0 for every
  pair). Demonstrates `SPECTATOR_BASIS_SPECIES` end-to-end.
- `tests/heb_5metal_csro/` — same structure with literature-flavored
  CSRO targets (Hf-Zr clustering +0.30, Hf-Ti ordering −0.30, Ti-Ta
  ordering −0.30, Ta-Nb clustering +0.30). Optimizer achieves ±0.333
  (closest integer-count approximation at this size); unspecified
  pairs stay at α=0. Demonstrates per-pair targeting on a designed
  sublattice with a fixed spectator.
- `tests/heb_5metal_csro_1200/` — same CSRO targets at 1200 atoms
  (10×10×4 supercell). Achieves α ≈ ±0.302 — within 1% of target.
  Demonstrates precision improvement at MLIP-friendly cell sizes,
  with the required `MCSTEP=5000` and shell-focused `WEIGHT` recipe.
- `tests/heb_5metal_nonequi/` — Hf-rich (Hf₅₀Zr₃₀Ti₃₀Ta₂₀Nb₁₄)B₂ at
  432 atoms. Validates that per-sublattice c_j^M reference is computed
  correctly for non-equiatomic M-sublattice compositions; expected α
  matrix is exactly symmetric with α(Nb-X) ≈ −0.029 reflecting Nb
  being the rarest species.

## Supported lattice types

The Warren-Cowley formulation that SCRAPS implements assumes a
**uniform geometric reference** — every atom in the supercell sees
the same coordination structure in each neighbor shell, even if the
chemistry on different sites differs. Two clearly-defined regimes:

### Supported: single-Bravais-lattice-with-basis structures

Lattices where the basis may contain multiple Wyckoff positions, but
all positions share the same nearest-neighbor coordination. This
covers nearly all standard substitutional-alloy contexts:

- Pure FCC, BCC, HCP (ideal c/a)
- Ordered intermetallics with uniform coordination: **B2** (CsCl, e.g.
  NiAl), **L1₂** (Cu₃Au, e.g. Ni₃Al), **D0₃** (Fe₃Al), **B19′** in
  most regimes
- Any random or partially-ordered solid solution on FCC/BCC/HCP

For these, the Warren-Cowley α_ij = 1 − P(j|i)/c_j defined against the
global concentration c_j is the right reference, and SCRAPS computes
it correctly. The validated test cases (`tests/bcc_binary`,
`tests/nicocr_random`, `tests/nicocr_csro`, plus the pytest reference
structures B2 NiAl and L1₂ Ni₃Al) all fall in this regime.

### Supported: one designed sublattice + one or more spectator sublattices

Structures where compositional disorder is confined to a single
sublattice and other sublattices are occupied by a single fixed
species. The canonical example is the **AlB₂-type high-entropy
diboride (HEB)**, where 4–6 transition metals share the metal
sublattice and pure boron sits on a fixed honeycomb sublattice.
Also covers:

- HEB family: (Hf, Zr, Ti, Ta, Nb, Cr, Mo, W, V) on M with pure B
- Carbides and nitrides of similar structure (high-entropy MX_n)
- Substitutional alloys with structural vacancies pinned to specific
  sites
- Any structure where the user can identify which basis positions
  belong to the compositionally-disordered (designed) sublattice

Use the `SPECTATOR_BASIS_SPECIES <basis_idx> <species_idx>` INPUT
keyword to declare which basis positions are spectator (one line per
spectator basis position). For an HEB with M at POSCAR basis 0 and B
at basis positions 1, 2 (with B as species 5):

```
SPECTATOR_BASIS_SPECIES 1 5
SPECTATOR_BASIS_SPECIES 2 5
```

The optimizer pins those positions to the declared species and only
shuffles atoms on the designed (non-spectator) sublattice. Warren-
Cowley α is computed using per-sublattice reference concentrations
c_j^M = N_j^M / N^M (M = non-spectator atoms) and counting only
M-M neighbor pairs. Output α matrix omits spectator-species rows and
columns. See `tests/heb_5metal_random/` and `tests/heb_5metal_csro/`.

A practical note on shell numbering: in HEB-type lattices, the global
shell ordering puts B-B and M-B distances before the first M-M shell.
Set `MAXSHELLNUM` large enough to reach the M-M shells (≥ 8–10 for
AlB₂-type with typical c/a ≈ 1.1), or your fitness function won't
include them.

#### MLIP-friendly cells: WEIGHT and MCSTEP scaling

When you go beyond a few hundred atoms to MLIP-friendly cell sizes
(≥ ~1000), two parameters need attention:

1. **MCSTEP and GITER should scale roughly with N.** The Cuckoo
   Search inner loop attempts a fixed number of single-site swaps
   regardless of cell size, so at large N the per-atom swap budget
   drops and the optimizer leaves you near random init. Empirical
   guideline: `MCSTEP = GITER ≈ 10 × N_metal_atoms`. For a 400-M-atom
   HEB, use ~5000 each instead of the default 500.

2. **WEIGHT must focus on the targeted shell when multiple M-M
   shells exist.** In a small HEB cell (e.g. 5×5×2 = 50 M atoms with
   cutoff = c = 1.1) only one M-M shell fits inside the cutoff. In
   larger cells (10×10×4 = 400 M atoms with cutoff = 4.4) several
   M-M shells fit. With `WEIGHT 1 1 1 …` the optimizer puts equal
   pressure on every shell, including the untargeted ones, which
   pulls back toward SQS. Set `WEIGHT 0 0 … 1 … 0 0` with `1` only
   at the index of your targeted shell. See
   `tests/heb_5metal_csro_1200/README.md` for the concrete recipe.

With these two tuned, 1200-atom HEB CSRO targeting hits α to ±0.002
in ~25 seconds on 4 ranks.

### Supported: multi-sublattice structures with disjoint species per sublattice

**New in v0.4.0.** Structures with two or more chemically-disordered
sublattices where each species belongs to exactly one sublattice:

- **Rocksalt HEA-carbides / nitrides**: 4–6 transition metals on the
  metal sublattice, C or N on the anion sublattice.
- **Full Heusler (L2₁) with independent X, Y, Z mixing**: e.g.
  designed quaternary Heuslers with chemical disorder on multiple
  Wyckoff orbits simultaneously.
- **Double perovskites / spinels** with cation site mixing on two
  Wyckoff positions.

Use the `SUBLATTICE` INPUT keyword:

```
SUBLATTICE <id> BASIS <b1> <b2> ... SPECIES <s1> <s2> ...
```

Each block declares which basis positions and which global species
indices belong to that sublattice. Species and basis positions must
be disjoint across sublattices; with disjoint species,
`pairGr[i][j][shell]` and `targetSROpair[i][j][shell]` naturally
distinguish intra- and cross-sublattice pairs without a new
dimension. Cross-sublattice pair counts are determined by composition
and the optimizer leaves them alone; intra-sublattice pair counts are
optimized via swap moves gated to stay within each sublattice.

For automatic Wyckoff-orbit detection from a POSCAR (the standard
crystallographic partition), run:

```bash
python3 tools/sublattice_detect.py POSCAR --write INPUT
```

This appends draft `SUBLATTICE` blocks for each Wyckoff orbit with
`<fill in>` SPECIES placeholders — you then edit them to assign
species. See `tools/sublattice_detect.py --help` for `--symprec`
tuning and JSON output.

### Supported: lattices with non-uniform coordination via per-Wyckoff SUBLATTICE (v0.4.2+)

Lattices where different Wyckoff orbits have physically different
coordination numbers (fluorite cation z=8 vs anion z=4; sigma phase
5 orbits with z=12-15; Laves phases; MAX phases) work as long as
each Wyckoff orbit gets its own `SUBLATTICE` block. The uniformity
check applies *within* each sublattice rather than across all atoms,
and the per-sublattice fitness uses the right per-(σ, τ, k)
coordination tensor.

```bash
python3 tools/sublattice_detect.py POSCAR --write INPUT  # auto-detects orbits
```

Without `SUBLATTICE` blocks, the same POSCARs still abort with the
legacy non-uniform-coordination message (the optimizer assumes
single-sublattice uniform z) — see `tests/fluorite_unsupported/`.

For these lattice types, use ATAT's `mcsqs`
(<https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/>),
which supports sublattice-resolved SQS construction natively.

## Credits

Original SCRAPS / Hybrid-Cuckoo-Search code by Rahul Singh, Aayush Sharma,
Prashant Singh, Ganesh Balasubramanian, and Duane D. Johnson (Iowa State
University / Ames National Laboratory). Please cite their paper if you use
this code:

> R. Singh, A. Sharma, P. Singh, G. Balasubramanian, D. D. Johnson,
> *Accelerating computational modeling and design of high-entropy alloys*,
> Nature Computational Science **1**, 54–61 (2021).
> [doi:10.1038/s43588-020-00006-7](https://doi.org/10.1038/s43588-020-00006-7)

Extension work and bug fixes by Raymundo Arroyave (Texas A&M).

## License

MIT (preserved from upstream). See `SCRAPs/ReadMe.md` and individual
source-file headers for the original Iowa State copyright notice.
