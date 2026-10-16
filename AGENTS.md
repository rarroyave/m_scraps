# AGENTS.md — orientation for AI coding agents

This file is the working contract for any AI agent (Claude Code,
Codex, Copilot, etc.) operating on the `scraps-perpair` repository.
It is not a duplicate of `README.md` or `CONTRIBUTING.md` — read
those for the human-facing project description and contributor
workflow. This file covers the *non-obvious* conventions an agent
will get wrong by default.

For deeper background:

- `README.md` — what the project is and how to use it
- `CONTRIBUTING.md` — branching, commit/PR conventions, coding style
- `docs/OVERVIEW.md` — technical architecture map
- `paper/paper.md` — peer-review-shaped summary of contributions

---

## TL;DR

`scraps-perpair` is a C++/Python tool that generates multicomponent
alloy and ceramic supercells with **user-specified per-pair, per-shell
Warren-Cowley short-range-order (SRO) parameters**, with multi-
sublattice support. The C++ core (`SCRAPs/`) is an MPI-parallel
optimizer; the Python layer (`tools/`) wraps it for typical end-user
workflows (cell-size selection, verification, ASE/CIF round-tripping,
high-throughput composition sweeps).

The single most important fact for an agent: **MPI semantics matter
everywhere**. Tests, examples, and user workflows assume `mpirun`
is matched to the `mpic++` that built `SCRAPs/scraps`. Mixing them
(common on macOS: anaconda's MPICH `mpirun` ahead of Homebrew Open
MPI on PATH) puts every rank into silent **singleton mode** where
each "rank" runs an independent search and only one survives. The
helper `find_mpirun()` exists in many tests for exactly this reason
— copy it; don't roll your own resolution.

The canonical checkout is `~/src/tools/scraps-perpair`; do not develop in the
legacy `~/codes/scraps-perpair` rollback copy. Run production calculations on
the Mac Studio and keep generated cells, logs, sweeps, and checkpoints under
`~/runs/scraps-perpair`, not inside the repository.

---

## Build, test, run

Single source of truth:

```bash
./install.sh                        # MPI build + locked Python environment
./scripts/test.sh -q                # 183 collected; 4 generated-fixture tests skip on a clean clone
python -m pytest tools/tests/test_<file>.py -x -q   # single file, faster
```

Build deps: `mpic++` (Open MPI on macOS, Open MPI or MPICH on Linux),
Eigen, C++17. Python deps in `tools/requirements.txt`
(`pymatgen`, `ase`, `matplotlib`, `pytest`). CI runs the matrix
{Ubuntu, macOS} × {Python 3.11, 3.12} on every push and PR — see
`.github/workflows/test.yml`.

**Tests that need scraps or mpirun gate themselves:**

```python
SCRAPS_BIN = REPO / "SCRAPs" / "scraps"
pytestmark = pytest.mark.skipif(
    not SCRAPS_BIN.exists() or find_mpirun() is None,
    reason="scraps binary not built or mpirun missing",
)
```

Copy this pattern in any new C++-touching test. A fresh clone with
no build must still pass pytest (skips are fine; failures are not).

**Reproducibility:** any test that depends on the optimizer's exact
output (rather than just `fitness ≤ threshold`) must pin `RNG_SEED`
in INPUT and reference the seed in a comment near the assertion.

---

## Running the tool to fulfill a user request

If a user asks you to *generate* a supercell (not modify the codebase),
work through these decisions in order. **Most user requests are best
served by the Python API or `scraps_wizard`, not by hand-writing INPUT.**

### Entry-point decision tree

| User intent | Use |
| --- | --- |
| "Generate an SQS / SRO supercell for <composition> on <structure>" — programmatic, one cell | `tools/scraps_ase.py` `run_sqs_from_atoms(...)` (Python API) |
| Same, but from the shell | `tools/run_sqs.py <primitive.poscar> --composition ... --target-atoms ...` |
| User is a novice / unsure what cell size, AUTO_BUDGET, or template fits | `python3 tools/scraps_wizard.py quick --template <fcc|bcc|b2|rocksalt|heusler-l21|perovskite-abo3> --target-atoms <N>` (or `interactive`) |
| User gave their own CIF/POSCAR rather than a stock template | `python3 tools/scraps_wizard.py from-structure <file>` |
| Sweep over many compositions on a grid or simplex | `tools/scraps_htp.py` (parallel, resumable, JSON manifest output) |
| Multi-sublattice structure (carbide, Heusler, perovskite, ...) and not sure of the orbits | `tools/sublattice_detect.py <structure>` first; it auto-emits draft `SUBLATTICE` blocks |
| Is the requested target even achievable on cell size N? | `tools/suggest_cell.py` (computes integer-feasibility for the target α pattern) |
| User explicitly wants to drive `scraps` directly with a hand-written INPUT | `mpirun -np 4 SCRAPs/scraps` in a dir with `POSCAR` + `INPUT` (last resort; use `find_mpirun` to pick the right `mpirun`) |

### Canonical Python-API example (the one most user requests collapse to)

```python
from ase.build import bulk
from scraps_ase import run_sqs_from_atoms

primitive = bulk("Cu", "fcc", a=3.56, cubic=True)   # 4-atom FCC basis
sqs = run_sqs_from_atoms(
    primitive,
    species=["Ni", "Co", "Cr"],          # species labels for the OUTPUT
    element_counts=[36, 36, 36],         # integer counts; must sum to len(atoms) * prod(cell_dim)
    cell_dim=(3, 3, 3),                  # 4 * 3^3 = 108 atoms total
    max_shellnum=2, weights=[1.0, 1.0],  # shells the fitness considers
    sro_targets=[(2, 2, 0, -0.10)],      # alpha(Cr-Cr) at 1NN = -0.10 (literature CSRO)
    auto_budget=2,                       # 1=basic, 2=thorough, 3=exhaustive
    rng_seed=42,                         # pin for reproducibility
    ranks=4,                             # MPI ranks
    work_dir="my_run",                   # outputs land here
)
print("fitness:", sqs.info["scraps_final_fitness"])
print("log:", sqs.info["scraps_log"])     # full scraps stdout
sqs.write("output.cif")                   # ASE writes CIF, POSCAR, etc.
```

What the call produces in `work_dir`:

| File | Meaning |
| --- | --- |
| `POSCAR` | The primitive cell SCRAPs read in. |
| `INPUT` | The keyword file SCRAPs was driven by. Read this to confirm what the run actually targeted. |
| `SCRAPS.vasp` | The optimized supercell in SCRAPs-native POSCAR dialect (count blocks, no species labels). |
| `scraps.log` | Full stdout including the SRO expansion banner (v0.4.12+), per-iteration fitness, and the final α-matrix printout. |
| `scraps` | The binary copied in for the run. |

The returned `Atoms` carries `info["scraps_final_fitness"]`,
`info["scraps_work_dir"]`, `info["scraps_log"]` — use these in
follow-up assertions instead of re-parsing the log.

### Species ordering matters

The `species=[...]` list maps positionally onto `element_counts=[...]`
**and** onto the species indices in `sro_targets=[(i, j, shell, α)]`.
In the example above:
- index 0 = Ni
- index 1 = Co
- index 2 = Cr

So `(2, 2, 0, -0.10)` means α(Cr-Cr) at shell 0 (1NN). Get this
wrong and the optimizer chases a pattern on the wrong species pair.
The wizard / CLI hide this by accepting `"Ni:0.333,Co:0.333,Cr:0.333"`
syntax; the Python API does not.

### Minimal hand-written INPUT (for the direct-`scraps` path)

```
TYP     3              # number of species
GITER   1000           # global iteration cap (cuckoo "generations")
ELEMENT 36 36 36       # per-species atom counts (must sum to total atoms)
Cell_DIM 3 3 3         # supercell tiling of the POSCAR primitive
NESTS   24             # cuckoo population size
MCSTEP  100            # MC swaps per nest per iteration
TOTALIter 30           # outer iterations (cuckoo escape rounds)
MAXSHELLNUM 2          # shells included in the fitness
WEIGHT 1 1             # per-shell weights (length == MAXSHELLNUM)
EARLY_STOP_PATIENCE 5  # halt after 5 stagnant outer iters
RNG_SEED 42            # required if you want reproducibility

# Per-pair, per-shell targets (omit for vanilla SQS):
SRO 2 2 0 -0.10        # alpha(Cr-Cr) at 1NN = -0.10 (literature CSRO)
```

If `AUTO_BUDGET 1` (or 2 / 3) is set, you may omit `NESTS`, `MCSTEP`,
`TOTALIter`, `EARLY_STOP_PATIENCE` — the C++ side fills them from the
profile once it knows the supercell size. This is the preferred
modern style.

### Verifying outputs

Treat SCRAPs's on-screen α printout as informational; treat
`tools/verify_sro.py` as ground truth:

```bash
python3 tools/verify_sro.py work_dir/SCRAPS.vasp \
    --species Ni,Co,Cr --shells 2 --json
```

JSON form returns `{"shells": [{"index": k, "alpha_matrix": [[..]], ...}, ...]}`;
species are in the order you passed via `--species`. Use this when
you need to report "the run hit α(Cr-Cr) = X" back to the user with
confidence.

### Compute-budget guidance

The first knob to reach for is `AUTO_BUDGET` (1, 2, 3 = basic,
thorough, exhaustive). Heuristics that work in practice:

- ≤ 128 atoms, vanilla SQS: `auto_budget=1` is enough.
- ≤ 256 atoms with literature CSRO targets: `auto_budget=2`.
- ≥ 500 atoms, or steep multi-pair targets, or multi-sublattice
  with two simultaneously disordered sublattices: `auto_budget=3`.

Wall-clock scales roughly linearly in cell size and quadratically
in shell count. A 108-atom NiCoCr CSRO run on 4 ranks finishes in
under 2 s; a 500-atom run in under 5 s. If a run is taking longer
than a minute on ≤ 500 atoms, something is wrong — check the log
for the singleton-MPI warning, for INFEASIBLE messages, or for
WEIGHT/MAXSHELLNUM mismatches.

### "Is this cell size sensible?" — check before running

Integer pair counts have to land for the optimizer to hit α
targets exactly. `suggest_cell.py` ranks candidate `(cell_dim,
ELEMENT)` tuples by their α=0 (SQS) feasibility deviation:

```bash
python3 tools/suggest_cell.py <primitive.poscar> \
    --composition Ni:0.333,Co:0.333,Cr:0.333 \
    --target-atoms 108
```

It prints the top-N cell choices with their integer-rounded
ELEMENT counts and per-pair feasibility deviation; pick from the
top of that list. The same heuristic is invoked automatically by
`scraps_wizard quick` and `scraps_htp`. For steep CSRO targets
beyond pure SQS, generate a candidate run and read the achieved-α
back via `verify_sro.py` — there is no closed-form feasibility
checker for arbitrary per-pair α patterns yet.

### Multi-sublattice example (rocksalt HEA carbide)

For structures with two or more chemically-disordered Wyckoff
orbits, every disordered orbit gets its own `SUBLATTICE` block.
For the legacy Warren–Cowley pair objective, species and basis positions must
be **disjoint** across blocks. Figure mode supports shared species with
explicit local counts: `(id, basis_positions, species_indices, counts)` for
every group, serialized as `SUBLATTICE ... SPECIES ... COUNTS ...`. Basis
positions remain disjoint; local budgets plus spectators must sum to ELEMENT.
Python-API form:

```python
from ase.build import bulk
from scraps_ase import run_sqs_from_atoms

# Rocksalt primitive: 2 basis sites (cation at 0,0,0; anion at 0.5,0.5,0.5).
prim = bulk("MgO", "rocksalt", a=4.5, cubic=True)   # 8-atom cubic conv. cell

# 5-metal cation HEA + pure-C anion sublattice (e.g. MoNbTaVW-C).
# Species ordering: 0-4 = metals (on basis position 0), 5 = C (on basis position 1).
sqs = run_sqs_from_atoms(
    prim,
    species=["Mo", "Nb", "Ta", "V", "W", "C"],
    element_counts=[26, 26, 26, 26, 24, 128],   # sums to 8 * 4^3 = 256
    cell_dim=(4, 4, 4),
    max_shellnum=2, weights=[1.0, 1.0],
    sublattices=[
        # (sublattice_id, basis_positions, species_indices)
        (0, [0], [0, 1, 2, 3, 4]),    # metal sublattice (5 species)
        (1, [1], [5]),                # carbon sublattice (pure)
    ],
    sro_targets=[(0, 4, 0, -0.05)],   # mild Mo-W avoidance on the metal site
    auto_budget=3, rng_seed=42, ranks=4,
    work_dir="rocksalt_hea",
)
```

If unsure of the orbits for a structure, run
`python3 tools/sublattice_detect.py <cif_or_poscar>` first — it
emits draft `SUBLATTICE` blocks via spglib's Wyckoff analysis and
the user (or you) fills in the species. Two basis positions in the
same Wyckoff orbit *must* be in the same `SUBLATTICE` block.

For HEB-style structures where one sublattice is a chemically-pure
structural scaffold (AlB₂ borides, fluorite oxides with single anion
species), the simpler `spectator_basis_species=[(basis_position,
species_index)]` kwarg fixes those basis positions to a single
species without needing a full `SUBLATTICE` declaration on the
designed sublattice.

### Canonical worked examples to copy from

For every classical structure class, `tests/<case>/` holds a
tracked `INPUT` + `POSCAR` pair you can copy as a starting point.
The directory naming is structure-class prefixed:

```bash
ls tests/      # b2_bcc_binary, d03_bcc_a3b, l10_fcc_ab, l11_fcc_ab,
               # l12_fcc_a3b, l21_bcc_x2yz, heb_5metal_csro,
               # nicocr_csro, stoich_fcc_*, ...
```

When a user asks for "a working INPUT for L1₂ Cu₃Au-type on FCC,"
the answer is `tests/l12_fcc_a3b/INPUT` — copy it, adjust species
and `Cell_DIM`, and run. Cases not tracked here (Laves C15,
fluorite, σ-FeCr, spinel, pyrochlore) are exercised programmatically
in `tools/tests/test_non_uniform_coordination.py`, which is the
place to look for the corresponding `sublattices=[...]` argument
shapes.

### Reading `scraps.log` — what success and failure look like

The log carries everything you need to judge a run. Search for these
landmarks in order:

| Pattern | What it means |
| --- | --- |
| `WARNING: SCRAPS is running in SINGLETON MODE` | mpirun/mpic++ mismatch; every "rank" is independent. Output is one rank's result; fix `mpirun` resolution and re-run. |
| `WARNING: ... weighted target pair count(s) are non-integer` | Cell size cannot reach the requested α exactly. Minimum achievable fitness is printed. Use `suggest_cell.py` for a feasible cell. |
| `[SRO] expanding same-species SRO ...` | The v0.4.12+ banner showing how `SRO i i shell α` lines were expanded into cross-species cells. Confirm the β values are what you expect. |
| `(skipped: explicit cross-species override present, alpha=...)` | An explicit cross-species line is overriding the symmetric split — v0.4.13+. The applied value follows `alpha=`. |
| `Warning: requested alpha(i,i) ... implies alpha(i,i) = ...` | v0.4.13 sum-rule mismatch: explicit cross-species overrides break the user-stated same-species target. The optimizer will chase the cross-species values, not the same-species claim. |
| `Iter N: best fitness = X` (or similar trajectory lines) | Convergence trajectory. Plot via `tools/plot_convergence.py`. A long flat tail at high fitness means stuck on a plateau. |
| `FINAL fitness (best across all N merged nests) = X` | The reported optimum across MPI ranks. This is the value SCRAPs claims it reached. |

A clean successful run shows: no `WARNING:` lines, expansion banner
matches expectations, fitness drops monotonically across iterations,
and `FINAL fitness` is near the relevant scale (see next).

### Fitness scale — what counts as "hit"

The fitness sums squared pair-count residuals over weighted shells.
For a 108-atom NiCoCr run with `max_shellnum=2, weights=[1,1]`:

| `FINAL fitness` | Interpretation |
| --- | --- |
| `≈ 0` (< 1) | Exact SQS / CSRO hit. α deviations from target < 0.005. |
| `~10` | Sub-α=0.02 deviation per pair on average — typical "good" finish for non-integer-feasible cells. |
| `~100` | Optimizer plateaued; α deviations ~0.05-0.10. Try `auto_budget=3` or a different `RNG_SEED`. |
| `>= sqrt(fit2_floor)` printed near a non-integer warning | The cell can't reach lower; choose a different cell size. |

Don't try to compare fitness across runs with different
`(MAXSHELLNUM, WEIGHT, cell size)` — the scale changes with all
three. Compare achieved α from `verify_sro.py` instead.

### Round-tripping an existing run

If a user hands you a `SCRAPS.vasp` and asks "what's in this?" or
"convert this to CIF":

```python
from scraps_ase import read_scraps_output

atoms = read_scraps_output("SCRAPS.vasp", species=["Ni", "Co", "Cr"])
atoms.write("output.cif")        # any ASE-supported format
print(atoms.get_chemical_formula())
```

Then check the SRO pattern:

```bash
python3 tools/verify_sro.py SCRAPS.vasp --species Ni,Co,Cr --shells 2 --json
```

If the user wants the achieved α written as an `--target` comparison
against an INPUT file:

```bash
python3 tools/verify_sro.py SCRAPS.vasp --species Ni,Co,Cr \
    --target INPUT --tolerance 0.02
```

Exit code 0 = pass, 1 = some pair exceeded tolerance — useful for
CI-style regression sweeps.

### Common pitfalls

These are the four fatal-or-silent errors that an agent will
plausibly hit on a first attempt without obvious diagnostics:

- **`WEIGHT` length mismatch.** `MAXSHELLNUM 2` with `WEIGHT 1 1 1`
  (three weights) aborts via `MPI_Abort` with a clear message.
  Always: `len(weights) == max_shellnum`.
- **`ELEMENT` counts don't sum.** The per-species counts must sum
  to `len(atoms) * prod(cell_dim)`. The Python API computes this
  via `len(atoms) * cell_dim[0] * cell_dim[1] * cell_dim[2]`;
  a hand-written INPUT with an off-by-one sum aborts at startup.
  When in doubt, use Hamilton rounding (see `run_cocrni_csro.py`
  for the pattern).
- **Forgetting `species=[...]` in `read_scraps_output`.** SCRAPS's
  POSCAR dialect omits species labels (only count blocks). Without
  the `species` argument, ASE reads a structure of `"X"` placeholder
  atoms — silently wrong, no error. Always pass the species list
  in the same order as `ELEMENT`.
- **`mpirun -np 1` for "quick testing."** Cuckoo Search needs ≥ 2
  nests for the cross-nest abandonment mechanism that escapes local
  minima; 1 rank with the default `NESTS` setting still works
  (multiple nests per rank), but the per-rank parallelism that makes
  scraps fast is gone. Use ≥ 4 ranks for production runs and ≥ 2
  for sanity checks; reserve `-np 1` for parser-only smoke tests.

---

## Layout

```
SCRAPs/             C++ source (main.cpp, readInput.cpp, Lattice.cpp)
                    + scraps binary (gitignored)
tools/              Python wrappers, CLIs, and pytest suite
  tests/            pytest suite (183 tests; new tests go here)
  requirements.txt  pip deps for tools + tests
docs/               Technical docs, mcsqs comparisons, research cases
  research_cases/   End-to-end DFT-ready case studies (CoCrNi CSRO is #1)
notebooks/          Jupyter-executed tutorials
paper/              JOSS submission (paper.md is source; paper.pdf gitignored)
tests/              Integration test cases (worked structures: B2, L1_2,
                    L1_0, D0_3, L1_1, L2_1, fluorite, Laves, ...)
build.sh            Build script — edit for non-Homebrew MPI/Eigen paths
```

**No `CHANGELOG.md`.** The change history lives in `git log` and
the `vX.Y.Z` annotated-tag set. Run `git log --oneline -20` to see
recent commits; major user-facing features cite their introducing
version in commit messages and in this file (e.g. v0.4.5 AUTO_BUDGET,
v0.4.11 simplex equiatomic-warning, v0.4.12 same-species SRO
auto-expansion, v0.4.13 explicit cross-species overrides). When
an agent needs to know "when did X behavior change," `git log -S
<keyword>` is faster than guessing.

---

## Conventions that aren't optional

### Commit + tag style

Look at `git log --oneline -10` for the pattern. Standard form:

```
v0.4.13: explicit cross-species SRO overrides + sum-rule mismatch warnings

<wrapped paragraph explaining the why and the API surface change>

- bullet for each material sub-change
- ...

Co-Authored-By: <tool-name> <noreply@example.com>
```

Versioned releases live as **annotated tags** (`git tag -a vX.Y.Z`),
pushed *separately* from the commit:

```bash
git push origin main
git push origin vX.Y.Z
```

**Never push, tag, or release without explicit user permission.**
This is a one-author repository with no protected-branch enforcement;
the user is the gatekeeper, not the CI. "I've made the change locally"
is the correct stopping point on first ask.

### New INPUT keywords

Every new SCRAPs INPUT keyword requires *four* coordinated changes
or the parser/docs go out of sync:

1. Parser code in `SCRAPs/readInput.cpp` (the keyword's `if(key.compare(...))`
   branch).
2. **Add the keyword to the `KNOWN_KEYS` set** in `readInput.cpp` —
   otherwise valid INPUT files produce a spurious
   `Warning: unrecognized INPUT keyword` line.
3. A pytest in `tools/tests/test_input_parser.py` (or a dedicated
   `test_<feature>.py`) covering happy path and at least one error case.
4. Documentation: `README.md` (INPUT-format section) AND
   `docs/OVERVIEW.md`. The paper (`paper/paper.md`) only gets
   updated for user-facing contributions worth describing in the
   summary or contributions section.

### SRO target conventions (v0.4.12+)

- `SRO i j shell alpha` with `i != j` sets the cross-species target
  directly.
- `SRO i i shell alpha` (same-species) **auto-expands** via the
  Warren-Cowley sum rule into N−1 cross-species lines using the
  symmetric split β = −c_i · α_ii / (1 − c_i). This is one of
  infinitely many valid splits.
- **As of v0.4.13**: explicit cross-species lines override the
  symmetric split *regardless of input order*, and a warning fires
  if the implied α_ii from the final cross-species pattern
  disagrees with a user-stated same-species value.

Do not "fix" the fitness function to sum over `i == j` pairs. The
fitness deliberately sums over independent degrees of freedom only;
the diagonal is determined by the sum rule and would over-constrain
if added. The auto-expansion is the correct place to handle
same-species targets.

### Coding style

- C++17, 4-space indent, no tabs. Mix of camelCase/snake_case —
  match the surrounding file.
- User-input validation aborts via `MPI_Abort` + clear `cout`
  diagnostic naming the offending field. See `poscar_abort` and
  `input_abort` in `readInput.cpp`.
- New comments only when *why* is non-obvious. Don't narrate the
  code; don't reference issue numbers or PRs that will rot.
- Python: type hints encouraged. CLI tools follow the
  `parse_args() -> Namespace` pattern of `tools/run_sqs.py`,
  `tools/verify_sro.py`, `tools/suggest_cell.py`.
- Tests: pytest, `tmp_path` for scratch. Never write into the
  repo tree from a test. Don't mock `pymatgen` / `ase` / MPI —
  these are real-system dependencies and mocks here have masked
  bugs historically (real-DB-not-mocks principle).

### `verify_sro.py` as ground truth

When asserting on optimizer output, prefer running the independent
`tools/verify_sro.py` (pymatgen-driven) on the resulting `SCRAPS.vasp`
rather than reading SCRAPs's own `printtoscreen()` output. The
verifier is intentionally independent of SCRAPs's internal accounting
and is the harness for cross-checking SRO patterns.

```python
import json, subprocess, sys
proc = subprocess.run(
    [sys.executable, "tools/verify_sro.py", str(work_dir / "SCRAPS.vasp"),
     "--species", "Ni,Co,Cr", "--shells", "2", "--json"],
    capture_output=True, text=True, timeout=120,
)
data = json.loads(proc.stdout)   # {"shells": [{"index": int, "alpha_matrix": [[..]], ...}, ...]}
```

---

## Things an agent should NOT do without being asked

- **No new markdown docs.** Don't create `IMPLEMENTATION.md`,
  `DESIGN.md`, plan files, or status reports as a side effect of
  doing a coding task. Edit existing docs where the audience
  expects them (README, OVERVIEW, CONTRIBUTING, AGENTS,
  research_cases/).
- **No refactors outside the task scope.** A bug fix or feature
  addition does not include "while I'm here" cleanups, renamings,
  or comment removals. The user reviews diffs line-by-line.
- **No defensive code for impossible paths.** Don't validate
  internal-only function arguments; don't add fallback branches
  for conditions that can't occur. Validate at I/O boundaries
  (INPUT, POSCAR, CLI flags) only.
- **No deletion of unfamiliar files.** Investigate before deleting.
- **No `--no-verify`, `--no-gpg-sign`, `--amend` on published
  commits, or force-pushes.** Create a new commit instead.
- **No pushing without explicit permission.** "commit" means
  commit locally; the user says "push" when they want it on origin.

---

## What gets regenerated vs tracked

`.gitignore` is the authoritative answer; key reminders:

- `SCRAPs/scraps` (build artifact) — gitignored.
- `paper/paper.pdf` — gitignored; rebuild from `paper/paper.md`
  with whatever pandoc/JOSS workflow you prefer.
- `docs/research_cases/results/<run>/{POSCAR,INPUT,SCRAPS.vasp,scraps.log,sqs.cif}`
  — gitignored. The case-study workflow regenerates them. Only
  the summary JSON and the matplotlib figure that the markdown
  embeds are tracked.
- `notebooks/nb*/`, `notebooks/*.cif`, `notebooks/POSCAR_*` —
  notebook execution artifacts, gitignored.
- `tests/<case>/{scraps,scraps.log,SCRAPS.vasp}` — per-case run
  outputs, gitignored. INPUT + POSCAR are tracked.

If your work touches any of the gitignored outputs, regenerate
them locally to verify but don't `git add -f` them.

---

## Quick sanity checklist before saying "done"

1. `bash build.sh` succeeds clean.
2. `python -m pytest tools/tests -q` is fully green (no new failures
   or unexpected skips).
3. If you added an INPUT keyword: `KNOWN_KEYS` updated, parser test
   added, `README.md` + `docs/OVERVIEW.md` reflect it.
4. If you touched the JOSS paper or research cases: regenerated
   any figures the markdown embeds.
5. Diff is scoped to the requested change. No drive-by edits.
6. Commit message follows the in-repo style; co-author trailer
   present if AI-assisted.
7. **Did NOT push or tag without explicit permission.**
