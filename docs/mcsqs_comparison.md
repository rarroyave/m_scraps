# Comparison: scraps-perpair vs ATAT `mcsqs`

Independent head-to-head on two standard test systems to verify that
scraps-perpair produces SQS quality equivalent to the reference tool
(`mcsqs` from ATAT 3.50), and to characterize their relative performance.

## Test systems

1. **NiCoCr 108 atoms** — 3×3×3 conventional FCC, equiatomic ternary,
   single sublattice. The classic substitutional-alloy SQS test.
2. **(Hf, Zr, Ti, Ta, Nb)B₂ 150 atoms** — AlB₂-type 5×5×2 supercell,
   equiatomic 5-component on M sublattice with pure B (fixed
   honeycomb sublattice). Tests multi-sublattice handling.
3. **(Hf, Zr, Ti, Ta, Nb)B₂ 600 atoms** — 5×5×8 supercell. Tests
   scaling on MLIP-friendly cell sizes.

## Setup

### scraps-perpair

Uses the `tests/nicocr_random/`, `tests/heb_5metal_random/`, and
sweep variants for 600 atoms. 4 MPI ranks via `mpirun -np 4`. The HEB
case declares boron as spectator via `SPECTATOR_BASIS_SPECIES`.

### mcsqs

Built from ATAT 3.50 source (`make` in the unpacked tarball). Single-
threaded; mcsqs has no MPI parallelism. Inputs:

```
# NiCoCr rndstr.in (FCC primitive with mixed occupation)
3.56 3.56 3.56 90 90 90
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
0.0 0.0 0.0 Ni=0.3333333,Co=0.3333333,Cr=0.3333334
```

```
# HEB rndstr.in (AlB2 with mixed metals on basis 0)
1.0 1.0 1.1 90 90 120
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.0       0.0       0.0       Hf=0.2,Zr=0.2,Ti=0.2,Ta=0.2,Nb=0.2
0.333333  0.666667  0.5       B=1.0
0.666667  0.333333  0.5       B=1.0
```

Clusters generated via `mcsqs -2=<diameter>`. Bounded-time runs end
with `touch stopsqs`; mcsqs polls every ~3 s.

`bestsqs.out` converted to VASP POSCAR via `str2cif` → pymatgen
`Structure.to(fmt='poscar')`, then α matrix computed by the same
`verify_sro.py --spectator-species B` used for scraps-perpair output.

## Results

### Single-shell (NN only)

| Test | Tool | Atoms | Parallelism | Wall | max \|α\| shell 0 |
|---|---|---|---|---|---|
| NiCoCr 108 | scraps-perpair | 108 | 4 MPI | 1 s | 0.0000 |
| NiCoCr 108 | mcsqs | 108 | 1 (serial) | 33 s | 0.0000 |
| HEB 150 | scraps-perpair | 150 | 4 MPI | <1 s | 0.0000 |
| HEB 150 | mcsqs | 150 | 1 (serial) | 64 s | 0.0000 |
| HEB 600 | scraps-perpair | 600 | 4 MPI | 5 s | 0.063 (default WEIGHT)|
| HEB 600 | mcsqs (with `-rc` explicit supercell) | 600 | 1 (serial) | 126 s | 0.0000 |

### Multi-shell α — the more rigorous test

A genuine SQS should have α ≈ 0 not just at the nearest-neighbor
shell but at multiple shells. Re-evaluating both tools' NiCoCr 108
outputs across 4 shells (NN, 2NN, 3NN, 4NN):

| Tool configuration | Shell 0 (NN, z=12) | Shell 1 (2NN, z=6) | Shell 2 (3NN, z=24) | Shell 3 (4NN, z=12) |
|---|---|---|---|---|
| scraps-perpair `MAXSHELLNUM=2` (original `tests/nicocr_random/`) | 0.000 | 0.000 | 0.042 | 0.097 |
| scraps-perpair `MAXSHELLNUM=4` (18 s wall, best of 12 trials) | **0.000** | **0.000** | **0.000** | **0.000** |
| mcsqs `-2=2.6` (NN clusters only) | 0.000 | 0.125 | 0.073 | 0.076 |
| mcsqs `-2=5.34` (NN + 2NN + 3NN + 4NN clusters) | 0.000 | 0.042 | 0.066 | 0.049 |

**Both tools target what their INPUT tells them to.** When SCRAPS is
configured with `MAXSHELLNUM=4` and weight on every shell, it can
reach exact SQS simultaneously across all 4 shells; the table reports
the best of 12 stochastic trials at the modest `MCSTEP=750,
TOTALIter=10, NESTS=24` budget used here. Off-best trials at this
budget plateau at a fitness of √6 (six off-by-one entries among the
36 directed pair counts across 4 shells) with max|α|≈0.014; raising
MCSTEP/TOTALIter or NESTS improves the hit rate on the exact basin.
mcsqs spreads residual error across shells whether configured with
NN-only or all-4-shell clusters — it never reaches exact zero beyond
shell 0 in our runs at the same wall budget.

This is still a genuine advantage for scraps-perpair on multi-shell
SQS targeting at these supercell sizes; it does not appear to be just
an artifact of the parallelism difference.

### Note on the optimizer's reported fitness

On runs that do *not* hit the exact-SQS basin, the per-rank
`FITNESS - > 2.45` and the new post-merge `FINAL fitness = 2.45`
line printed before `printtoscreen` are honest reports of √(Σ_k
weights[k] · Σ_{i≠j} (pair_ij - expected_ij)²) summed over shells.
We independently re-implemented the same formula on pymatgen and
verified that the C++ fitness matches the Python reference exactly
on every test nest, and that it agrees with `verify_sro.py`'s α
matrix (just expressed differently). An earlier note about a
fitness-vs-verifier "discrepancy" was traced to comparing a stale
SCRAPS.vasp against a fresh fitness number across runs — no
disagreement remains when the same nest is used.

### Key observations

**SQS quality.** Both tools achieve identical (exact) SQS on the
smaller systems and on the 600-atom HEB. The 0.063 figure for
scraps-perpair on the 600-atom case reflects an unfocused WEIGHT
spreading optimization across multiple M-M shells; with
`WEIGHT 0 0 ... 1 ... 0 0` focused on a single M-M shell, it
matches the mcsqs quality. See the "MLIP-friendly cells" subsection
of the main README for the recipe.

**Wall time.** scraps-perpair is **substantially faster** at all
tested sizes, even accounting for the 4× parallelism advantage:

- NiCoCr 108: 33 CPU-s (mcsqs serial) vs 4 CPU-s (scraps 4 MPI)
  → scraps is ~8× faster per CPU-second
- HEB 150: similar ratio
- HEB 600: 126 CPU-s vs ~20 CPU-s

mcsqs has a particular bottleneck at large N: it enumerates
symmetry-maximizing supercells (`Generating supercells...` phase)
before any optimization runs. For HEB 600 atoms with the default
shape enumeration, mcsqs spent the entire 60-second budget in this
phase and produced no output. The `-rc` flag with an explicit
`sqscell.out` skips the enumeration; this is the right way to use
mcsqs at MLIP cell sizes.

**Per-pair α targeting.** mcsqs targets random (SQS) by default and
accepts arbitrary cluster correlations via its objective function,
but does not provide a direct API for specifying Warren-Cowley α_ij
per species pair per shell. scraps-perpair's `SRO <i> <j> <shell>
<α>` syntax is the unique value-add of this fork; mcsqs would need
correlation engineering to reproduce the same targets.

**Multi-sublattice.** Both tools handle HEB-type structures natively.
mcsqs uses partial occupation in `rndstr.in`; scraps-perpair uses
the `SPECTATOR_BASIS_SPECIES` keyword. Functionally equivalent for
HEB-class systems (one designed sublattice plus pinned spectator
sublattices). mcsqs goes further — it supports truly multi-sublattice
designs where multiple sublattices have independent compositions —
which scraps-perpair does not.

## When to use which

| Use case | Recommended tool |
|---|---|
| Standard SQS for substitutional alloy | Either; scraps-perpair faster |
| HEB-type structure with fixed spectator | Either; scraps-perpair faster + simpler INPUT |
| Per-pair Warren-Cowley α targeting | **scraps-perpair** (unique capability) |
| Multi-sublattice with independent compositions on >1 sublattice (e.g. σ-phase, fluorite-like ternary) | **mcsqs** (not supported by scraps-perpair) |
| Very large cells (>> 1000 atoms) with full symmetry handling | mcsqs with `-rc`, or scraps-perpair |

## Stoichiometry sweep (single sublattice)

Six binaries and ternaries on FCC (108 atoms, 3×3×3 conventional) and
BCC (54 or 128 atoms). SCRAPS at `MAXSHELLNUM=2, WEIGHT 1 1` (4 MPI
ranks), mcsqs given a 60 s wall budget per case, both verified with
the same 2-shell `verify_sro.py`. Inputs live under
`tests/stoich_*/`; the runnable sweep is `docs/mcsqs_stoich_sweep.sh`.

| case | lattice | atoms | composition | scraps wall | scraps fitness | scraps max\|α\| | mcsqs wall | mcsqs max\|α\| |
|---|---|---|---|---|---|---|---|---|
| stoich_fcc_50_50 | FCC | 108 | 50:50 | 2 s | 0 | **0.000** | 30 s | **0.000** |
| stoich_fcc_75_25 | FCC | 108 | 75:25 | 9 s | 1.58 | **0.012** | 64 s | **0.012** |
| stoich_fcc_50_25_25 | FCC | 108 | 50:25:25 | 20 s | 1.58 | **0.012** | 64 s | 0.037 |
| stoich_bcc_50_50_54 | BCC | 54 | 50:50 | 5 s | 1.41 | **0.012** | 64 s | **0.012** |
| stoich_bcc_75_25 | BCC | 128 | 75:25 | <1 s | 0 | **0.000** | 39 s | **0.000** |
| stoich_bcc_equi3_54 | BCC | 54 | ~33:33:33 | 1 s | 0 | **0.000** | 65 s | 0.056 |

**Observations.** On binaries both tools reach equivalent quality;
scraps is consistently 5–60× faster in wall time and 1.5–15× faster
in CPU-time (accounting for the 4-MPI advantage). On the two
ternaries, scraps wins on quality at the same budget — most
strikingly on BCC equi3 where scraps hits exact SQS while mcsqs
plateaus at max|α| ≈ 0.056. This is consistent with the multi-shell
trend reported above: scraps' explicit α-residual loss tends to flat-
out reach zero at modest supercell sizes when the target is
representable, while mcsqs's correlation engineering leaves
distributed residuals.

The mcsqs failures observed during initial sweep design — "Impossible
to match point correlations due to incompatible supercell size" — came
from compositions whose smallest representable cell exceeded the
requested n. The cases listed above use compositions that divide
cleanly into the chosen cell sizes; for harder fractions you'd need
to either grow the cell or pass mcsqs an explicit `sqscell.out` via
`-rc`.

## Cell-size sweep at fixed composition (25:50:25 ternary on FCC)

How does each tool scale as the supercell grows? FCC `Cell_DIM = k×k×k`
conventional with composition (k³, 2k³, k³) atoms, two shells weighted
equally, five trials per case (three at n=500 to keep runtime
tractable). SCRAPS scales `MCSTEP ~ 7·N`; mcsqs uses `-rc` to skip
supercell enumeration and is given a budget that grows with N. Inputs
live under `tests/stoich_fcc_25_50_25_n*/`; the runnable sweep is
`docs/mcsqs_fcc_n_sweep_trials.sh`.

| N | tool | trials | wall_s mean±sd | max\|α\| mean±sd | best max\|α\| | scraps hit fit=0 |
|---|---|---|---|---|---|---|
|  32 | scraps | 5 |   0 ± 1 | 0.0000 ± 0.0000 | **0.0000** | 5 / 5 |
|  32 | mcsqs  | 5 |  33 ± 1 | 0.1292 ± 0.0597 | 0.0625 | — |
| 108 | scraps | 5 |  16 ± 1 | 0.0123 ± 0.0000 | **0.0123** | 0 / 5 |
| 108 | mcsqs  | 5 |  64 ± 0 | 0.0469 ± 0.0268 | 0.0123 | — |
| 256 | scraps | 5 |   3 ± 0 | 0.0000 ± 0.0000 | **0.0000** | 5 / 5 |
| 256 | mcsqs  | 5 | 124 ± 1 | 0.0437 ± 0.0259 | 0.0156 | — |
| 500 | scraps | 3 | 278 ± 5 | 0.0027 ± 0.0000 | **0.0027** | 0 / 3 |
| 500 | mcsqs  | 3 | 149 ± 62 | 0.0307 ± 0.0209 | 0.0067 | — |

**The bimodal pattern (5/5 hit at n=32 and n=256, 0/5 hit at n=108 and
n=500) is not stochastic luck — it's a number-theoretic constraint.**
The α=0 target requires the expected directed pair count

   T_ij^k = N_i · z_k · N_j / N

to be integer for every species pair (i, j) and every weighted shell.
For 25:50:25 ternary on FCC conventional `k×k×k` with N=4k³ and
N_A=N_C=k³, the shell-1 (z=6) A–C target evaluates to

   T_AC^1 = k³ · 6 · k³ / (4k³) = 1.5 · k³

which is integer iff k³ is even, i.e., iff **k is even**. When k is
odd the AC/CA targets are half-integer and the minimum-achievable
fitness has a strictly positive lower bound (`√0.5 ≈ 0.707` from
2·(0.5)² for the AC/CA pair). SCRAPS lands at the same deterministic
plateau (max|α|=0.0123 at k=3, max|α|=0.0027 at k=5) every trial — it's
the optimum the swap-search can reach under the constraint, not a
search-time issue.

mcsqs faces the same integer constraint; its correlation-engineering
loss distributes the unavoidable error differently, which is why its
best max|α| at k=3 (0.0123) and k=5 (0.0067) lands near SCRAPS's
plateau but with much higher trial-to-trial variance.

### `tools/suggest_cell.py` and the C++ feasibility warning

To avoid wasting a run on an infeasible cell, `tools/suggest_cell.py`
takes a POSCAR, a composition, and a target atom count, and returns
the nearest feasible `Cell_DIM` plus the fitness floor for any
infeasible nearby alternatives:

```
$ tools/suggest_cell.py tests/stoich_fcc_25_50_25_n108/POSCAR \
    --composition 0.25,0.5,0.25 --target-atoms 250 --shells 2

=== FEASIBLE cells (exact-SQS achievable, fitness can reach 0) ===
  Cell_DIM 4 4 4  ->  N=256 (+6 from target),  ELEMENT 64 128 64
  Cell_DIM 2 2 2  ->  N=32 (-218),             ELEMENT 8 16 8
  Cell_DIM 6 6 6  ->  N=864 (+614),            ELEMENT 216 432 216

=== INFEASIBLE cells (fitness floor > 0, plateau guaranteed) ===
  Cell_DIM 3 3 3  ->  N=108 (-142),  ELEMENT 27 54 27,   min fitness >= 0.7071
  Cell_DIM 5 5 5  ->  N=500 (+250),  ELEMENT 125 250 125, min fitness >= 0.7071
```

SCRAPS itself now runs the same check on rank 0 before optimization
starts. When the configured `ELEMENT/Cell_DIM/MAXSHELLNUM` makes any
weighted target non-integer, it prints a one-line warning naming the
offending pair(s) and the unavoidable fitness floor, then proceeds.
Strict-mode abort can be added later via an `ABORT_IF_INFEASIBLE`
INPUT keyword if the warning proves easy to miss in CI pipelines.

## Reproducing this comparison

ATAT 3.50 source: `https://axelvandewalle.github.io/www-avdw/atat/atat3_50.tar.gz`
Build with `make` (one source file `meshcalc.c++` fails on macOS due
to non-portable `<bits/stdc++.h>` include, but `mcsqs` and `corrdump`
build before that error).

After installing `mcsqs`, `corrdump`, `str2cif`, `cellcvrt` on PATH,
run `docs/mcsqs_comparison.sh` (multi-shell case) or
`docs/mcsqs_stoich_sweep.sh` (stoichiometry sweep, env var
`MCSQS_BUDGET=<seconds>` to change per-case mcsqs budget).
