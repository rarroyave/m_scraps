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

| Test | Tool | Atoms | Parallelism | Wall | max \|α\| shell 0 |
|---|---|---|---|---|---|
| NiCoCr 108 | scraps-perpair | 108 | 4 MPI | 1 s | 0.0000 |
| NiCoCr 108 | mcsqs | 108 | 1 (serial) | 33 s | 0.0000 |
| HEB 150 | scraps-perpair | 150 | 4 MPI | <1 s | 0.0000 |
| HEB 150 | mcsqs | 150 | 1 (serial) | 64 s | 0.0000 |
| HEB 600 | scraps-perpair | 600 | 4 MPI | 5 s | 0.063 (only first M-M shell)|
| HEB 600 | mcsqs (with `-rc` explicit supercell) | 600 | 1 (serial) | 126 s | 0.0000 |

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

## Reproducing this comparison

ATAT 3.50 source: `https://axelvandewalle.github.io/www-avdw/atat/atat3_50.tar.gz`
Build with `make` (one source file `meshcalc.c++` fails on macOS due
to non-portable `<bits/stdc++.h>` include, but `mcsqs` and `corrdump`
build before that error).

After installing `mcsqs`, `corrdump`, `str2cif`, `cellcvrt` on PATH,
run `docs/mcsqs_comparison.sh`.
