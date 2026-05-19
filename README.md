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

## Build

Requires a working MPI compiler, Eigen headers, and a C++17 (or newer)
compiler. On macOS with Homebrew:

```bash
brew install eigen open-mpi
./build.sh
```

The build script invokes `/opt/homebrew/bin/mpic++` explicitly — Anaconda's
bundled MPI compiler is broken on Apple Silicon (looks for a missing
`arm64-apple-darwin*-clang++`), so you have to bypass it. On Linux or
Intel Mac with system MPI, edit `build.sh` to use plain `mpic++`.

Output binary: `SCRAPs/scraps`.

## INPUT file format

All upstream keywords are unchanged. The new directive:

```
SRO  <species_i>  <species_j>  <shell_index>  <alpha_target>
```

Species indices are 0-based and follow the order in `ELEMENT`. Shell index
is 0-based (0 = nearest-neighbor). Lines may appear in any order; entries
are symmetrized automatically (α_ij = α_ji). Pairs not mentioned default
to α = 0.

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

## Known limitations

### `neighbourMatrix()` uses per-atom shell numbering

`Lattice::neighbourMatrix()` builds a separate distance→shell map for
each atom (via `removedublicates(localdists)` inside a per-atom loop).
Two atoms with floating-point precision differences near a shell
boundary can end up with the same physical neighbor assigned to
different shell indices. On a geometrically uniform 3×3×3 cubic BCC
supercell, for example, atom 0 may report 6 neighbors in shell 1
while atom 28 reports 5 (with the missing neighbor showing up in
shell 2 instead).

The original `printtoscreen()` masked this by always accessing
`maxneighbourspershell[nest[i]]` — `nest[i]` is a species index in
`[0, _typesatoms)`, so only atoms 0 or 1 were ever read, and those
atoms had consistent shell assignments.

After the printtoscreen formula was fixed in this fork (commit
[`5f7e8f1`](../../commit/5f7e8f1)), `Lattice::coordination_per_shell()`
takes atom 0 as the canonical reference and emits a one-time warning
if any other atom disagrees. This means SRO results are still
correct on every test case we ship, but downstream code that depends
on atom-2-or-later coordination data would see slightly different
shell boundaries than atom 0.

**Complete fix:** rework `neighbourMatrix()` to use **global** shell
numbering — sort all unique pairwise distances across the whole
supercell once, assign shell indices once, reuse for all atoms.
Tracked but not yet implemented.

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
