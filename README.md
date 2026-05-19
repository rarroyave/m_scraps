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

### Not supported: multi-sublattice structures with non-uniform coordination

Lattices where different sublattice sites have **physically different
coordination numbers**:

- **Sigma phase** (σ, 5 Wyckoff positions with z = 12–15)
- **Laves phases** (C14, C15, C36)
- **Cage structures** (NaZn₁₃, Th₆Mn₂₃, A15)
- **Fluorite** (CaF₂-type, cation z=8 vs anion z=4)
- **Layered structures** with in-plane vs out-of-plane coordination
  asymmetry (graphite, hBN, MAX phases)
- Lattices with **interstitial sublattices** (C in austenite, H in
  metals)

For these, the meaningful quantity is **sublattice-conditional SRO**
α^(σ,τ)_ij defined per sublattice pair, against per-sublattice
concentrations c_j^(τ). scraps-perpair does not currently implement
sublattice resolution, and `Lattice::coordination_per_shell()` will
abort with a diagnostic if it detects truly non-uniform coordination
across atoms (the soft warning previously emitted has been upgraded
to a hard abort).

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
