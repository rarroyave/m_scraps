# Fluorite — intentionally rejected (multi-sublattice)

This case is **not** supposed to run successfully. It exists to verify
that `Lattice::coordination_per_shell()` correctly detects multi-sublattice
structures with non-uniform coordination and aborts with the expected
diagnostic.

## Structure

Fluorite-like 12-atom basis in a unit cubic cell:

- **4 cation sites** at the FCC positions: (0,0,0), (½,½,0), (½,0,½), (0,½,½)
- **8 anion sites** at the tetrahedral interstitials: (¼,¼,¼), (¾,¼,¼), … (¾,¾,¾)

Built out to a 2×2×2 supercell (96 atoms total: 32 cation positions +
64 anion positions; assigned 32 species 1 + 64 species 2 in `INPUT`).

## Why it's not supported

Cation sites have **z=8** nearest-neighbor anions at distance a√3/4.
Anion sites have **z=4** nearest-neighbor cations at the same distance.
The coordination is physically non-uniform across sublattices, so the
Warren-Cowley α = 1 − P(j|i)/c_j defined against a single global c_j
is not a meaningful reference.

## Expected behavior

```bash
mpirun -np 2 ./scraps
```

should exit with code 1 in ~1 second, after printing:

```
Creating Lattice.....
Lattice Built.....
ERROR: non-uniform coordination across atoms (shell 0: atom 0 z=8, atom 32 z=4).
  This indicates a multi-sublattice structure with physically distinct site environments
  (sigma phase, Laves, fluorite, layered, interstitial, etc.). ...
  For sublattice-resolved SRO, use ATAT's mcsqs ...
```

## What to do for fluorite-type problems

Use ATAT's `mcsqs` (https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/),
which supports sublattice-resolved SQS construction natively.
