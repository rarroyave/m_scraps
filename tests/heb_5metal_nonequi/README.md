# HEB 5-metal non-equiatomic at 432 atoms

Validates that the per-sublattice reference concentration c_j^M is
computed correctly when the M sublattice composition is non-equiatomic.

## Structure

(Hf₅₀Zr₃₀Ti₃₀Ta₂₀Nb₁₄)B₂ in a 6×6×4 supercell:

- **432 atoms** (144 M + 288 B)
- Hf-rich: 50/30/30/20/14 → 34.7% Hf, 20.8% Zr, 20.8% Ti, 13.9% Ta, 9.7% Nb
- No explicit CSRO target (want SQS)
- B declared as spectator

## What this tests

A common bug in spectator-aware SRO implementations is using global c_j
(species fraction over all atoms including B) instead of c_j^M (species
fraction over M-sublattice atoms only). With B as 67% of atoms here,
the global vs M-sublattice c_j values differ by 3× — a bug would
produce α values systematically off by similar factors.

Expected behavior:
- α(Nb-X) ≈ −0.029 for every X (Nb is the rarest metal at 14/144, so
  the discrete-count step is largest for any pair involving Nb)
- Other off-diagonal α should be small (max ~0.08)
- α matrix should be exactly symmetric — α(Hf-Zr) = α(Zr-Hf) etc.
- α(i,i) emerges from the sum rule Σ_j c_j^M α_ij = 0

If the c_j vs c_j^M bug were present, α(metal-metal) values would be
3× too large or too small.

## Run

```bash
ln -sf ../../SCRAPs/scraps ./scraps
mpirun -np 4 ./scraps
```

Expected wall time ~3 s; output α matrix on shell 8 (first M-M
shell) with the properties above.
