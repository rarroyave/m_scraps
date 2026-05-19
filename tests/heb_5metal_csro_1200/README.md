# HEB 5-metal CSRO at 1200 atoms — large-cell precision test

Demonstrates that per-pair CSRO targeting can hit specified α values
to ~1% precision in MLIP-friendly cell sizes (1200 atoms).

## Structure

(Hf,Zr,Ti,Ta,Nb)B₂ AlB₂-type, 10×10×4 supercell:

- **1200 atoms total** (400 M + 800 B)
- Equiatomic on M sublattice: 80 atoms of each metal
- B declared as spectator via `SPECTATOR_BASIS_SPECIES`

## CSRO targets (shell 8 only)

| Pair | Target α | Achieved α (typical run) | Error |
|---|---|---|---|
| Hf-Zr | +0.30 | +0.302 | 0.002 |
| Hf-Ti | −0.30 | −0.302 | 0.002 |
| Ti-Ta | −0.30 | −0.302 | 0.002 |
| Ta-Nb | +0.30 | +0.302 | 0.002 |

Untargeted pairs stay at α ≈ 0 (within ±0.01). Achievable α-precision
at this supercell size is roughly 1/(N_i × z) = 1/(80 × 128) ≈ 10⁻⁴,
so the limiting factor is the optimizer's MCMC budget, not the
discrete grid.

## Important: WEIGHT structure for multi-M-M-shell systems

In an AlB₂ supercell large enough to span multiple M-M shells (here
SCRAPS sees M-M pairs at internal shells 8, 11, 14), the WEIGHT
vector must be focused on the targeted shell. Equal weights on all
shells (the default-looking `WEIGHT 1 1 1 ...`) puts SQS pressure on
shells 11 and 14, which competes with the CSRO target on shell 8.
The compromise solution stays near SQS at every shell.

This INPUT uses:
```
WEIGHT  0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
```

— weight 1 only at index 8 (the in-plane M-M shell). The unweighted
shells get whatever α the optimizer leaves them at; for random initial
configurations they end up near SQS naturally.

## MCSTEP scaling

`MCSTEP=500` works fine at 150 atoms but produces under-converged
results at 1200 atoms (with WEIGHT all-shells-equal, achieved CSRO
moved barely from random). This case uses `MCSTEP=5000, GITER=5000` —
roughly proportional to N. Wall time ~25 s on 4 ranks.

## Run

```bash
ln -sf ../../SCRAPs/scraps ./scraps
mpirun -np 4 ./scraps
```

Expected: ~25 s wall time, final fitness around 1.5, achieved α
matrix at SCRAPS shell 8 matching the table above.
