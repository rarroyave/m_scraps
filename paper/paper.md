---
title: 'scraps-perpair: per-pair and multi-sublattice Warren-Cowley short-range-order targeting for multicomponent alloys and ceramics'
tags:
  - C++
  - Python
  - materials science
  - high-entropy alloys
  - high-entropy ceramics
  - short-range order
  - special quasi-random structures
  - SQS
  - multi-sublattice
  - density functional theory
authors:
  - name: Raymundo Arroyave
    orcid: 0000-0002-0773-9939
    affiliation: 1
affiliations:
  - name: Department of Materials Science and Engineering, Texas A&M University, USA
    index: 1
date: 23 May 2026
bibliography: paper.bib
---

# Summary

`scraps-perpair` is an open-source C++/Python code for generating
multicomponent alloy and ceramic supercells with user-specified
Warren-Cowley short-range-order (SRO) parameters [@Cowley:1950]. It
extends the upstream Hybrid Cuckoo Search engine of @Singh:2021 with
four capabilities that turn a single-sublattice special-quasi-random-
structure (SQS) tool into a general multi-sublattice SRO design tool:
(1) per-pair, per-shell $\alpha_{ij}^{(k)}$ targets via a single `SRO`
INPUT keyword (the upstream supported only a scalar $\alpha$ applied
to every species pair); (2) spectator-sublattice support for
high-entropy-ceramic-class structures such as AlB$_2$-type diborides
[@HEB:review]; (3) **multi-sublattice SQS and CSRO design** via a
`SUBLATTICE` INPUT keyword that partitions the basis into per-Wyckoff
orbits, with a per-sublattice concentration reference $c_j^{\tau}$ in
the fitness formula that gives the chemically correct target for
structures with two or more chemically-disordered sublattices
(rocksalt HEA carbides, double perovskites, Heusler L2$_1$, fluorite-
type oxides, sigma and Laves phases); and (4) a defensive
POSCAR/INPUT parser with comprehensive regression tests. The code
ships with a Python tool suite (`verify_sro`, `suggest_cell`,
`run_sqs`, `scraps_ase`, `sublattice_detect`) that covers structure
verification, cell-size feasibility analysis, end-to-end workflows,
Atomic Simulation Environment (`ase` [@ase:2017]) interoperability,
and automatic Wyckoff-orbit detection via spglib [@spglib:2018].

# Statement of need

Special quasi-random structures (SQS) [@Zunger:1990] are the standard
tool for modeling random alloys in *ab initio* calculations, and
ATAT's `mcsqs` [@vanderWalle:2013] is the de facto reference
implementation. However, mcsqs targets pair correlations through
cluster expansion, with no direct API for specifying
$\alpha_{ij}^{(k)}$ on a per-species-pair basis, and its internal
supercell enumeration scales poorly for multi-sublattice cells with
$N \gtrsim 400$ atoms. These limitations bite in four application
domains:

1. **Compositionally complex alloys with measured chemical
   short-range order.** Recent diffuse-scattering, atom-probe, and
   *ab initio* studies of NiCoCr, NbMoTaW, and related medium-/
   high-entropy alloys [@Zhang:2020; @Owen:2017] have made
   quantitative $\alpha_{ij}^{(k)}$ values available; researchers
   want to reproduce those patterns in modeling supercells, not just
   generate random ones.

2. **Targeting classical ordered intermetallics.** B2 (CsCl),
   L1$_2$ (Cu$_3$Au), L1$_0$ (CuAu), D0$_3$ (Fe$_3$Al), L1$_1$
   (CuPt), and Heusler L2$_1$ structures all have characteristic
   $\alpha$ signatures with different values at different shells and
   species pairs. A per-pair API expresses these naturally; a
   scalar-$\alpha$ API cannot.

3. **High-entropy ceramics with one designed and one or more
   spectator sublattices** such as the AlB$_2$-type diborides where
   the metal sublattice carries the chemical disorder and the boron
   sublattice is a fixed structural scaffold.

4. **Truly multi-sublattice design** where two or more sublattices
   are independently chemically disordered with potentially distinct
   composition targets and SRO patterns. The canonical examples
   include rocksalt-type high-entropy carbides and nitrides where
   both the metal and anion sublattices carry disorder
   [@HEC:Sarker], double perovskites, full Heusler L2$_1$ with
   independent X/Y/Z mixing [@Heusler:Graf], spinel-type
   ferrites/oxides with cation site disorder, and sigma- or
   Laves-phase intermetallics with five and two crystallographically
   distinct Wyckoff orbits respectively. These structures have
   physically different per-orbit coordination numbers (e.g.
   fluorite-type cation $z=8$, anion $z=4$ at 1NN), which the
   global single-reference Warren-Cowley formulation in upstream
   SCRAPS and in mcsqs's correlation-engineering treatment does not
   handle natively.

`scraps-perpair` was developed to fill these gaps while preserving
the underlying Hybrid Cuckoo Search optimizer that makes SCRAPS
competitive with `mcsqs` on speed and SQS quality.

# Algorithmic and implementation contributions

Beyond the per-pair API, the fork includes:

- **Natural same-species SRO targets with Warren-Cowley sum-rule
  auto-expansion.** SCRAPs's fitness sums over cross-species pairs
  only ($i \ne j$), since at fixed composition $\alpha_{ii}$ is
  linearly determined by the cross-species $\alpha_{ij}$ through
  $\sum_j c_j \alpha_{ij} = 0$. Users nevertheless write the natural
  literature form (`SRO 2 2 0 -0.10` for $\alpha_{\mathrm{Cr,Cr}}^{(1)}
  = -0.10$); the parser auto-expands via the symmetric split
  $\alpha_{ij} = -c_i \alpha_{ii} / (1 - c_i)$ for each $j \ne i$.
  Explicit cross-species `SRO i j shell alpha` lines override the
  symmetric split regardless of input order, and a diagnostic warning
  fires when the implied $\alpha_{ii}$ recomputed from the final
  cross-species pattern disagrees with the user-stated same-species
  value, so the optimizer never silently chases a mutually inconsistent
  target.

- **Multi-sublattice fitness with per-sublattice $c_j^{\tau}$
  reference and per-$(\sigma, \tau, k)$ coordination tensor.** Each
  `SUBLATTICE` block declares a set of basis positions and an allowed
  species set; species are required to be disjoint across sublattices.
  The expected pair count becomes
  $T_{ij}^{(k)} = N_i (1 - \alpha) z^{\sigma \to \tau}_k N_j / N_{\tau}$
  with $\sigma = \mathrm{sublattice}(i)$ and $\tau = \mathrm{sublattice}(j)$.
  All five swap-proposal sites (cuckoo, fast random, targeted,
  cluster-$K$, simulated annealing) are gated to reject
  cross-sublattice swaps. For non-uniform-coordination structures
  (fluorite, sigma, Laves, MAX phases) the per-Wyckoff `SUBLATTICE`
  workflow unlocks SRO design that the legacy global-uniformity
  check rejected. For single-sublattice and HEB-spectator cases the
  formula reduces exactly to the legacy expression, preserving
  backward compatibility.

- **Wyckoff-orbit auto-detection** via `tools/sublattice_detect.py`,
  which reads a POSCAR through pymatgen, runs a space-group analysis
  via spglib [@spglib:2018], and emits draft `SUBLATTICE` blocks for
  each Wyckoff orbit with placeholder SPECIES entries the user fills
  in. Two basis positions in the same Wyckoff orbit have identical
  pair-correlation environments under symmetry, so the orbit
  partition is the unique coarsest partition that respects the
  lattice symmetry.

- **Alternative MC kernels** beyond the upstream cuckoo search:
  `MC_KERNEL=sa` selects simulated annealing with a log-spaced
  temperature schedule; `MC_KERNEL=pt` selects parallel tempering
  across MPI ranks with Metropolis exchanges between adjacent
  temperatures. Both share the delta-fitness machinery and the swap
  proposal mechanism with the cuckoo path.

- **$K$-atom cluster swap moves** (`CLUSTER_SWAP_K`) for the
  algorithmic-trap cases where single-swap MC plateaus.

- **Warm-start API** (`INIT_NEST`) accepting a per-atom species
  vector as the initial configuration, useful for iterative
  refinement workflows and for steep multi-pair targets where pure
  random initialization plateaus far from the global minimum.

- **Six bug fixes to the upstream code**, including a species-index
  pair-gating bug in the fitness function that silently mis-classified
  approximately 40% of pairs depending on memory layout, an
  uninitialized `totalatoms` field in the POSCAR reader, and an
  off-by-$N$ Warren-Cowley formula in the on-screen $\alpha$ printer
  that disagreed with what the optimizer was minimizing.

- **A delta-fitness Monte Carlo path** maintaining the pair-count
  tensor as state and applying $O(z)$ updates per swap instead of
  recomputing the full $O(N z)$ fitness, yielding 2-18$\times$
  speedup on optimization-loop-bound cases with identical
  convergence to the legacy recompute path.

- **A defensive POSCAR parser** handling the full canonical VASP-5
  layout (positive and negative scale factor, optional species label
  line, multi-species count groups, Selective dynamics,
  Direct/Cartesian coordinate keyword), with `MPI_Abort` and a clear
  diagnostic on malformed input.

- **User-facing heuristics** (`AUTO_MCSTEP`, `EARLY_STOP_PATIENCE`,
  `RNG_SEED`, `RANDOM_ONLY`, `CONVERGENCE_LOG`, `FAST_SWAPS`,
  `TARGETED_SWAPS`) that lift the manual-tuning burden and make
  optimization runs reproducible and auditable.

- **A feasibility check** (`tools/suggest_cell.py` + a C++ startup
  warning) that flags when the configured
  `(ELEMENT, Cell_DIM, MAXSHELLNUM, SUBLATTICE)` makes the target
  pair counts non-integer, in which case exact SQS is mathematically
  unreachable on that cell. The check uses the per-sublattice
  $z^{\sigma \to \tau}_k$ and $N_{\tau}$ quantities consistently with
  the fitness formula.

# Validation

Six classical ordered structures (B2, L1$_2$, L1$_0$, D0$_3$, L1$_1$,
and L2$_1$/Heusler) and six multi-sublattice structure classes
(fluorite, Laves C15, Heusler L2$_1$ end-to-end, sigma $\sigma$-FeCr,
spinel, pyrochlore) are recovered as worked test cases in the
repository. A 183-test pytest suite, every test pinned via
`RNG_SEED`, covers structure verification, feasibility math,
end-to-end optimization regressions, convergence-trajectory checks,
ASE I/O round-tripping, multi-sublattice parser validation, Wyckoff-
orbit detection, per-sublattice SQS / CSRO quality on non-uniform-
coordination cells, and same-species SRO sum-rule expansion with
explicit-override diagnostics. Continuous integration on Ubuntu builds
the C++ and runs the full suite against Python 3.11 and 3.12.

A worked end-to-end research case reproducing the Zhang et al. NiCoCr
1NN Cr-Cr avoidance pattern [@Zhang:2020] across 108-, 256-, and 500-
atom FCC supercells (`docs/research_cases/01_cocrni_csro.md`) reaches
the literature target $\alpha_{\mathrm{Cr,Cr}}^{(1)} = -0.10$ to
within 0.003 at every cell size in under 2 s of wall time per
supercell on a laptop, and exercises the natural same-species INPUT
form together with the sum-rule auto-expansion described above.

Head-to-head comparisons with ATAT's `mcsqs` 3.50 on four
structurally distinct multi-sublattice cells (`docs/mcsqs_comparison.md`)
show `scraps-perpair` converging to exact per-sublattice $\alpha=0$
in well under one second at every cell size tested (16 to 432 atoms),
while `mcsqs` requires 2 to 130 seconds per case and leaves residual
max$|\alpha| \in [0.1, 0.7]$ at its convergence point. For the
`mcsqs`-untargetable per-pair CSRO patterns, `scraps-perpair` hits
arbitrary target $\alpha$ values on multiple sublattices
simultaneously to within numerical precision in $\sim 1$ s on
128-atom cells. An independent random-baseline ablation, pinned to
the same `RNG_SEED` and using the `RANDOM_ONLY` keyword,
demonstrates 12-15$\times$ quality improvement over
best-of-$N$-random-nests on infeasibility-plateau cases and
qualitatively infinite improvement on ordered-structure targets,
where random search has $\sim 10^{-35}$ probability per nest of
finding the target.

# Mathematics

The Warren-Cowley parameter for species pair $(i, j)$ at coordination
shell $k$ is
$$\alpha_{ij}^{(k)} = 1 - \frac{P(j \mid i, k)}{c_j^{\tau}},$$
where $P(j \mid i, k)$ is the conditional probability that the
neighbor of species $i$ at shell $k$ is species $j$ and $c_j^{\tau} =
N_j / N_{\tau}$ is the concentration of $j$ on its own sublattice
$\tau$. For single-sublattice systems $N_{\tau} = N$ and the formula
reduces to the standard expression. The loss function on a candidate
supercell is
$$E = \sum_{k} w_k \sum_{i \neq j} \left( N_{ij}^{(k)} - T_{ij}^{(k)} \right)^2,$$
with target pair counts
$$T_{ij}^{(k)} = N_i \left( 1 - \alpha_{ij}^{(k),\,\text{target}} \right) z^{\sigma \to \tau}_k N_j / N_{\tau}$$
and $N_{ij}^{(k)}$ the directed pair count in the candidate, where
$\sigma = \mathrm{sublattice}(i)$, $\tau = \mathrm{sublattice}(j)$,
and $z^{\sigma \to \tau}_k$ is the average count of $\tau$-sublattice
neighbors of a $\sigma$-sublattice atom at shell $k$ (a per-
sublattice-pair coordination tensor built once at startup from the
lattice and the declared SUBLATTICE partition). The optimizer
minimizes $E$ over the discrete space of valid atomic arrangements
(swaps confined to atoms on the same sublattice) via a cuckoo-search
metaheuristic with Monte Carlo swap moves and an $\epsilon$-band
stochastic acceptance to escape micro-plateaus, or via the optional
simulated annealing and parallel tempering kernels.

# Acknowledgements

This fork builds on the upstream Hybrid Cuckoo Search code by Rahul
Singh, Prashant Singh, Aayush Sharma, Ganesh Balasubramanian, and
Duane D. Johnson [@Singh:2021]. The MIT license is preserved with
dual copyright (Iowa State University and the fork author).

# References
