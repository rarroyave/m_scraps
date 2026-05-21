---
title: 'scraps-perpair: per-pair Warren-Cowley short-range-order targeting for multicomponent alloy supercells'
tags:
  - C++
  - Python
  - materials science
  - high-entropy alloys
  - short-range order
  - special quasi-random structures
  - SQS
  - density functional theory
authors:
  - name: Raymundo Arroyave
    orcid: 0000-0002-0773-9939
    affiliation: 1
affiliations:
  - name: Department of Materials Science and Engineering, Texas A&M University, USA
    index: 1
date: 20 May 2026
bibliography: paper.bib
---

# Summary

`scraps-perpair` is an open-source C++/Python code for generating
multicomponent alloy supercells with user-specified Warren-Cowley
short-range-order (SRO) parameters [@Cowley:1950]. It extends the
upstream Hybrid Cuckoo Search engine of @Singh:2021 with three
capabilities that turn a special-quasi-random-structure (SQS) tool
into a general SRO-targeting tool: (1) per-pair, per-shell
$\alpha_{ij}^{(k)}$ targets via a single `SRO` INPUT keyword (the
upstream supported only a scalar $\alpha$ applied to every species
pair); (2) spectator-sublattice support for high-entropy
ceramic-class structures such as AlB$_2$-type diborides
[@HEB:review]; and (3) a defensive POSCAR/INPUT parser with
comprehensive regression tests. The code ships with a Python tool
suite (`verify_sro`, `suggest_cell`, `run_sqs`, `scraps_ase`) that
covers structure verification, cell-size feasibility analysis,
end-to-end workflows, and Atomic Simulation Environment
(`ase` [@ase:2017]) interoperability.

# Statement of need

Special quasi-random structures (SQS) [@Zunger:1990] are the
standard tool for modeling random alloys in *ab initio*
calculations, and ATAT's `mcsqs` [@vanderWalle:2013] is the de facto
reference implementation. However, mcsqs targets pair correlations
through cluster expansion, with no direct API for specifying
$\alpha_{ij}^{(k)}$ on a per-species-pair basis. This becomes a
limitation in three increasingly important application domains:

1. **Compositionally complex alloys with measured chemical
   short-range order.** Recent diffuse-scattering, atom-probe, and
   *ab initio* studies of NiCoCr, NbMoTaW, and related medium-/
   high-entropy alloys [@Zhang:2020; @Owen:2017] have made
   quantitative $\alpha_{ij}^{(k)}$ values available; researchers
   want to reproduce those patterns in modeling supercells, not
   just generate random ones.

2. **Targeting classical ordered intermetallics.** B2 (CsCl), L1$_2$
   (Cu$_3$Au), L1$_0$ (CuAu), D0$_3$ (Fe$_3$Al), L1$_1$ (CuPt), and
   Heusler L2$_1$ structures all have characteristic $\alpha$
   signatures with different values at different shells and species
   pairs. A per-pair API expresses these naturally; a scalar-$\alpha$
   API cannot.

3. **High-entropy ceramics** such as diborides, carbides, and oxides
   with one designed and one or more spectator sublattices, where
   pair statistics must be restricted to the active sublattice.

`scraps-perpair` was developed to fill these gaps while preserving
the underlying Hybrid Cuckoo Search optimizer that makes SCRAPS
competitive with `mcsqs` on speed and SQS quality.

# Algorithmic and implementation contributions

Beyond the per-pair API, the fork includes:

- **Six bug fixes to the upstream code**, including a species-index
  pair-gating bug in the fitness function that silently mis-classified
  approximately 40% of pairs depending on memory layout, an
  uninitialized `totalatoms` field in the POSCAR reader, and an
  off-by-$N$ Warren-Cowley formula in the on-screen $\alpha$
  printer that disagreed with what the optimizer was minimizing.
- **A delta-fitness Monte Carlo path** maintaining the pair-count
  tensor as state and applying $O(z)$ updates per swap instead of
  recomputing the full $O(N z)$ fitness, yielding 2-18$\times$
  speedup on optimization-loop-bound cases with identical
  convergence to the legacy recompute path.
- **A defensive POSCAR parser** handling the full canonical VASP-5
  layout (positive and negative scale factor, optional species
  label line, multi-species count groups, Selective dynamics,
  Direct/Cartesian coordinate keyword), with `MPI_Abort` and a
  clear diagnostic on malformed input.
- **User-facing heuristics** (`AUTO_MCSTEP`,
  `EARLY_STOP_PATIENCE`, `RNG_SEED`, `RANDOM_ONLY`,
  `CONVERGENCE_LOG`, `FAST_SWAPS`, `INIT_NEST`) that lift the
  manual-tuning burden and make optimization runs reproducible and
  auditable.
- **A feasibility check** (`tools/suggest_cell.py` + a C++
  startup warning) that flags when the configured
  `(ELEMENT, Cell_DIM, MAXSHELLNUM)` makes the target pair counts
  non-integer, in which case exact SQS is mathematically
  unreachable on that cell.

# Validation

Six classical ordered structures (B2, L1$_2$, L1$_0$, D0$_3$, L1$_1$,
and L2$_1$/Heusler) are recovered as worked test cases in the
repository. A 79-test pytest suite, every test pinned via
`RNG_SEED`, covers structure verification, feasibility math,
end-to-end optimization regressions, convergence-trajectory checks,
and ASE I/O round-tripping. Continuous integration on Ubuntu builds
the C++ and runs the full suite against Python 3.11 and 3.12.

A head-to-head comparison with ATAT's `mcsqs` 3.50 on FCC and BCC
substitutional alloys and high-entropy diborides (documented in
`docs/mcsqs_comparison.md`) shows `scraps-perpair` matching or
beating `mcsqs` on max-$|\alpha|$ at comparable wall budgets, with
5-60$\times$ shorter wall times and 1.5-15$\times$ shorter CPU
times. An independent random-baseline ablation, pinned to the
same RNG seed and using the `RANDOM_ONLY` keyword, demonstrates
12-15$\times$ quality improvement over best-of-$N$-random-nests
on infeasibility-plateau cases and qualitatively infinite
improvement on ordered-structure targets (where random search has
$\sim 10^{-35}$ probability per nest of finding the target).

# Mathematics

The Warren-Cowley parameter for species pair $(i, j)$ at
coordination shell $k$ is
$$\alpha_{ij}^{(k)} = 1 - \frac{P(j \mid i, k)}{c_j},$$
where $P(j \mid i, k)$ is the conditional probability that the
neighbor of species $i$ at shell $k$ is species $j$, and $c_j$ is
the bulk concentration of $j$. The loss function on a candidate
supercell is
$$E = \sum_{k} w_k \sum_{i \neq j} \left( N_{ij}^{(k)} - T_{ij}^{(k)} \right)^2,$$
with target pair counts
$$T_{ij}^{(k)} = N_i \left( 1 - \alpha_{ij}^{(k),\,\text{target}} \right) z_k N_j / N$$
and $N_{ij}^{(k)}$ the directed pair count in the candidate. SCRAPS
minimizes $E$ over the discrete space of atomic arrangements via a
cuckoo-search metaheuristic with Monte Carlo swap moves and an
$\epsilon$-band stochastic acceptance to escape micro-plateaus.

# Acknowledgements

This fork builds on the upstream Hybrid Cuckoo Search code by Rahul
Singh, Prashant Singh, Aayush Sharma, Ganesh Balasubramanian, and
Duane D. Johnson [@Singh:2021]. The MIT license is preserved with
dual copyright (Iowa State University and the fork author).

# References
