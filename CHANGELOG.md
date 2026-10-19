# Changelog

## Unreleased — planned 0.6.0

Release preparation only: no v0.6.0 tag, GitHub release, or Zenodo deposition
has been published. The existing 0.5.1 DOI continues to identify that frozen
archive; a new DOI will be added only after an approved release is archived.

### Small-cell design with explicit accuracy and geometry limits

- Add `m_scraps bounds` and `sro_bounds` for necessary lower bounds on explicit
  pair-SRO errors, including integer capacities, periodic bond multiplicities,
  self-image restrictions, and binary cut parity. Support single and multiple
  labeled sublattices, shared species, and fixed spectators.
- Add `m_scraps enumerate` and `enumerate_sro` to certify selected pair-SRO
  minima in a fixed cell. Symmetry reduction verifies label preservation and
  exact directed target-bond multiplicities, then generates occupation orbits
  through subset stabilizers. Export witness structures and rational errors.
- Stop enumeration when a witness exactly attains the rigorous lower bound;
  distinguish this proof from exhaustive coverage. Retain `--exhaustive` for
  full-coverage audits and `--symmetry none` for labeled-site reference traversal.
  Time-limited searches without a proof remain explicitly uncertified.
- Add `m_scraps search-cells` and `search_cells` to explore bounded ranges of
  integer supercells, remove equivalent lattices, enforce exact compositions,
  and filter geometry using reduced-basis aspect ratios and shortest repeats.
  Screen impossible tolerances by their bounds, use capped exact enumeration or
  seeded Python pair-SRO SA, and return several candidates with geometry/error
  tradeoffs and independently checkable witnesses.
- Add [notebook 08](notebooks/08_small_cell_search.ipynb), covering BCC and
  multi-sublattice rocksalt, loose versus attained bounds, early-stop proofs,
  constrained geometry, and tolerance screening. It needs no ATAT or icet and
  invokes no native optimizer; package CI executes it outside the checkout.

### Reporting, attribution, and maintenance

- Add a self-contained HTML benchmark report with filterable trial metrics,
  censored threshold times, an embedded figure, and source checksums. Archived
  benchmark measurements are unchanged; no larger campaign is introduced.
- Expand the upstream comparison and document benchmark scope, metric
  definitions, backend differences, and timing limitations.
- Validate citation schema, ORCID checksums, author metadata, and consistency
  across package, CMake, Python, citation, Zenodo, and release-tag versions.
- Support Open MPI 5/PRRTE oversubscription in hosted CI correctness tests while
  retaining Open MPI 4 compatibility. Production optimizer settings are unchanged.

### Scope and compatibility

The new bounds and exact certificates concern **explicit pair-SRO error** with
uniform coordination in each requested sublattice-pair shell. They do **not**
bound decorated pair/triplet/quadruplet correlation errors or the native weighted
bond-count objective. Componentwise bounds can remain loose; a zero bound alone
does not prove feasibility.

Certificates apply to a **specified cell, composition, sublattice partition,
target list, and selected metric**. Bounded cell-shape searches do **not** certify
a global optimum across all cell sizes or shapes. Geometry constraints express
user preferences, not DFT convergence guarantees. The cell-search SA uses the
explicit pair-SRO metric and is distinct from the native SCRAPS SA kernel.

Bounds reports use schema version 2; enumeration reports use schema version 3.
A bound-attainment certificate may have `complete: false`: `complete` denotes
full enumeration coverage. Consumers should inspect `status` and
`certificate_basis` separately. The configuration cap applies before symmetry
reduction; an already-certified initial witness bypasses enumeration entirely.

Multi-sublattice generation and the native simulated-annealing solver were
introduced in **0.5.0** and remain supported. This release extends them with
small-cell feasibility/design workflows; it does not claim those earlier
capabilities as new inventions. Original SCRAPS attribution, MIT licensing,
PhaseForge and AFRL/ARCTOS acknowledgments, and the request to cite both the
original paper and the appropriate software archive remain in place.

## 0.5.1 — 2026-10-19

- Correct the author ORCID in citation, Zenodo, and paper metadata.
- Archive the release on Zenodo:
  [10.5281/zenodo.23475896](https://doi.org/10.5281/zenodo.23475896).
- Scientific code is unchanged from 0.5.0. The DOI was subsequently added to
  the GitHub README, citation metadata, and release notes.

## 0.5.0 — 2026-10-19

- Package m_scraps with Python and CLI interfaces and legacy compatibility.
- Add multi-sublattice capabilities with independently constrained species and
  compositions, including fixed and disordered sublattices; provide examples
  for diborides, MAX phases with M/A disorder, and Heusler phases.
- Add a simulated annealing solver (`MC_KERNEL sa`) with Metropolis acceptance
  and configurable exponential cooling within the existing outer search loop,
  alongside the default cuckoo MC kernel.
- Provide SRO and pair/triplet/quadruplet correlation targeting and independent
  structure analysis.
- Include a limited, reproducible four-method benchmark, frozen ATAT references,
  optional external comparisons, and automated tests and notebook checks.
- Fix missing-pair penalties in the full, fast-swap, and simulated-annealing
  SRO objectives.
- Preserve original SCRAPS attribution; add MIT licensing and PhaseForge and
  AFRL/ARCTOS funding acknowledgments.
- Zenodo ingestion failed because of invalid ORCID metadata; use the corrected
  0.5.1 archive for citation.

Earlier upstream development remains documented in the Git history.
