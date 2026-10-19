# Changelog

## Unreleased

- Add bounded cell-size/shape search with reduced-basis aspect and shortest-repeat
  constraints, symmetry-equivalent lattice removal, exact composition checks,
  pair-SRO bound screening, exact enumeration or seeded SA, and Pareto candidates
  with exported witnesses. Certificates remain specific to individual cells.

- Enable symmetry-reduced enumeration by default, verifying sublattice labels
  and exact target bond graphs, generating representatives through subset
  stabilizers, and recording orbit coverage. Retain `--symmetry none` as an
  exhaustive reference; the configuration cap still applies before reduction.
- Add capped exact pair-SRO enumeration for small single- and multi-sublattice
  cells, with rational comparisons, completion certificates, witness structures,
  and explicitly uncertified time-limited or interrupted results.
- Add `m_scraps bounds` and the Python `sro_bounds` API for independent integer
  pair-count SRO lower bounds, achieved upper bounds, and witness-based
  optimality certificates. Reject nonuniform coordination; report limitations.
- Add a standalone HTML benchmark report with filterable trial metrics,
  censored threshold times, an embedded comparison figure, and source checksums.
- Expand the upstream comparison to distinguish scientific capabilities from
  performance, workflow, and validation improvements, with backend limitations
  stated explicitly.
- Add CI checks for citation schema, ORCID checksums, author metadata, and
  consistent package, Python, CMake, Zenodo, citation, and release-tag versions.
- Make benchmark scope, error definitions, and timing limitations easier to find.
- Add this release history. No scientific algorithms or archived results change.

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
