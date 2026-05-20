# Contributing to scraps-perpair

Welcome. This guide covers the practical bits — building, testing,
branching, and PR conventions. For the *what is this project* answer
see `README.md` and `docs/OVERVIEW.md` (also as PDF).

## Build

You need a working MPI installation (Open MPI or MPICH), Eigen
headers, and a C++17 compiler. On macOS with Homebrew:

```bash
brew install eigen open-mpi
./build.sh
```

On Linux:

```bash
sudo apt install libopenmpi-dev openmpi-bin libeigen3-dev   # Debian/Ubuntu
MPICXX=mpic++ EIGEN_INC=/usr/include/eigen3 ./build.sh
```

Output binary: `SCRAPs/scraps`.

### macOS launcher gotcha

If `which mpirun` returns something like `/opt/anaconda3/bin/mpirun`
(anaconda ships MPICH's Hydra), launching scraps via that will silently
fall back to singleton mode — each MPI "rank" runs an independent
search and only one rank's output survives. Always launch with the
`mpirun` from the same MPI as the `mpic++` that built scraps:

```bash
/opt/homebrew/bin/mpirun -np 4 SCRAPs/scraps    # Homebrew Open MPI
```

The Python wrappers (`tools/run_sqs.py`, `tools/scraps_ase.py`) and
the pytest suite auto-resolve this; only matters when invoking
`scraps` directly.

## Test

Python tools + end-to-end optimization regressions:

```bash
pip install -r tools/requirements.txt
pytest tools/tests/
```

73 tests, ~30 seconds locally. Tests that need the scraps binary or
mpirun skip gracefully when those aren't available, so a fresh
clone passes `pytest` before `./build.sh`.

CI (`.github/workflows/test.yml`) installs MPI and Eigen on the
Ubuntu runner, builds scraps, and runs the full suite against
Python 3.11 and 3.12. Every PR triggers CI; please keep it green.

## Workflow

### Branching

- `main` is protected — direct pushes only for the owner; all
  contributions land via PR.
- Feature branches: `feature/<short-description>` (e.g.
  `feature/warm-start-api`).
- Bug fixes: `fix/<short-description>`.
- One topic per branch; keep branches narrow.

### Commits

- First line ≤ 72 characters, in imperative voice
  (`Add early-stop heuristic`, not `Added` or `Adding`).
- Body wrapped at ~72 columns; use bullet lists when multiple distinct
  things change.
- Reference issues in the body or footer: `Closes #12` or `See #12`.
- If you used Claude / Copilot / other AI assistance, add a trailer:

  ```
  Co-Authored-By: <tool name and version> <noreply@example.com>
  ```

### Pull requests

- Open against `main`.
- Title: same conventions as a commit subject line.
- Description should include:
  - **What** changed and **why** (the why is more important than the
    what — the diff already shows what).
  - **How tested**: which pytest cases cover this, whether you ran
    `./build.sh && pytest tools/tests/` locally.
  - **Risks / breaking changes**: any INPUT keyword changes, any
    behavior changes that existing users would notice.
- Keep PRs small — under ~300 lines of changes is a good target.
  Big architectural changes are easier as a series of small PRs.
- Ensure CI is green before requesting review.

## Coding conventions

### C++ (`SCRAPs/`)

- C++17. We don't currently use C++20 features.
- Indentation: 4 spaces, no tabs.
- Naming: existing code is a mix of `camelCase` and `snake_case`;
  match the surrounding style of the file you're editing.
- Error handling: aborts that come from user-input validation
  (POSCAR / INPUT) use `MPI_Abort` + a clear `cout` diagnostic
  naming the offending field. See `poscar_abort` and `input_abort`
  in `readInput.cpp` for the pattern.
- New INPUT keywords: add to the `KNOWN_KEYS` set in `readInput.cpp`
  so unrecognized-keyword warnings don't trigger on them. Also
  document them in `README.md` (INPUT format section) and in
  `docs/OVERVIEW.md`.
- Performance-critical paths in `Lattice.cpp` and the MC loop in
  `main.cpp::empty_nests_fast`: maintain the delta-update invariants
  carefully. Adding state? Update it on accept AND undo on reject.

### Python (`tools/`)

- Standard `pip install -r tools/requirements.txt` runtime.
- Type hints encouraged but not required.
- One CLI tool per module, with a `parse_args()` helper that returns
  an `argparse.Namespace`. See `suggest_cell.py` / `run_sqs.py` /
  `verify_sro.py` for the pattern.
- Tests go in `tools/tests/test_<module>.py`. Use pytest
  `tmp_path` for scratch directories — never write into the
  repo tree from a test.
- When a test runs the scraps binary, gate it on
  `SCRAPS_BIN.exists()` and resolve `mpirun` via the helper at
  the top of `tools/tests/test_scraps_optimization.py`.

## Reproducibility

If a regression test depends on the optimizer's exact behavior
(rather than just "fitness ≤ X"), pin `RNG_SEED` in the INPUT and
include the seed in a comment near the assertion. Without a fixed
seed, the test will be flaky across machines and Python versions.

For ablations / benchmarks, use the `RANDOM_ONLY`, `FAST_SWAPS`, and
`TARGETED_SWAPS` keywords with a fixed `RNG_SEED` to compare
configurations on the same initial random pool.

## Where to start

If you're new to the project:

1. Read `README.md` (5-min overview).
2. Skim `docs/OVERVIEW.md` (10-min technical map).
3. Walk through `notebooks/01_first_scraps_run.ipynb` end to end.
4. Check the [issues list](https://github.com/rarroyave/scraps-perpair/issues)
   for `good-first-issue` tags.

Questions are welcome — open an issue with the `question` label and
describe what you're trying to do.

## Maintainer

The fork is maintained by Raymundo Arroyave
(<raymundo.arroyave@gmail.com>). The upstream code is by Singh, Singh,
Sharma, Balasubramanian, and Johnson; see citation in
`docs/OVERVIEW.md`.
