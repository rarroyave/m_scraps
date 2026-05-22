# SCRAPs tutorial notebooks

Five Jupyter notebooks walking through the most common use cases. Each
notebook is self-contained, executes end-to-end, and ships with cell
outputs so you can preview without running. To re-run locally:

```bash
jupyter notebook notebooks/<file>.ipynb
# or to execute all in-place and refresh outputs:
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 \
    notebooks/*.ipynb
```

To regenerate the `.ipynb` files from source (the markdown + code cells
are programmatically defined in `build_notebooks.py`):

```bash
python3 notebooks/build_notebooks.py
```

## Pick by use case

| You want to design ... | Open this notebook | What it shows |
|---|---|---|
| A single-sublattice SQS (e.g. NiCoCr 108-atom FCC) | **01_first_scraps_run.ipynb** | Shortest path from ASE `Atoms` to SQS supercell via `run_sqs_from_atoms`, with the `AUTO_BUDGET` keyword (no manual MC tuning), independent α verification, and 2D visualization. |
| A single-sublattice cell with **non-zero α targets** from literature (CSRO) | **02_targeting_csro.ipynb** | The `sro_targets=` parameter takes a list of `(species_i, species_j, shell, alpha)` tuples; one of the fork's headline features. |
| A high-entropy ceramic with one disordered metal sublattice and a fixed spectator (boride, carbide, oxide, ...) | **03_heb_end_to_end.ipynb** | `SPECTATOR_BASIS_SPECIES` workflow + the `suggest_cell.py` feasibility helper. Walks through (Hf,Zr,Ti,Ta,Nb)B₂. |
| Two or more chemically-disordered sublattices (rocksalt HEA-carbides, full Heusler, double perovskites, spinels, fluorite-type oxides, sigma/Laves phases) | **04_multi_sublattice.ipynb** | The `SUBLATTICE` keyword (v0.4.1+) and per-Wyckoff non-uniform-coordination (v0.4.2+). Walks through B2 (Ni,Co)(Ti,Zr) SQS + CSRO, then demonstrates `sublattice_detect.py` auto-detection on a Heusler L2₁ POSCAR. |
| To diagnose a failed or unexpected run | **05_troubleshooting.ipynb** | Five common failure modes with diagnoses: integer-feasibility floors, CSRO not hitting target, multi-sublattice uniformity errors, SCRAPS-vs-`verify_sro` discrepancies, and SCRAPS-vs-mcsqs disagreement. Reference material, not sequential. |

## What you need to install

- `scraps` C++ binary at `SCRAPs/scraps` (run `./build.sh` from the
  repo root after installing MPI and Eigen).
- Python deps: `pip install -r tools/requirements.txt` (pymatgen,
  ASE, numpy, matplotlib, pytest).

## After running

Each notebook's `work_dir` (`nb1_first_run/`, `nb4_csro/`, etc.) is
gitignored — these are scratch run dirs that hold the POSCAR, INPUT,
SCRAPS.vasp output, and scraps.log for each run. Inspect them
directly to see what the wrapper wrote, or delete them between runs
for a clean slate.
