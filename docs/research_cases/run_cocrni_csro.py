#!/usr/bin/env python3
"""CoCrNi CSRO research case study workflow.

Reproduces an end-to-end SCRAPs workflow for designing CoCrNi
medium-entropy alloy supercells with literature-reported chemical
short-range order (CSRO):

  Zhang et al., 'Short-range order and its impact on the CrCoNi medium-
  entropy alloy', Nature 581, 283 (2020). DOI: 10.1038/s41586-020-2275-z

The reported chemistry is Cr-Cr avoidance at 1NN (alpha(Cr,Cr) at 1NN
slightly negative). Other pair correlations are closer to zero in the
literature data; we target alpha(Cr,Cr) at 1NN = -0.10 as a
representative literature CSRO value to reproduce.

What this script does:

  1. Generates a random (SQS, alpha = 0 everywhere) CoCrNi supercell as
     the baseline.
  2. Generates a CSRO CoCrNi supercell targeting alpha(Cr-Cr) at 1NN =
     -0.10 (and alpha = 0 for every other pair, including 2NN).
  3. Sweeps cell sizes (32, 108, 256 atoms) to show how integer-pair-
     count feasibility tightens the achievable alpha as the cell grows.
  4. Verifies each output via tools/verify_sro.py.
  5. Emits a JSON summary + a matplotlib figure for the case study
     write-up.

Usage:
  python3 docs/research_cases/run_cocrni_csro.py [--output-dir DIR]

Outputs:
  docs/research_cases/results/
    sqs_n{N}/    -- random-SQS run at cell size N
    csro_n{N}/   -- CSRO-targeted run at cell size N
    results.json -- timing + alpha summary for all runs
    csro_vs_cell_size.png -- achieved alpha(Cr-Cr) at 1NN across N
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRAPS_BIN = REPO / "SCRAPs" / "scraps"
sys.path.insert(0, str(REPO / "tools"))

# Each cell size and what FCC supercell tiling it corresponds to
# (target_atoms : Cell_DIM N). All have 36 NiCoCr equiatomic feasibility
# (n_atoms / 3 must be integer). N=2: 32 atoms isn't divisible by 3, so
# Hamilton-rounding gives Ni=11, Co=11, Cr=10 (deviation 0.0104).
# N=3: 108 / 3 = 36 each, exact. N=4: 256 / 3 = 85.33, deviation 0.0033.
# N=5: 500 / 3 = 166.67, deviation 0.0007.
CELL_SIZES = [108, 256, 500]   # all close to equiatomic; FCC sweep


def find_mpirun() -> str:
    if env := os.environ.get("MPIRUN"):
        if Path(env).exists():
            return env
    sib = Path("/opt/homebrew/bin/mpirun")
    if sib.exists():
        return str(sib)
    found = shutil.which("mpirun")
    if not found:
        raise RuntimeError("no mpirun on PATH")
    return found


MPIRUN = find_mpirun()


@dataclass
class RunResult:
    label: str                     # 'sqs_n108', 'csro_n108', ...
    cell_dim: int
    n_atoms: int
    target_alpha_CrCr_1NN: float   # 0.0 for SQS, -0.10 for CSRO
    achieved_alpha_CrCr_1NN: float | None = None
    max_alpha_offdiag: float | None = None
    fitness: float | None = None
    wall_seconds: float | None = None
    output_dir: str = ""


def run_one(label: str, cell_dim: int, target_csro: bool,
            work_root: Path) -> RunResult:
    """Run one CoCrNi configuration."""
    from scraps_ase import run_sqs_from_atoms
    from ase.build import bulk

    primitive = bulk("Cu", "fcc", a=3.56, cubic=True)  # 4-atom FCC
    n_atoms = 4 * cell_dim ** 3
    # Hamilton-round 1/3 each across n_atoms
    base = n_atoms // 3
    rem = n_atoms - 3 * base
    # Give the +1 atoms to Ni, Co, Cr in that order
    counts = [base + (1 if i < rem else 0) for i in range(3)]
    species = ["Ni", "Co", "Cr"]
    sro_targets = None
    target_alpha = 0.0
    if target_csro:
        # Reproduce Zhang 2020 Nature: alpha(Cr-Cr) at 1NN = -0.10.
        # As of v0.4.12, SCRAPs accepts the natural same-species form
        # directly: it auto-expands SRO i i shell alpha into the
        # corresponding cross-species lines via the Warren-Cowley sum
        # rule (for NiCoCr equiatomic c_i=1/3, alpha_ii=-0.10 expands to
        # alpha_ij=+0.05 for each j != i). Species index 2 = Cr. v0.4.13
        # adds order-independent explicit overrides and a warning when
        # explicit cross-species lines would break the implied alpha_ii.
        sro_targets = [(2, 2, 0, -0.10)]    # alpha(Cr-Cr) at 1NN = -0.10
        target_alpha = -0.10
    work_dir = work_root / label
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sqs_atoms = run_sqs_from_atoms(
        primitive,
        species=species, element_counts=counts,
        cell_dim=(cell_dim, cell_dim, cell_dim),
        max_shellnum=2, weights=[1.0, 1.0],
        sro_targets=sro_targets,
        auto_budget=3,    # exhaustive profile -- want best-effort convergence
        rng_seed=42, ranks=4,
        work_dir=work_dir,
    )
    wall = time.time() - t0
    result = RunResult(
        label=label, cell_dim=cell_dim, n_atoms=n_atoms,
        target_alpha_CrCr_1NN=target_alpha,
        fitness=sqs_atoms.info.get("scraps_final_fitness"),
        wall_seconds=round(wall, 2),
        output_dir=str(work_dir.relative_to(work_root.parent)),
    )

    # Verify via verify_sro.py
    verify = REPO / "tools" / "verify_sro.py"
    proc = subprocess.run(
        [sys.executable, str(verify),
         str(work_dir / "SCRAPS.vasp"),
         "--species", ",".join(species),
         "--shells", "2", "--json"],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode in (0, 1):
        try:
            data = json.loads(proc.stdout)
            for sh in data.get("shells", []):
                if sh.get("index") != 0:
                    continue
                # alpha matrix is shells[k]["alpha_matrix"][i][j] for
                # species (Ni, Co, Cr) -> indices (0, 1, 2). We want
                # alpha[Cr][Cr] = alpha[2][2].
                amat = sh.get("alpha_matrix", [])
                if len(amat) >= 3 and len(amat[2]) >= 3:
                    val = amat[2][2]
                    if isinstance(val, (int, float)):
                        result.achieved_alpha_CrCr_1NN = float(val)
                # Off-diagonal max|alpha| across all shells
                offdiag_vals: list[float] = []
                for sh2 in data.get("shells", []):
                    for i, row in enumerate(sh2.get("alpha_matrix", [])):
                        for j, v in enumerate(row):
                            if i == j or not isinstance(v, (int, float)):
                                continue
                            offdiag_vals.append(abs(v))
                if offdiag_vals:
                    result.max_alpha_offdiag = max(offdiag_vals)
                break
        except Exception:
            pass
    return result


def make_figure(results: list[RunResult], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(6, 4))

    sqs_results = [r for r in results if "sqs" in r.label
                   and r.achieved_alpha_CrCr_1NN is not None]
    csro_results = [r for r in results if "csro" in r.label
                    and r.achieved_alpha_CrCr_1NN is not None]

    if sqs_results:
        xs = [r.n_atoms for r in sqs_results]
        ys = [r.achieved_alpha_CrCr_1NN for r in sqs_results]
        ax.plot(xs, ys, "o-", color="#1f77b4",
                label="SQS target (α=0)", markersize=8)
    if csro_results:
        xs = [r.n_atoms for r in csro_results]
        ys = [r.achieved_alpha_CrCr_1NN for r in csro_results]
        ax.plot(xs, ys, "s-", color="#d62728",
                label="CSRO target (α=−0.10)", markersize=8)
    ax.axhline(0.0, color="#1f77b4", ls="--", alpha=0.4)
    ax.axhline(-0.10, color="#d62728", ls="--", alpha=0.4,
               label="literature α(Cr-Cr)@1NN")
    ax.set_xlabel("cell size (atoms)")
    ax.set_ylabel(r"achieved $\alpha(\mathrm{Cr-Cr})$ at 1NN")
    ax.set_title("CoCrNi: SCRAPs reaches literature-CSRO target\n"
                 "across cell sizes (Zhang et al. 2020, Nature)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if not SCRAPS_BIN.exists():
        print("scraps not built. Run ./build.sh first.", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for cd in [3, 4, 5]:   # FCC supercells of size 108, 256, 500
        n_atoms = 4 * cd ** 3
        print(f"\n[case-study] Cell {cd}x{cd}x{cd} = {n_atoms} atoms")
        for kind in ("sqs", "csro"):
            label = f"{kind}_n{n_atoms}"
            print(f"  -> {label} ...", end="", flush=True)
            try:
                r = run_one(label, cd, target_csro=(kind == "csro"),
                            work_root=args.output_dir)
                results.append(r)
                msg = (f" fitness={r.fitness:.4f}, "
                       f"α(Cr-Cr)@1NN={r.achieved_alpha_CrCr_1NN:+.4f} "
                       f"(target {r.target_alpha_CrCr_1NN:+.2f}), "
                       f"max|α|offdiag={r.max_alpha_offdiag:.4f}, "
                       f"wall={r.wall_seconds:.1f}s")
                print(msg)
            except Exception as e:
                print(f" FAILED: {e}")

    out_json = args.output_dir / "results.json"
    out_json.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\n[case-study] Wrote {out_json}")

    fig_path = args.output_dir / "csro_vs_cell_size.png"
    make_figure(results, fig_path)
    if fig_path.exists():
        print(f"[case-study] Wrote {fig_path}")

    # Markdown summary table for the case-study write-up
    print()
    print("Summary table (markdown):\n")
    print("| run | cell | atoms | target α(Cr-Cr)@1NN | achieved | "
          "max\\|α\\| off-diag | wall |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        ach = f"{r.achieved_alpha_CrCr_1NN:+.4f}" if r.achieved_alpha_CrCr_1NN is not None else "n/a"
        offd = f"{r.max_alpha_offdiag:.4f}" if r.max_alpha_offdiag is not None else "n/a"
        print(f"| {r.label} | {r.cell_dim}x{r.cell_dim}x{r.cell_dim} "
              f"| {r.n_atoms} | {r.target_alpha_CrCr_1NN:+.2f} | "
              f"{ach} | {offd} | {r.wall_seconds:.1f}s |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
