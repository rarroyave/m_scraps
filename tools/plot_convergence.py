#!/usr/bin/env python3
"""Plot the optimizer trajectory recorded by SCRAPS' CONVERGENCE_LOG.

`CONVERGENCE_LOG <path>` in INPUT writes a per-iteration CSV with
columns `iteration, best_fitness, wall_seconds` (iteration `-1` is
the pre-MC random-pool baseline). This script reads one or more such
CSV files and produces a fitness-vs-iteration plot.

Use cases:
  - Diagnosing your own SCRAPS runs ("did the optimizer actually
    descend, or get stuck at iter 0?").
  - Comparing trajectories across `FAST_SWAPS`, `TARGETED_SWAPS`,
    `RNG_SEED`, or `INIT_NEST` variants.
  - Generating figures for papers (--style publication writes a
    minimalist PDF suitable for inclusion in JOSS or similar).

Usage:
  tools/plot_convergence.py path/to/conv.csv [path/to/conv2.csv ...]
  tools/plot_convergence.py *.csv --labels uniform targeted --output plot.pdf

When multiple files are given without --labels, the file basename is
used as the legend label.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", type=Path,
                   help="One or more CONVERGENCE_LOG CSV files. "
                        "If a path has a `.<pid>` suffix (from macOS singleton "
                        "mode), it's read directly; glob patterns are NOT "
                        "expanded by this script -- use your shell.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="One label per input file for the legend. "
                        "Default: file basename.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output image path (.png, .pdf, .svg). "
                        "Default: show interactively.")
    p.add_argument("--style", choices=["default", "publication"], default="default",
                   help="`publication` uses larger fonts, thinner grid, and "
                        "no title -- suitable for paper figures.")
    p.add_argument("--logy", action="store_true",
                   help="Log-scale the fitness axis (useful for runs that "
                        "drop by orders of magnitude).")
    p.add_argument("--title", type=str, default=None,
                   help="Plot title. Default: 'SCRAPS optimization trajectory'.")
    p.add_argument("--xaxis", choices=["iteration", "wall"], default="iteration",
                   help="Plot fitness against outer iteration (default) "
                        "or wall-clock seconds.")
    p.add_argument("--mark-baseline", action="store_true",
                   help="Draw a horizontal dashed line at the iteration=-1 "
                        "best (the pre-MC random-baseline fitness). Useful "
                        "for showing how much the MC improved over random "
                        "search.")
    return p.parse_args()


def read_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    """Return (iterations, fitnesses, wall_seconds) from a CONVERGENCE_LOG csv."""
    iters: list[int] = []
    fits: list[float] = []
    walls: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iters.append(int(row["iteration"]))
                fits.append(float(row["best_fitness"]))
                walls.append(float(row["wall_seconds"]))
            except (KeyError, ValueError) as e:
                print(f"  skipping malformed row in {path}: {row}: {e}",
                      file=sys.stderr)
    return iters, fits, walls


def main() -> int:
    args = parse_args()

    if args.labels is not None and len(args.labels) != len(args.logs):
        print(f"ERROR: --labels has {len(args.labels)} entries but "
              f"got {len(args.logs)} input files.", file=sys.stderr)
        return 2

    if args.style == "publication":
        plt.rcParams.update({
            "font.size": 12,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.4,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
        })

    fig, ax = plt.subplots(figsize=(6, 4))
    baseline = None
    for i, path in enumerate(args.logs):
        if not path.exists():
            print(f"WARNING: {path} does not exist, skipping.", file=sys.stderr)
            continue
        iters, fits, walls = read_log(path)
        if not iters:
            print(f"WARNING: {path} has no parseable rows, skipping.",
                  file=sys.stderr)
            continue
        label = args.labels[i] if args.labels else path.stem
        x = walls if args.xaxis == "wall" else iters
        ax.plot(x, fits, marker="o", markersize=3, label=label)
        # Capture iteration=-1 baseline from the first log
        if baseline is None and -1 in iters:
            baseline = fits[iters.index(-1)]

    if args.mark_baseline and baseline is not None:
        ax.axhline(baseline, color="0.5", linestyle="--", linewidth=0.8,
                   label=f"random baseline ({baseline:.3g})")

    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("wall-clock seconds" if args.xaxis == "wall" else "outer iteration")
    ax.set_ylabel("best fitness (√Σ residual²)")
    ax.grid(True, which="both")
    if len(args.logs) > 1 or args.mark_baseline:
        ax.legend(loc="upper right")
    if args.style != "publication":
        title = args.title or "SCRAPS optimization trajectory"
        ax.set_title(title)

    fig.tight_layout()
    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
