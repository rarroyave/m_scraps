#!/usr/bin/env python3
"""High-throughput composition-space SQS sweeps for scraps-perpair.

Generates many SQS at once across a composition grid. Reuses the
scraps_wizard pipeline (template + composition + AUTO_BUDGET → SQS)
for each composition, aggregates results into a single manifest, and
optionally runs in parallel.

Three sweep types:

    linear   Binary alloy along composition axis (Ni 0% → 100% Co).
    simplex  N-ary alloy on the Gibbs simplex at resolution k.
    list     Explicit composition list from a CSV or JSON file.

Output layout (`--output-dir sweep_out`):

    sweep_out/
    ├── manifest.json               # summary across all compositions
    ├── Ni1.00_Co0.00/              # one subdir per composition
    │   ├── POSCAR INPUT SCRAPS.vasp scraps.log sqs.cif
    │   └── meta.json
    └── ...

manifest.json schema:

    {
      "sweep_type": "linear" | "simplex" | "list",
      "template": "fcc",
      "n_compositions": 11,
      "target_atoms": 100,
      "results": [
        {
          "composition": {"Ni": 1.0, "Co": 0.0},
          "tag": "Ni1.00_Co0.00",
          "cell_dim": 3,
          "n_atoms": 108,
          "fitness": 0.0,
          "max_alpha": 0.0,
          "wall_seconds": 1.4,
          "status": "success" | "skipped" | "failed",
          "error": null | "...",
        },
        ...
      ]
    }

Examples:

    # Binary sweep at 10% resolution
    scraps_htp.py linear --template fcc --species Ni,Co \\
        --range 0:1 --step 0.1 --target-atoms 100 --output-dir nico_sweep

    # Ternary simplex (NiCoCr) at 20% resolution
    scraps_htp.py simplex --template fcc --species Ni,Co,Cr \\
        --resolution 5 --target-atoms 100 --output-dir nicocr_sweep

    # 4 parallel workers on a multi-core box (default 4 MPI ranks each):
    scraps_htp.py simplex --template fcc --species Ni,Co,Cr \\
        --resolution 10 --workers 4 --ranks-per-worker 2

    # Resume an interrupted sweep (existing outputs are skipped):
    scraps_htp.py simplex --template fcc --species Ni,Co,Cr \\
        --resolution 10 --output-dir nicocr_sweep
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))


# ----- composition enumeration --------------------------------------------

def enumerate_simplex(n_species: int, resolution: int):
    """Yield every integer composition on the n-simplex with `resolution`
    steps -- each output is an n-tuple of non-negative integers summing
    to `resolution`. Total count = C(resolution + n - 1, n - 1).

    Example: n=3, resolution=5 -> 21 compositions including (5,0,0),
    (4,1,0), (3,2,0), (3,1,1), (2,2,1), (2,1,2), ..., (0,0,5).

    resolution=0 is valid (yields a single all-zeros tuple); this is the
    natural base case for the recursive expansion.
    """
    if n_species < 1:
        raise ValueError(f"n_species must be >= 1, got {n_species}")
    if resolution < 0:
        raise ValueError(f"resolution must be >= 0, got {resolution}")
    if n_species == 1:
        yield (resolution,)
        return
    for first in range(resolution + 1):
        for rest in enumerate_simplex(n_species - 1, resolution - first):
            yield (first,) + rest


def linear_compositions(species: list[str],
                        lo: float, hi: float,
                        step: float) -> list[dict[str, float]]:
    """Binary alloy along composition axis: species[0] from `lo` to `hi`
    in increments of `step`, species[1] = 1 - species[0]. Returns a list
    of {species: fraction} dicts."""
    if len(species) != 2:
        raise ValueError(f"linear sweep requires exactly 2 species, "
                         f"got {len(species)}: {species}")
    out = []
    n_steps = int(round((hi - lo) / step))
    for i in range(n_steps + 1):
        x = lo + i * step
        if x < -1e-9 or x > 1 + 1e-9:
            continue
        x = max(0.0, min(1.0, x))
        out.append({species[0]: x, species[1]: 1.0 - x})
    return out


def simplex_compositions(species: list[str],
                         resolution: int) -> list[dict[str, float]]:
    """N-ary on the Gibbs simplex at integer resolution."""
    if resolution < 1:
        raise ValueError(f"simplex resolution must be >= 1, got {resolution}")
    return [
        {sp: count / resolution for sp, count in zip(species, comp)}
        for comp in enumerate_simplex(len(species), resolution)
    ]


def list_compositions_from_csv(path: Path) -> list[dict[str, float]]:
    """Read a CSV where the header row is species symbols and each
    subsequent row is one composition (fractions summing to 1)."""
    out = []
    with path.open() as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            fracs = {sp: float(v) for sp, v in zip(header, row) if v.strip()}
            out.append(fracs)
    return out


def list_compositions_from_json(path: Path) -> list[dict[str, float]]:
    """Read a JSON file containing a list of {species: fraction} dicts."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level must be a list of "
                         f"composition dicts")
    return [{k: float(v) for k, v in d.items()} for d in data]


def composition_tag(fracs: dict[str, float], precision: int = 2) -> str:
    """Stable, filesystem-safe tag for a composition. E.g.:
        {"Ni": 0.5, "Co": 0.5} -> "Ni0.50_Co0.50"
    Preserves species order from the input dict (Python 3.7+ dict order)."""
    fmt = f".{precision}f"
    return "_".join(f"{sp}{frac:{fmt}}" for sp, frac in fracs.items())


# ----- per-composition runner ---------------------------------------------

@dataclass
class CompositionResult:
    composition: dict[str, float]
    tag: str
    cell_dim: int | None = None
    n_atoms: int | None = None
    fitness: float | None = None
    # max_alpha == max_alpha_cross (off-diagonal: cross-species). Kept
    # as the legacy field for backward compatibility; new code should
    # prefer max_alpha_cross + max_alpha_self for the split metric.
    # max_alpha_self can exceed +/-1 for asymmetric compositions
    # (e.g. Nb:0.1,Mo:0.1,Ta:0.8 at 128 atoms) by composition
    # normalization without indicating optimizer failure; the
    # cross-species metric is the SQS-quality verdict.
    max_alpha: float | None = None
    max_alpha_cross: float | None = None
    max_alpha_self: float | None = None
    wall_seconds: float | None = None
    status: str = "pending"          # success | skipped | failed
    error: str | None = None
    output_dir: str | None = None


def run_one_composition(template_name: str,
                        fixed_compositions: list[dict[str, float]],
                        swept_composition: dict[str, float],
                        swept_orbit_index: int,
                        target_atoms: int,
                        output_dir: Path,
                        ranks: int,
                        budget: int = 2,
                        resume: bool = True) -> CompositionResult:
    """Run one composition through the wizard pipeline. Returns a
    CompositionResult; never raises (failures are caught and reported in
    .status = 'failed').

    `fixed_compositions` is the list of compositions for non-swept orbits
    (in template orbit order, skipping swept_orbit_index).
    `swept_composition` is the composition for the swept orbit.
    """
    import scraps_wizard

    if template_name not in scraps_wizard.TEMPLATES:
        return CompositionResult(
            composition=swept_composition,
            tag=composition_tag(swept_composition),
            status="failed",
            error=f"Unknown template {template_name!r}",
        )
    template = scraps_wizard.TEMPLATES[template_name]

    # Assemble the per-orbit composition list in template order.
    per_orbit = list(fixed_compositions)
    per_orbit.insert(swept_orbit_index, swept_composition)
    if len(per_orbit) != len(template.orbits):
        return CompositionResult(
            composition=swept_composition,
            tag=composition_tag(swept_composition),
            status="failed",
            error=f"Template {template_name} has {len(template.orbits)} "
                  f"orbits but got {len(per_orbit)} compositions",
        )

    tag = composition_tag(swept_composition)
    work_dir = output_dir / tag
    result = CompositionResult(
        composition=dict(swept_composition),
        tag=tag,
        output_dir=str(work_dir.relative_to(output_dir.parent)),
    )

    if resume and (work_dir / "sqs.cif").exists():
        # Already done in a previous run.
        meta_path = work_dir / "meta.json"
        if meta_path.exists():
            try:
                prior = json.loads(meta_path.read_text())
                result.cell_dim = prior.get("cell_dim")
                result.n_atoms = prior.get("n_atoms")
                result.fitness = prior.get("fitness")
                result.max_alpha = prior.get("max_alpha")
                result.max_alpha_cross = prior.get("max_alpha_cross",
                                                   prior.get("max_alpha"))
                result.max_alpha_self = prior.get("max_alpha_self")
                result.wall_seconds = prior.get("wall_seconds")
                result.status = "skipped"
                return result
            except Exception:
                pass

    t0 = time.time()
    try:
        wr = scraps_wizard.run_pipeline(
            template, per_orbit,
            target_atoms=target_atoms,
            work_dir=work_dir, ranks=ranks,
            output_format="cif", budget=budget, quiet=True,
        )
    except Exception as e:
        result.status = "failed"
        result.error = repr(e)
        result.wall_seconds = round(time.time() - t0, 2)
        return result

    result.cell_dim = wr.cell_dim
    result.n_atoms = wr.n_atoms
    result.fitness = wr.fitness
    result.wall_seconds = round(time.time() - t0, 2)
    result.status = "success"

    # Compute max|alpha| via verify_sro for single-sublattice cases
    # (multi-sublattice uses the per-sublattice calculator).
    try:
        if len(template.orbits) == 1:
            # Filter to species with positive count -- on simplex-edge
            # compositions (binaries, ternaries within a quaternary
            # sweep) some species have count 0 and SCRAPs writes them
            # out of the POSCAR. verify_sro would reject a longer
            # --species list than the POSCAR's species-block count.
            present = [sp for sp, c in swept_composition.items() if c > 0]
            if len(present) >= 2:
                verify = REPO_ROOT / "tools" / "verify_sro.py"
                proc = subprocess.run(
                    [sys.executable, str(verify),
                     str(work_dir / "SCRAPS.vasp"),
                     "--species", ",".join(present),
                     "--shells", "2", "--json"],
                    capture_output=True, text=True, timeout=60,
                )
                if proc.returncode in (0, 1):
                    data = json.loads(proc.stdout)
                    cross_vals: list[float] = []
                    self_vals: list[float] = []
                    for sh in data.get("shells", []):
                        amat = sh.get("alpha_matrix", [])
                        for i, row in enumerate(amat):
                            for j, v in enumerate(row):
                                if not isinstance(v, (int, float)):
                                    continue
                                (self_vals if i == j else cross_vals).append(abs(v))
                    if cross_vals:
                        result.max_alpha_cross = max(cross_vals)
                        result.max_alpha = result.max_alpha_cross
                    if self_vals:
                        result.max_alpha_self = max(self_vals)
            # Pure compositions (1 present species) have no SRO defined;
            # leave result.max_alpha = None to signal that.
    except Exception:
        pass

    (work_dir / "meta.json").write_text(json.dumps({
        "composition": result.composition,
        "tag": result.tag,
        "cell_dim": result.cell_dim,
        "n_atoms": result.n_atoms,
        "fitness": result.fitness,
        "max_alpha": result.max_alpha,
        "max_alpha_cross": result.max_alpha_cross,
        "max_alpha_self": result.max_alpha_self,
        "wall_seconds": result.wall_seconds,
    }, indent=2))
    return result


# ----- sweep orchestrator -------------------------------------------------

def run_sweep(compositions: list[dict[str, float]],
              template_name: str,
              fixed_compositions: list[dict[str, float]],
              swept_orbit_index: int,
              target_atoms: int,
              output_dir: Path,
              ranks_per_worker: int,
              workers: int,
              budget: int = 2,
              resume: bool = True) -> list[CompositionResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[scraps-htp] {len(compositions)} compositions, "
          f"template={template_name}, target_atoms={target_atoms}, "
          f"workers={workers}, ranks/worker={ranks_per_worker}",
          file=sys.stderr)
    if resume:
        existing = sum(1 for c in compositions
                       if (output_dir / composition_tag(c) / "sqs.cif").exists())
        if existing:
            print(f"[scraps-htp] {existing} composition(s) already done "
                  f"(resuming)", file=sys.stderr)

    results: list[CompositionResult] = []
    if workers <= 1:
        for i, comp in enumerate(compositions, 1):
            tag = composition_tag(comp)
            print(f"[scraps-htp] [{i:>3}/{len(compositions)}] {tag} ... ",
                  end="", flush=True, file=sys.stderr)
            r = run_one_composition(
                template_name, fixed_compositions, comp,
                swept_orbit_index, target_atoms,
                output_dir, ranks_per_worker, budget, resume,
            )
            results.append(r)
            if r.status == "success":
                msg = f"ok ({r.n_atoms} atoms, "
                if r.max_alpha_cross is not None:
                    msg += f"|α|x={r.max_alpha_cross:.4f}"
                    if r.max_alpha_self is not None and r.max_alpha_self > 1.0:
                        msg += f" [self={r.max_alpha_self:.2f}, asym]"
                    msg += ", "
                msg += f"{r.wall_seconds:.1f}s)"
            elif r.status == "skipped":
                msg = "skipped (already done)"
            else:
                msg = f"FAILED ({r.error})"
            print(msg, file=sys.stderr)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_comp = {
                ex.submit(run_one_composition,
                          template_name, fixed_compositions, comp,
                          swept_orbit_index, target_atoms,
                          output_dir, ranks_per_worker, budget, resume): comp
                for comp in compositions
            }
            for i, fut in enumerate(
                concurrent.futures.as_completed(future_to_comp), 1
            ):
                r = fut.result()
                results.append(r)
                tag = r.tag
                if r.status == "success":
                    msg = f"ok ({r.n_atoms} atoms, "
                    if r.max_alpha is not None:
                        msg += f"max|α|={r.max_alpha:.4f}, "
                    msg += f"{r.wall_seconds:.1f}s)"
                elif r.status == "skipped":
                    msg = "skipped"
                else:
                    msg = f"FAILED ({r.error})"
                print(f"[scraps-htp] [{i:>3}/{len(compositions)}] {tag} "
                      f"... {msg}", file=sys.stderr)
    # Sort by composition tag for stable output.
    results.sort(key=lambda r: r.tag)
    return results


def write_manifest(output_dir: Path, results: list[CompositionResult],
                   sweep_type: str, template_name: str,
                   target_atoms: int) -> Path:
    n_success = sum(1 for r in results if r.status == "success")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_failed = sum(1 for r in results if r.status == "failed")
    manifest = {
        "sweep_type": sweep_type,
        "template": template_name,
        "n_compositions": len(results),
        "n_success": n_success,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "target_atoms": target_atoms,
        "results": [asdict(r) for r in results],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[scraps-htp] {n_success} success, {n_skipped} skipped, "
          f"{n_failed} failed", file=sys.stderr)
    print(f"[scraps-htp] Manifest -> {path}", file=sys.stderr)
    return path


# ----- subcommands ---------------------------------------------------------

def parse_fixed_compositions(args, template_orbits) -> tuple[
    int, list[dict[str, float]]
]:
    """Return (swept_orbit_index, fixed_compositions_in_template_order).
    `args.fixed` is a list of "ORBIT_INDEX:Species:Frac,..." strings.
    """
    n_orbits = len(template_orbits)
    if n_orbits == 1:
        return 0, []
    if not args.fixed:
        raise SystemExit(
            f"Template has {n_orbits} sublattices; you must specify "
            f"--fixed for all non-swept orbits. Example: "
            f"--fixed 1:O:1.0 --fixed 2:O:1.0 (orbit 0 will be swept)."
        )
    import scraps_wizard
    fixed: dict[int, dict[str, float]] = {}
    for s in args.fixed:
        if ":" not in s:
            raise SystemExit(f"--fixed {s!r}: expected ORBIT_INDEX:COMP")
        idx_s, comp = s.split(":", 1)
        try:
            idx = int(idx_s)
        except ValueError:
            raise SystemExit(f"--fixed {s!r}: orbit index must be int")
        try:
            fixed[idx] = scraps_wizard.parse_composition(comp)
        except ValueError as e:
            raise SystemExit(f"--fixed {s!r}: {e}")
    swept_idx_candidates = [i for i in range(n_orbits) if i not in fixed]
    if len(swept_idx_candidates) != 1:
        raise SystemExit(
            f"Exactly one orbit must be unfixed (swept). Template has "
            f"{n_orbits} orbits; --fixed covers {sorted(fixed.keys())}; "
            f"unfixed = {swept_idx_candidates}."
        )
    swept_idx = swept_idx_candidates[0]
    ordered_fixed = [fixed[i] for i in range(n_orbits) if i != swept_idx]
    return swept_idx, ordered_fixed


def cmd_linear(args) -> int:
    species = [s.strip() for s in args.species.split(",")]
    if len(species) != 2:
        print("ERROR: linear sweep needs exactly 2 species "
              "(comma-separated)", file=sys.stderr)
        return 2
    try:
        lo_s, hi_s = args.range.split(":")
        lo, hi = float(lo_s), float(hi_s)
    except (ValueError, AttributeError):
        print(f"ERROR: --range {args.range!r} must be LOW:HIGH "
              "(e.g. 0:1)", file=sys.stderr)
        return 2
    comps = linear_compositions(species, lo, hi, args.step)
    return _dispatch_sweep("linear", args, comps)


def cmd_simplex(args) -> int:
    species = [s.strip() for s in args.species.split(",")]
    if len(species) < 2:
        print("ERROR: simplex sweep needs >= 2 species", file=sys.stderr)
        return 2
    if args.resolution < 1:
        print(f"ERROR: --resolution must be >= 1", file=sys.stderr)
        return 2
    n_sp = len(species)
    # Equiatomic (each species at 1/n_sp) lives on the simplex grid only
    # when resolution % n_species == 0. Warn the user otherwise so they
    # don't unknowingly miss the chemistry-central point of an HEA sweep.
    if args.resolution % n_sp != 0:
        nearest_lower = (args.resolution // n_sp) * n_sp
        nearest_upper = nearest_lower + n_sp
        suggested = nearest_upper if nearest_lower == 0 else (
            nearest_lower if (args.resolution - nearest_lower)
                              < (nearest_upper - args.resolution)
            else nearest_upper
        )
        print(f"[scraps-htp] NOTE: resolution {args.resolution} does not "
              f"divide n_species ({n_sp}); the equiatomic composition "
              f"(each species at 1/{n_sp}) is NOT on the simplex grid. "
              f"Use --resolution {suggested} (or any multiple of "
              f"{n_sp}) to include it.",
              file=sys.stderr)
    comps = simplex_compositions(species, args.resolution)
    return _dispatch_sweep("simplex", args, comps)


def cmd_list(args) -> int:
    path = Path(args.compositions)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2
    if path.suffix.lower() == ".csv":
        comps = list_compositions_from_csv(path)
    elif path.suffix.lower() == ".json":
        comps = list_compositions_from_json(path)
    else:
        print(f"ERROR: --compositions must be .csv or .json "
              f"(got {path.suffix})", file=sys.stderr)
        return 2
    if not comps:
        print(f"ERROR: no compositions found in {path}", file=sys.stderr)
        return 2
    return _dispatch_sweep("list", args, comps)


def _dispatch_sweep(sweep_type: str, args,
                    compositions: list[dict[str, float]]) -> int:
    import scraps_wizard
    if args.template not in scraps_wizard.TEMPLATES:
        print(f"ERROR: unknown template {args.template!r}", file=sys.stderr)
        return 2
    template = scraps_wizard.TEMPLATES[args.template]
    swept_idx, fixed = parse_fixed_compositions(args, template.orbits)

    output_dir = Path(args.output_dir).resolve()
    results = run_sweep(
        compositions=compositions,
        template_name=args.template,
        fixed_compositions=fixed,
        swept_orbit_index=swept_idx,
        target_atoms=args.target_atoms,
        output_dir=output_dir,
        ranks_per_worker=args.ranks_per_worker,
        workers=args.workers,
        budget=args.budget,
        resume=not args.no_resume,
    )
    write_manifest(output_dir, results, sweep_type, args.template,
                   args.target_atoms)
    return 0 if all(r.status != "failed" for r in results) else 1


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scraps-htp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--template", required=True,
                        help="Structure template name (see scraps_wizard "
                             "list-templates)")
        sp.add_argument("--target-atoms", type=int, default=100,
                        help="Target supercell size (default 100)")
        sp.add_argument("--output-dir", default="./sweep_out",
                        help="Output directory (default ./sweep_out)")
        sp.add_argument("--ranks-per-worker", type=int, default=4,
                        help="MPI ranks per SCRAPs invocation (default 4)")
        sp.add_argument("--workers", type=int, default=1,
                        help="Parallel composition workers (default 1). "
                             "Each worker spawns its own SCRAPs process "
                             "with --ranks-per-worker MPI ranks, so "
                             "workers x ranks-per-worker is the total "
                             "core budget.")
        sp.add_argument("--budget", type=int, default=2,
                        choices=[1, 2, 3],
                        help="AUTO_BUDGET MC profile (default 2 = thorough)")
        sp.add_argument("--no-resume", action="store_true",
                        help="Don't skip compositions whose sqs.cif "
                             "already exists; re-run everything.")
        sp.add_argument("--fixed", action="append", default=[],
                        help="For multi-sublattice templates: pin one "
                             "orbit at a fixed composition. Format: "
                             "ORBIT_INDEX:Species:Frac,Species:Frac. "
                             "Repeat for every orbit EXCEPT the one "
                             "being swept. Example: --fixed 1:O:1.0 "
                             "(orbit 1 is pure oxygen).")

    sp = sub.add_parser("linear",
                        help="Binary sweep along composition axis")
    sp.add_argument("--species", required=True,
                    help="Comma-separated species (exactly 2)")
    sp.add_argument("--range", required=True,
                    help="LOW:HIGH fraction range for species[0] "
                         "(e.g. 0:1)")
    sp.add_argument("--step", type=float, default=0.1,
                    help="Step size (default 0.1)")
    add_common(sp)
    sp.set_defaults(func=cmd_linear)

    sp = sub.add_parser("simplex",
                        help="N-ary on the Gibbs simplex at integer resolution")
    sp.add_argument("--species", required=True,
                    help="Comma-separated species (>= 2)")
    sp.add_argument("--resolution", type=int, default=5,
                    help="Number of integer steps along each simplex "
                         "axis (default 5; for ternary gives 21 "
                         "compositions)")
    add_common(sp)
    sp.set_defaults(func=cmd_simplex)

    sp = sub.add_parser("list",
                        help="Explicit composition list from CSV or JSON")
    sp.add_argument("--compositions", required=True,
                    help="Path to a .csv (header row = species, each "
                         "row = composition) or .json (list of "
                         "{species: fraction} dicts)")
    add_common(sp)
    sp.set_defaults(func=cmd_list)

    return p


def main() -> int:
    args = make_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
