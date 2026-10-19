"""Exhaustive, capped pair-SRO optimization for fixed finite periodic cells."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import time

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list

from sro_bounds import KEYS, METRICS, sro_bounds
from structure_analysis import analyze_structure


def configuration_count(composition):
    """Count labeled-site decorations, without symmetry reduction."""
    total = 1
    for group in composition.values():
        remaining = group['sites']
        for count in group['counts'].values():
            total *= math.comb(remaining, count)
            remaining -= count
    return total


def occupations(labels, composition):
    """Yield each allowed fixed-composition decoration exactly once, lazily."""
    names = sorted(composition)
    groups = [(np.flatnonzero(labels == name).tolist(),
               [(symbol, count) for symbol, count in sorted(composition[name]['counts'].items()) if count])
              for name in names]
    current = [''] * len(labels)

    def assign_group(group):
        if group == len(groups):
            yield tuple(current)
            return
        positions, species = groups[group]

        def assign_species(remaining, index):
            symbol, count = species[index]
            if index == len(species) - 1:
                for site in remaining:
                    current[site] = symbol
                yield from assign_group(group + 1)
                return
            for selected in combinations(remaining, count):
                selected_set = set(selected)
                for site in selected:
                    current[site] = symbol
                yield from assign_species([site for site in remaining if site not in selected_set], index + 1)

        yield from assign_species(positions, 0)

    yield from assign_group(0)


def enumerate_sro(atoms, *, cutoff, targets, shells=2, sublattices=None,
                  reference=None, tolerance=0.2, shell_tolerance=1e-5,
                  metric='max_error', max_configurations=100000, max_seconds=None,
                  symmetry='auto', symprec=1e-5, stop_at_bound=True, progress=None):
    """Return (report, witness Atoms); certify by bound attainment or exhaustion.

    Uses the uniform-coordination pair-SRO conventions of sro_bounds. By default,
    verified sublattice/target-preserving site symmetries reduce the search to
    orbit representatives. symmetry="none" retains exhaustive reference mode. All
    comparisons use rational arithmetic (mean squared error for RMS). The
    structure returned uses reference geometry/order when reference is supplied.
    Exact bound attainment stops early unless stop_at_bound=False. A time
    limit or Ctrl-C without a proof returns best_found. The initial decoration supplies a witness even if
    the time limit expires during preparation. No resume state is produced.
    """
    start = time.monotonic()
    if symmetry not in ('auto', 'none'):
        raise ValueError('symmetry must be auto or none')
    if not math.isfinite(symprec) or symprec <= 0:
        raise ValueError('symprec must be positive and finite')
    if isinstance(max_configurations, bool) or not isinstance(max_configurations, int) or max_configurations < 1:
        raise ValueError('max_configurations must be a positive integer')
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be positive and finite')
    bound = sro_bounds(atoms, cutoff=cutoff, targets=targets, shells=shells,
        sublattices=sublattices, reference=reference, tolerance=tolerance,
        shell_tolerance=shell_tolerance, metric=metric)
    total = configuration_count(bound['composition'])
    if total > max_configurations and not (stop_at_bound and bound['result']['status'] == 'certified_minimum'):
        raise ValueError(f'{total} configurations exceed --max-configurations {max_configurations}; '
                         'choose a smaller cell or explicitly raise the cap')
    analysis = analyze_structure(atoms, cutoff=cutoff, shells=shells, sublattices=sublattices,
        reference=reference, tolerance=tolerance, shell_tolerance=shell_tolerance,
        sro_targets=targets)
    geometry = reference if reference is not None else atoms
    labels = np.array(sublattices['site_labels'] if sublattices else ['all'] * len(atoms), dtype=str)
    ii, jj, distances = neighbor_list('ijd', geometry, cutoff)
    rows = {tuple(row[k] for k in KEYS): row for row in analysis['sro']}
    prepared = []
    for entry, target in zip(bound['entries'], targets):
        key = tuple(target[k] for k in KEYS)
        row = rows[key]
        source, destination, _, a, b = key
        selected = ((labels[ii] == source) & (labels[jj] == destination)
                    & (distances >= row['distance_min']) & (distances <= row['distance_max']))
        prepared.append((ii[selected], jj[selected], a, b,
                         Fraction(entry['denominator_exact']), Fraction(str(target['target']))))

    def evaluate(symbols):
        symbols = np.asarray(symbols)
        counts, residuals = [], []
        for centers, neighbors, a, b, denominator, target in prepared:
            count = int(np.count_nonzero((symbols[centers] == a) & (symbols[neighbors] == b)))
            counts.append(count)
            residuals.append(1 - Fraction(count) / denominator - target)
        absolute = [abs(r) for r in residuals]
        metrics = dict(max_error=max(absolute),
                       rms_error=sum(r * r for r in residuals) / len(residuals),
                       mean_absolute_error=sum(absolute) / len(absolute))
        return metrics, counts, residuals

    best_symbols = tuple(np.array(atoms.get_chemical_symbols())[analysis['mapping']['input_indices']])
    best_metrics, best_counts, best_residuals = evaluate(best_symbols)
    best = (best_metrics, best_symbols, best_counts, best_residuals)
    def emit(evaluated):
        if progress is not None:
            value = best[0][metric]
            progress(dict(elapsed_seconds=time.monotonic() - start,
                          best_error=math.sqrt(value) if metric == 'rms_error' else float(value),
                          evaluated_configurations=evaluated))
    emit(0)
    exact_relaxed = Fraction(bound['metrics'][metric]['lower_bound_exact'])
    bound_met_initially = stop_at_bound and best_metrics[metric] == exact_relaxed
    permutations = np.arange(len(atoms))[None, :]
    if symmetry == 'auto' and total > 1 and not bound_met_initially:
        from enumeration_symmetry import site_permutations
        permutations = site_permutations(geometry, labels, prepared, symprec)
    if symmetry == 'auto' and len(permutations) > 1:
        from enumeration_symmetry import symmetry_occupations
        candidates = symmetry_occupations(labels, bound['composition'], permutations)
    else:
        candidates = ((candidate, 1) for candidate in occupations(labels, bound['composition']))
    if bound_met_initially:
        candidates = iter(())
    evaluated, covered, reason = 0, 0, 'bound_attained' if bound_met_initially else 'exhausted'
    try:
        for candidate, multiplicity in candidates:
            if max_seconds is not None and time.monotonic() - start >= max_seconds:
                reason = 'time_limit'
                break
            values, counts, residuals = evaluate(candidate)
            if values[metric] < best[0][metric]:
                best = (values, candidate, counts, residuals)
                emit(evaluated + 1)
            evaluated += 1
            covered += multiplicity
            if stop_at_bound and best[0][metric] == exact_relaxed:
                reason = 'bound_attained'
                break
    except KeyboardInterrupt:
        reason = 'interrupted'
    best_metrics, best_symbols, best_counts, best_residuals = best
    complete = covered == total
    if complete:
        reason = 'exhausted'
    elif reason == 'exhausted':
        raise RuntimeError('Enumeration count does not match the combinatorial count')
    exact_best = best_metrics[metric]
    exact_relaxed = Fraction(bound['metrics'][metric]['lower_bound_exact'])
    convert = math.sqrt if metric == 'rms_error' else float
    certified = complete or reason == 'bound_attained'
    certified_lower = exact_best if certified else exact_relaxed
    witness = geometry.copy()
    witness.set_chemical_symbols(best_symbols)
    entries = []
    for target, count, residual in zip(targets, best_counts, best_residuals):
        entries.append(dict(**{k: target[k] for k in KEYS}, target=float(target['target']),
            pair_count=count, alpha=float(residual + Fraction(str(target['target']))),
            residual=float(residual), residual_exact=str(residual)))
    result = dict(schema_version=3, objective='directed_pair_sro', method='symmetry_reduced_occupations' if symmetry == 'auto' else 'exhaustive_labeled_occupations',
        status='certified_minimum' if certified else 'best_found', complete=complete,
        certificate_basis='exhaustion' if complete else ('bound_attained' if certified else None),
        stop_at_bound=stop_at_bound,
        stop_reason=reason, selected_metric=metric, total_configurations=total,
        evaluated_configurations=evaluated, covered_configurations=covered,
        symmetry=dict(mode=symmetry, site_permutations=len(permutations), symprec=symprec,
                      policy='Preserve each sublattice label and each target directed-bond multigraph individually.'),
        initial_witness_evaluated=True,
        max_configurations=max_configurations, max_seconds=max_seconds,
        elapsed_seconds=time.monotonic() - start,
        best_error=convert(exact_best), best_error_exact=str(exact_best),
        exact_value_convention='squared error' if metric == 'rms_error' else 'error',
        lower_bound=convert(certified_lower), lower_bound_exact=str(certified_lower),
        relaxed_lower_bound=convert(exact_relaxed), relaxed_lower_bound_exact=str(exact_relaxed),
        optimality_gap=convert(exact_best) - convert(certified_lower),
        metrics_at_best={k: dict(value=math.sqrt(v) if k == 'rms_error' else float(v),
                                exact=str(v), exact_value_convention='squared error' if k == 'rms_error' else 'error')
                         for k, v in best_metrics.items()},
        witness=dict(symbols=list(best_symbols), site_labels=labels.tolist(), site_order='reference' if reference is not None else 'input',
                     geometry='reference' if reference is not None else 'input'),
        entries=entries, composition=bound['composition'], convention=bound['convention'],
        shell_coverage=bound['shell_coverage'],
        scope='Specified finite cell, geometry, compositions, sublattice constraints, explicit targets and selected metric only.',
        limitations=['Configuration cap applies to the full labeled-site count before symmetry reduction.',
                     'Symmetry never exchanges sublattice labels or chemical identities.',
                     'Uniform coordination within requested sublattice-pair shells is required.',
                     'Other reported metrics describe the selected witness, not independently optimized minima.',
                     'Pair-SRO error only; no decorated higher-order correlation certificate.',
                     'Bound attainment certifies the selected minimum without claiming exhaustive coverage.'])
    return result, witness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('structure', type=Path)
    parser.add_argument('--targets', required=True, type=Path)
    parser.add_argument('--cutoff', required=True, type=float)
    parser.add_argument('--shells', type=int, default=2)
    parser.add_argument('--sublattices', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--tolerance', type=float, default=0.2)
    parser.add_argument('--shell-tolerance', type=float, default=1e-5)
    parser.add_argument('--metric', choices=METRICS, default='max_error')
    parser.add_argument('--symmetry', choices=('auto', 'none'), default='auto',
                        help='Use objective-preserving symmetry reduction (default auto), or exhaustive reference mode')
    parser.add_argument('--exhaustive', action='store_true', help='Disable bound-attainment stopping to audit full coverage')
    parser.add_argument('--symprec', type=float, default=1e-5, help='Symmetry detection tolerance in angstrom')
    parser.add_argument('--max-configurations', type=int, default=100000)
    parser.add_argument('--max-seconds', type=float, help='Optional time limit including preparation')
    parser.add_argument('--output', required=True, type=Path, help='New directory for report and witness files')
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('Output directory already exists; choose a new path')
        report, witness = enumerate_sro(read(args.structure), cutoff=args.cutoff,
            targets=json.loads(args.targets.read_text()), shells=args.shells,
            sublattices=json.loads(args.sublattices.read_text()) if args.sublattices else None,
            reference=read(args.reference) if args.reference else None,
            tolerance=args.tolerance, shell_tolerance=args.shell_tolerance,
            metric=args.metric, max_configurations=args.max_configurations, max_seconds=args.max_seconds,
            symmetry=args.symmetry, symprec=args.symprec, stop_at_bound=not args.exhaustive)
        inputs = {'structure': args.structure, 'targets': args.targets,
                  'sublattices': args.sublattices, 'reference': args.reference}
        report['inputs'] = {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                            for name, path in inputs.items() if path is not None}
        args.output.mkdir(parents=True)
        write(args.output / 'best.extxyz', witness)
        write(args.output / 'best.cif', witness)
        write(args.output / 'best.POSCAR', witness, format='vasp', direct=True, sort=False)
        report['files'] = {name: dict(sha256=hashlib.sha256((args.output / name).read_bytes()).hexdigest())
                           for name in ('best.extxyz', 'best.cif', 'best.POSCAR')}
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f"{report['status']}: {report['selected_metric']}={report['best_error']:.10g}; "
              f"{report['evaluated_configurations']} representatives, "
              f"{report['covered_configurations']}/{report['total_configurations']} decorations covered; "
              f"{args.output / 'result.json'}")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
