"""Bounded, symmetry-distinct configuration ensembles for fixed-cell SRO targets."""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
from ase.io import read, write

from atat_reference import figure_model
from enumerate_sro import configuration_count
from enumeration_symmetry import site_permutations, symmetry_occupations
from search_cells import _anneal
from sro_bounds import KEYS, METRICS
from sro_stopping import SROBoundStop
from structure_analysis import analyze_structure


def _exact_metrics(symbols, prepared):
    residuals = [abs(1 - Fraction(int(np.count_nonzero((symbols[i] == a) & (symbols[j] == b)))) / d - t)
                 for i, j, a, b, d, t in prepared]
    return dict(max_error=max(residuals), mean_absolute_error=sum(residuals) / len(residuals),
                rms_error=sum(e * e for e in residuals) / len(residuals))


def _select(vectors, errors, count):
    """Lowest-error anchor, then greedy farthest-point selection in RMS distance."""
    selected = [min(range(len(errors)), key=lambda i: (errors[i], i))]
    distance = np.full(len(errors), np.inf)
    while len(selected) < min(count, len(errors)):
        distance = np.minimum(distance, np.mean((vectors - vectors[selected[-1]]) ** 2, axis=1))
        available = [i for i in range(len(errors)) if i not in selected]
        selected.append(min(available, key=lambda i: (-distance[i], errors[i], i)))
    return selected


def generate_ensemble(atoms, *, cutoff, targets, error_tolerance, count=8,
                      shells=2, sublattices=None, metric='max_error', method='auto',
                      max_configurations=10000, max_pool=256, max_seconds=30.0,
                      starts=8, steps=2000, seed=42, symprec=1e-5,
                      figures=None, species=None, selection='diversity', selection_seed=None):
    """Return (JSON-ready report, selected ASE Atoms), preserving compositions.

    The tolerance is absolute error in the selected pair-SRO metric. Small
    spaces use symmetry-aware enumeration; larger spaces use seeded Python SA
    starts, continuing past individual optimum witnesses. Selection is either
    greedy diversity (default) or uniform without replacement from that pool.
    Neither mode samples uniformly over all labeled occupations. selection_seed
    defaults to seed and never changes candidate generation.
    """
    start = time.monotonic()
    for name, value in dict(count=count, max_configurations=max_configurations, max_pool=max_pool,
                            starts=starts, steps=steps).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if max_pool < count:
        raise ValueError('max_pool must be at least count')
    if method not in ('auto', 'enumerate', 'sa'):
        raise ValueError('method must be auto, enumerate, or sa')
    if selection not in ('diversity', 'random'):
        raise ValueError('selection must be diversity or random')
    if selection_seed is None:
        selection_seed = seed
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int) or selection_seed < 0:
        raise ValueError('selection_seed must be a nonnegative integer')
    if metric not in METRICS:
        raise ValueError('Unknown metric')
    if not math.isfinite(error_tolerance) or error_tolerance < 0:
        raise ValueError('error_tolerance must be finite and nonnegative')
    if not math.isfinite(max_seconds) or max_seconds <= 0 or not math.isfinite(symprec) or symprec <= 0:
        raise ValueError('max_seconds and symprec must be positive and finite')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    config = deepcopy(sublattices) if sublattices else dict(
        site_labels=['all'] * len(atoms), allowed={'all': sorted(set(atoms.get_chemical_symbols()))})
    checker = SROBoundStop(atoms, cutoff=cutoff, targets=targets, shells=shells,
                           sublattices=config, metric=metric)
    bound = checker.bound
    total = configuration_count(bound['composition'])
    chosen_method = ('enumerate' if total <= max_configurations else 'sa') if method == 'auto' else method
    if chosen_method == 'enumerate' and total > max_configurations:
        raise ValueError('Occupation space exceeds max_configurations; use method=sa or raise the cap')
    for group, composition in bound['composition'].items():
        config.setdefault('counts', {})[group] = composition['counts']
    threshold = Fraction(str(error_tolerance)) ** (2 if metric == 'rms_error' else 1)
    screened = Fraction(bound['result']['lower_bound_exact']) > threshold
    labels = checker.labels
    permutations = np.arange(len(atoms))[None, :] if screened else site_permutations(
        atoms, labels, checker.prepared, symprec)
    analysis_args = dict(cutoff=cutoff, shells=shells, sublattices=config, sro_targets=targets)
    descriptor_args = dict(analysis_args)
    if figures is not None:
        model = figure_model(figures)
        if (model['sites'] != len(atoms) or not np.allclose(atoms.cell, model['cell'], atol=1e-8, rtol=0)
                or not np.allclose(atoms.get_scaled_positions(), model['positions'], atol=1e-8, rtol=0)):
            raise ValueError('Diversity FIGURES must match the input cell and site order')
        if species is None or len(species) != model['species'] or len(set(species)) != len(species):
            raise ValueError('Supply unique species in the FIGURES alphabet order')
        for group, alphabet in config['allowed'].items():
            indices = np.flatnonzero(labels == group)
            if any(not set(alphabet) <= {species[s] for s in model['allowed'][i]} for i in indices):
                raise ValueError('Sublattice species must be allowed by diversity FIGURES at every site')
            if model['groups'] and len({model['groups'][i] for i in indices}) != 1:
                raise ValueError('A search sublattice cannot span multiple diversity FIGURES groups')
        descriptor_args.update(figures=figures, species=list(species))
    initial_analysis = analyze_structure(atoms, **descriptor_args)
    if figures is not None:
        columns = [{k: f[k] for k in ('index', 'order', 'diameter')} for f in initial_analysis['figures']['features']]
    else:
        columns = [{k: row[k] for k in KEYS} for row in initial_analysis['sro'] if row['alpha'] is not None]
    pool, seen = [], set()
    evaluated, eligible, duplicates, covered, starts_run = 0, 0, 0, 0, 0
    reason = 'infeasible_by_bound' if screened else 'search_budget'
    deadline = start + max_seconds

    def collect(symbols, provenance):
        nonlocal evaluated, eligible, duplicates, reason
        if time.monotonic() >= deadline:
            reason = 'time_limit'
            return False
        evaluated += 1
        values = _exact_metrics(symbols, checker.prepared)
        if values[metric] > threshold:
            return True
        eligible += 1
        key = min(tuple(symbols[p]) for p in permutations)
        if key in seen:
            duplicates += 1
            return True
        seen.add(key)
        candidate = atoms.copy()
        candidate.set_chemical_symbols(key)
        # Evaluate the canonical decoration, so selection is independent of which
        # translated/rotated representative a solver happened to visit first.
        report = analyze_structure(candidate, **descriptor_args)
        vector = ([f['correlation'] for f in report['figures']['features']] if figures is not None else
                  [row['alpha'] for row in report['sro'] if row['alpha'] is not None])
        if not np.all(np.isfinite(vector)):
            raise ValueError('Diversity descriptors must be finite')
        pool.append(dict(symbols=key, metrics_exact={k: str(v) for k, v in values.items()},
            metrics={k: math.sqrt(v) if k == 'rms_error' else float(v) for k, v in values.items()},
            diversity_vector=vector, provenance=provenance,
            occupations_sha256=hashlib.sha256(','.join(key).encode()).hexdigest()))
        if len(pool) >= max_pool:
            reason = 'pool_limit'
            return False
        return True

    try:
        if not screened and collect(np.array(atoms.get_chemical_symbols()), dict(method='input', seed=None, step=0)):
            if chosen_method == 'enumerate':
                for symbols, multiplicity in symmetry_occupations(labels, bound['composition'], permutations):
                    if not collect(np.array(symbols), dict(method='enumerate', seed=None, representative=covered)):
                        break
                    covered += multiplicity
                else:
                    reason = 'exhausted'
            else:
                for run in range(starts):
                    if time.monotonic() >= deadline:
                        reason = 'time_limit'
                        break
                    run_seed = seed + run
                    initial = atoms.copy()
                    symbols = np.array(initial.get_chemical_symbols())
                    rng = np.random.default_rng(run_seed)
                    for group in sorted(config['allowed']):
                        positions = np.flatnonzero(labels == group)
                        symbols[positions] = rng.permutation(symbols[positions])
                    initial.set_chemical_symbols(symbols)
                    starts_run += 1
                    result, _ = _anneal(initial, config, targets, cutoff, shells, metric, steps,
                        run_seed, deadline, stop_at_bound=False,
                        visit=lambda s, step: collect(s, dict(method='python_pair_sro_sa', seed=run_seed, step=step)))
                    if result['stop_reason'] == 'visitor_stop':
                        break
                    if result['stop_reason'] == 'time_limit':
                        reason = 'time_limit'
                        break
                else:
                    reason = 'starts_exhausted'
    except KeyboardInterrupt:
        reason = 'interrupted'
    selected = []
    if pool:
        vectors = np.array([p['diversity_vector'] for p in pool])
        indices = (np.random.default_rng(selection_seed).choice(
            len(pool), size=min(count, len(pool)), replace=False).tolist()
            if selection == 'random' else
            _select(vectors, [Fraction(p['metrics_exact'][metric]) for p in pool], count))
        selected = [dict(pool[i], id=f'structure-{j:03d}', pool_index=i) for j, i in enumerate(indices)]
    witnesses = []
    for entry in selected:
        witness = atoms.copy()
        witness.set_chemical_symbols(entry['symbols'])
        witnesses.append(witness)
    spread = None
    if selected:
        vectors = np.array([p['diversity_vector'] for p in selected])
        distances = np.sqrt(np.mean((vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=2))
        spread = dict(min=vectors.min(axis=0).tolist(), max=vectors.max(axis=0).tolist(),
            mean=vectors.mean(axis=0).tolist(), population_std=vectors.std(axis=0).tolist(),
            pairwise_rms_distances=distances.tolist())
    selection_description = (
        'Uniform without replacement over collected symmetry-distinct candidates; no error or energy ranking.'
        if selection == 'random' else
        'Lowest-error anchor, then greedy farthest-point RMS distance; equal component weights; no variance rescaling.')
    report = dict(schema_version=2, objective='directed_pair_sro', metric=metric,
        exact_metric_conventions=dict(max_error='error', mean_absolute_error='error', rms_error='mean_squared_error'),
        status='requested_count_reached' if len(selected) == count else 'insufficient_candidates',
        stop_reason=reason, selected_count=len(selected), requested_count=count,
        error_tolerance=error_tolerance, selected=selected, bound=bound,
        sublattices=config, targets=deepcopy(targets),
        selection=dict(method=selection, seed=selection_seed, description=selection_description,
            population='collected_symmetry_orbits',
            pool_occupations_sha256=[p['occupations_sha256'] for p in pool]),
        search=dict(method=chosen_method, evaluated=evaluated, eligible=eligible, symmetry_duplicates=duplicates,
            pool_size=len(pool), site_symmetries=len(permutations), total_configurations=total,
            covered_configurations=covered, complete=reason == 'exhausted', starts_run=starts_run),
        settings=dict(cutoff=cutoff, shells=shells, seed=seed, starts=starts, steps=steps,
            max_configurations=max_configurations, max_pool=max_pool, max_seconds=max_seconds, symprec=symprec,
            selection=selection, selection_seed=selection_seed),
        diversity=dict(kind='decorated_figure_correlations' if figures is not None else 'pair_sro',
            columns=columns, selection=selection_description,
            figures_sha256=initial_analysis['figures']['sha256'] if figures is not None else None,
            species=list(species) if figures is not None else None, spread=spread),
        initial_structure_sha256=hashlib.sha256(json.dumps(dict(symbols=atoms.get_chemical_symbols(),
            positions=atoms.positions.tolist(), cell=atoms.cell.array.tolist()), sort_keys=True).encode()).hexdigest(),
        provenance=dict(source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('sro_ensemble.py', 'search_cells.py', 'sro_bounds.py', 'sro_joint_bounds.py',
                         'sro_stopping.py', 'enumeration_symmetry.py', 'structure_analysis.py', 'atat_reference.py')},
            dependencies={name: importlib.metadata.version(name) for name in ('numpy', 'ase', 'spglib')}),
        elapsed_seconds=time.monotonic() - start,
        limitations=['Fixed geometry and compositions; distinct under verified label- and target-preserving parent-cell symmetries. No species relabeling.',
            'A bounded candidate pool does not guarantee discovery of every eligible structure; greedy diversity is not globally optimal.',
            'Random selection is uniform only within the collected symmetry-orbit pool, without orbit-multiplicity weights; SA and truncated enumeration can bias that pool.',
            'Distinct structures can share the same descriptor vector; zero descriptor distance does not imply structural equivalence.',
            'No thermodynamic sampling distribution or statistical uncertainty estimate; spread describes only the selected configurations.',
            'Time limit is cooperative, includes preparation and collection, excludes final selection/export; library calls can overrun.'])
    if screened:
        report['status'] = 'infeasible_by_bound'
    return report, witnesses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('structure', type=Path)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--sublattices', type=Path)
    parser.add_argument('--cutoff', type=float, required=True)
    parser.add_argument('--shells', type=int, default=2)
    parser.add_argument('--error-tolerance', type=float, required=True)
    parser.add_argument('--metric', choices=METRICS, default='max_error')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--method', choices=['auto', 'enumerate', 'sa'], default='auto')
    parser.add_argument('--max-configurations', type=int, default=10000)
    parser.add_argument('--max-pool', type=int, default=256)
    parser.add_argument('--max-seconds', type=float, default=30)
    parser.add_argument('--starts', type=int, default=8)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--selection', choices=['diversity', 'random'], default='diversity',
                        help='Select diverse descriptors or sample uniformly from collected symmetry orbits')
    parser.add_argument('--selection-seed', type=int,
                        help='Random subset seed (defaults to --seed); does not alter candidate generation')
    parser.add_argument('--symprec', type=float, default=1e-5)
    parser.add_argument('--figures', type=Path, help='Optional diversity FIGURES, including pairs/triplets; same cell/site order')
    parser.add_argument('--species', help='Comma-separated FIGURES alphabet')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('Choose a new output directory')
        kwargs = vars(args).copy()
        for key in ('structure', 'output'):
            kwargs.pop(key)
        kwargs['targets'] = json.loads(args.targets.read_text())
        kwargs['sublattices'] = json.loads(args.sublattices.read_text()) if args.sublattices else None
        kwargs['species'] = args.species.split(',') if args.species else None
        report, witnesses = generate_ensemble(read(args.structure), **kwargs)
        report['inputs'] = {key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for key in ('structure', 'targets', 'sublattices', 'figures') if (path := getattr(args, key)) is not None}
        args.output.mkdir(parents=True)
        for entry, witness in zip(report['selected'], witnesses):
            directory = args.output / entry['id']
            directory.mkdir()
            write(directory / 'structure.extxyz', witness)
            write(directory / 'structure.cif', witness)
            write(directory / 'POSCAR', witness, format='vasp', direct=True, sort=False)
            entry['files'] = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(directory.iterdir())}
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f"Selected {len(witnesses)}/{args.count} structures; {report['stop_reason']}")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
