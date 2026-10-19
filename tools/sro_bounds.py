"""Rigorous necessary pair-count bounds for fixed-composition SRO targets.

This independent-component relaxation is not a lattice realizability solver.
It requires uniform coordination within each requested sublattice-pair shell.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

from structure_analysis import analyze_structure

KEYS = ('source', 'destination', 'shell', 'center_species', 'neighbor_species')
METRICS = ('max_error', 'rms_error', 'mean_absolute_error')


def pair_count_bound(center_bonds, neighbor_bonds, total_bonds, denominator,
                     target, *, same_species=False):
    """Minimize one residual over integer counts with capacity and parity bounds.

    Inclusion/exclusion gives max(0, A+B-E) <= n <= min(A,B), where
    A and B count source- and destination-species stubs among E directed edges.
    Same-species, same-sublattice counts are even on an undirected periodic
    graph (both orientations, including self-image edges, are retained).
    """
    low = max(0, center_bonds + neighbor_bonds - total_bonds)
    high = min(center_bonds, neighbor_bonds)
    step = 2 if same_species else 1
    low = ((low + step - 1) // step) * step
    high = (high // step) * step
    if low > high or denominator <= 0:
        raise ValueError('No admissible pair count or undefined SRO denominator')
    desired = denominator * (1 - target)
    floor = (desired // step) * step
    choices = {max(low, min(high, int(n))) for n in (floor, floor + step)}
    nearest = min(choices, key=lambda n: (abs(Fraction(n) - desired), n))
    error = abs(1 - Fraction(nearest) / denominator - target)
    return low, high, step, desired, nearest, error


def sro_bounds(atoms, *, cutoff, targets, shells=2, sublattices=None,
               reference=None, tolerance=0.2, shell_tolerance=1e-5,
               metric='max_error'):
    """Bound errors on only the explicitly supplied directed SRO components.

    Composition comes from the input occupations; optional sublattice counts
    must agree. Geometry and shell conventions match ``analyze_structure``.
    No ATAT, icet, native optimizer, or search is needed. Fractions preserve
    exact arithmetic for bounds/certificates; JSON numerical fields are rounded
    floating-point presentations of those values.
    """
    if metric not in METRICS:
        raise ValueError(f'Unknown metric: {metric}')
    if not isinstance(targets, list) or not targets:
        raise ValueError('Supply a nonempty list of SRO targets')
    report = analyze_structure(atoms, cutoff=cutoff, shells=shells,
        sublattices=sublattices, reference=reference, tolerance=tolerance,
        shell_tolerance=shell_tolerance, sro_targets=targets)
    geometry = reference if reference is not None else atoms
    labels = np.array(sublattices['site_labels'] if sublattices else ['all'] * len(atoms), dtype=str)
    centers, neighbors, distances = neighbor_list('ijd', geometry, cutoff)
    lookup = {tuple(row[k] for k in KEYS): row for row in report['sro']}
    entries, lower, achieved = [], [], []
    for target in targets:
        key = tuple(target[k] for k in KEYS)
        row = lookup[key]
        source, destination, shell, a, b = key
        # Reconstruct this exact analyzer shell, retaining all periodic images.
        selected = ((labels[centers] == source) & (labels[neighbors] == destination)
                    & (distances >= row['distance_min']) & (distances <= row['distance_max']))
        ii, jj = centers[selected], neighbors[selected]
        src_degrees = np.bincount(ii, minlength=len(atoms))[labels == source]
        dst_degrees = np.bincount(jj, minlength=len(atoms))[labels == destination]
        if not np.all(src_degrees == src_degrees[0]) or not np.all(dst_degrees == dst_degrees[0]):
            raise ValueError(f'Nonuniform coordination for {source}->{destination}, shell {shell}; '
                             'split the sites into uniform sublattices or use an ideal reference')
        z_src, z_dst = int(src_degrees[0]), int(dst_degrees[0])
        src, dst = report['composition'][source], report['composition'][destination]
        n_a, n_b = src['counts'][a], dst['counts'][b]
        center_bonds, neighbor_bonds = n_a * z_src, n_b * z_dst
        denominator = Fraction(center_bonds * n_b, dst['sites'])
        value = Fraction(str(target['target']))
        low, high, step, desired, nearest, error = pair_count_bound(
            center_bonds, neighbor_bonds, len(ii), denominator, value,
            same_species=source == destination and a == b)
        actual = abs(1 - Fraction(row['pair_count']) / denominator - value)
        lower.append(error)
        achieved.append(actual)
        entries.append(dict(**{k: target[k] for k in KEYS}, target=float(value),
            coordination=z_src, reverse_coordination=z_dst,
            denominator_exact=str(denominator), target_pair_count_exact=str(desired),
            admissible_count_min=low, admissible_count_max=high, count_step=step,
            nearest_relaxed_count=nearest, component_lower_bound=float(error),
            component_lower_bound_exact=str(error), achieved_pair_count=row['pair_count'],
            achieved_error=float(actual)))
    count = len(entries)
    exact_lower = {'max_error': max(lower), 'mean_absolute_error': sum(lower) / count,
                   'rms_error': sum(e * e for e in lower) / count}
    exact_upper = {'max_error': max(achieved), 'mean_absolute_error': sum(achieved) / count,
                   'rms_error': sum(e * e for e in achieved) / count}
    metrics = {}
    for name in METRICS:
        lo, hi = exact_lower[name], exact_upper[name]
        if lo > hi:
            raise RuntimeError('Pair-count bound exceeds the supplied valid witness')
        convert = math.sqrt if name == 'rms_error' else float
        low, high = convert(lo), convert(hi)
        metrics[name] = dict(lower_bound=low, achieved_upper_bound=high,
            optimality_gap=high - low, status='certified_minimum' if lo == hi else 'lower_bound',
            exact_value_convention='squared error' if name == 'rms_error' else 'error',
            lower_bound_exact=str(lo), achieved_upper_bound_exact=str(hi))
    return dict(schema_version=1, objective='directed_pair_sro', method='independent_integer_pair_counts',
        selected_metric=metric, result=metrics[metric], metrics=metrics,
        sites=len(atoms), composition=report['composition'], entries=entries,
        convention=report['convention'], shell_coverage=report['shell_coverage'],
        assumptions=[
            'Fixed cell, geometry, sublattice membership, and input composition.',
            'Uniform coordination within each requested sublattice-pair shell.',
            'Only explicit targets contribute; equal weights; shells are one-based per sublattice pair.',
            'Integer counts, source/destination capacities, and same-species parity are enforced.',
            'Components are relaxed independently; joint neighbor totals, cross-component reciprocity, '
            'SRO sum rules, and full lattice realizability are not jointly optimized.',
            'A zero lower bound alone does not establish feasibility. Certification requires the supplied witness to attain it.',
            'These are SRO-error bounds, not bounds on decorated figure-correlation or native weighted bond-count objectives.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('structure', type=Path, help='Fully occupied periodic cell; fixes composition')
    parser.add_argument('--targets', type=Path, required=True, help='Same JSON target list as analyze --targets')
    parser.add_argument('--cutoff', type=float, required=True, help='Neighbor cutoff in angstrom')
    parser.add_argument('--shells', type=int, default=2, help='Shells per sublattice pair (default 2)')
    parser.add_argument('--sublattices', type=Path, help='JSON site_labels, allowed, optional counts')
    parser.add_argument('--reference', type=Path, help='Ideal full-cell geometry and site order')
    parser.add_argument('--tolerance', type=float, default=0.2)
    parser.add_argument('--shell-tolerance', type=float, default=1e-5)
    parser.add_argument('--metric', choices=METRICS, default='max_error')
    parser.add_argument('--output', type=Path, help='JSON output; default stdout')
    args = parser.parse_args(argv)
    try:
        result = sro_bounds(read(args.structure), cutoff=args.cutoff,
            targets=json.loads(args.targets.read_text()), shells=args.shells,
            sublattices=json.loads(args.sublattices.read_text()) if args.sublattices else None,
            reference=read(args.reference) if args.reference else None,
            tolerance=args.tolerance, shell_tolerance=args.shell_tolerance, metric=args.metric)
        inputs = {'structure': args.structure, 'targets': args.targets,
                  'sublattices': args.sublattices, 'reference': args.reference}
        result['inputs'] = {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                            for name, path in inputs.items() if path is not None}
        payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(payload)
        else:
            print(payload, end='')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
