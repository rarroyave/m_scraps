"""Structure evaluation without an optimizer, ATAT, or icet installation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list

from atat_reference import correlations, figure_model, metrics


def map_sites(atoms, reference, tolerance=0.2):
    """Return input indices in reference order; reject ambiguous or missing sites.

    Cells must match within 1e-4 angstrom (legacy native output rounding).
    No automatic strain, origin, rotation, or vacancy fitting.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Mapping tolerance must be positive and finite')
    if len(atoms) != len(reference) or not np.allclose(atoms.cell, reference.cell, atol=1e-4, rtol=0):
        raise ValueError('Input and reference must have equal site counts and matching cells')
    delta = atoms.positions[None, :, :] - reference.positions[:, None, :]
    _, distance = find_mic(delta.reshape(-1, 3), reference.cell, pbc=True)
    distance = distance.reshape(len(reference), len(atoms))
    candidates = distance <= tolerance
    if not np.all(candidates.sum(axis=1) == 1) or not np.all(candidates.sum(axis=0) == 1):
        raise ValueError('Site mapping is missing or ambiguous within the requested tolerance')
    order = candidates.argmax(axis=1)
    return order, distance[np.arange(len(order)), order]


def analyze_structure(atoms, *, cutoff, shells=2, reference=None, sublattices=None,
                      figures=None, species=None, tolerance=0.2, shell_tolerance=1e-5,
                      wd=0.0, wn=1.0, sro_targets=None):
    """Evaluate directed, sublattice-resolved SRO and optional decorated figures.

    ``sublattices`` contains ``site_labels`` in reference order and ``allowed``
    mapping each label to chemical symbols; optional ``counts`` fixes composition.
    Shells use reference geometry if provided, otherwise the supplied geometry.
    Periodic images are counted explicitly, including in small supercells.
    """
    if not np.all(atoms.pbc) or abs(np.linalg.det(atoms.cell)) < 1e-10:
        raise ValueError('A nonsingular, fully periodic structure is required')
    if shells < 1 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('Positive cutoff and shell count are required')
    if not np.isfinite(shell_tolerance) or shell_tolerance <= 0:
        raise ValueError('Shell tolerance must be positive and finite')
    model = figure_model(figures) if figures else None
    if model is not None:
        figure_reference = Atoms('H' * model['sites'], cell=model['cell'],
                                 scaled_positions=model['positions'], pbc=True)
        if reference is not None and (len(reference) != len(figure_reference) or
                not np.allclose(reference.cell, figure_reference.cell, atol=1e-4, rtol=0) or
                not np.allclose(reference.get_scaled_positions(), figure_reference.get_scaled_positions(), atol=1e-8, rtol=0)):
            raise ValueError('Explicit reference must match figure site order and cell')
        reference = figure_reference
    geometry = reference if reference is not None else atoms
    order, displacements = map_sites(atoms, geometry, tolerance)
    symbols = np.array(atoms.get_chemical_symbols())[order]
    if sublattices is None:
        if model is not None and model['groups']:
            if species is None or len(species) != model['species']:
                raise ValueError('Supply species in the figure alphabet order')
            labels = [str(g) if g >= 0 else 'fixed_' + '_'.join(map(str, a))
                      for g, a in zip(model['groups'], model['allowed'])]
            allowed = {}
            for label, alphabet in zip(labels, model['allowed']):
                value = [species[i] for i in alphabet]
                if label in allowed and allowed[label] != value:
                    raise ValueError('Inconsistent site alphabets within figure group')
                allowed[label] = value
            sublattices = {'site_labels': labels, 'allowed': allowed}
        else:
            sublattices = {'site_labels': ['all'] * len(atoms),
                          'allowed': {'all': sorted(set(symbols))}}
    labels = np.array(sublattices['site_labels'], dtype=str)
    allowed = sublattices['allowed']
    if labels.shape != (len(atoms),) or set(labels) != set(allowed):
        raise ValueError('site_labels and allowed must define every reference site')
    composition = {}
    for label in sorted(allowed):
        alphabet = allowed[label]
        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError('Each sublattice needs a unique, nonempty species alphabet')
        actual = Counter(symbols[labels == label])
        if set(actual) - set(alphabet):
            raise ValueError(f'Forbidden occupation on sublattice {label}: {dict(actual)}')
        counts = {s: actual[s] for s in alphabet}
        expected = sublattices.get('counts', {}).get(label)
        if expected is not None and counts != expected:
            raise ValueError(f'Composition mismatch on sublattice {label}')
        composition[label] = {'counts': counts, 'sites': int(sum(actual.values())),
                              'fixed': len(alphabet) == 1}
    centers, neighbors, distances = neighbor_list('ijd', geometry, cutoff)
    rows = []
    coverage = []
    for source in sorted(allowed):
        for destination in sorted(allowed):
            selected = np.flatnonzero((labels[centers] == source) & (labels[neighbors] == destination))
            selected = selected[np.argsort(distances[selected])]
            groups = []
            for index in selected:
                if not groups or distances[index] - distances[groups[-1][0]] > shell_tolerance:
                    groups.append([])
                groups[-1].append(index)
            coverage.append(dict(source=source, destination=destination,
                                 shells_within_cutoff=len(groups), reported_shells=min(shells, len(groups))))
            for shell, indices in enumerate(groups[:shells], 1):
                ii, jj = centers[indices], neighbors[indices]
                for a in allowed[source]:
                    center_mask = symbols[ii] == a
                    bonds = int(center_mask.sum())
                    n_a = composition[source]['counts'][a]
                    for b in allowed[destination]:
                        count = int(np.sum(center_mask & (symbols[jj] == b)))
                        concentration = composition[destination]['counts'][b] / composition[destination]['sites']
                        alpha = 1 - count / bonds / concentration if bonds and concentration else None
                        rows.append(dict(source=source, destination=destination, shell=shell,
                                         center_species=a, neighbor_species=b, pair_count=count,
                                         center_bonds=bonds, coordination=bonds / n_a if n_a else None,
                                         neighbor_concentration=concentration, alpha=alpha,
                                         distance_min=float(distances[indices].min()),
                                         distance_max=float(distances[indices].max()),
                                         fixed_pair=composition[source]['fixed'] or composition[destination]['fixed']))
    result = dict(schema_version=1, sites=len(atoms), composition=composition,
                  mapping=dict(input_indices=order.tolist(), max_displacement=float(max(displacements))),
                  convention=dict(sro='1 - P(neighbor species | center species, shell, sublattice pair) / neighbor-sublattice concentration',
                                  pair_counts='directed periodic bonds; conditional probability normalized by observed center-species bonds',
                                  shell_geometry='reference' if reference is not None else 'input',
                                  shell_tolerance_angstrom=shell_tolerance, cutoff_angstrom=cutoff,
                                  requested_shells_per_sublattice_pair=shells), shell_coverage=coverage, sro=rows)
    if sro_targets is not None:
        keys = ('source', 'destination', 'shell', 'center_species', 'neighbor_species')
        lookup = {tuple(row[k] for k in keys): row for row in rows}
        comparisons, seen = [], set()
        for target in sro_targets:
            key = tuple(target[k] for k in keys)
            if key in seen:
                raise ValueError('Duplicate SRO target')
            seen.add(key)
            row = lookup.get(key)
            if row is None or row['alpha'] is None:
                raise ValueError(f'SRO target has no evaluable pair: {key}')
            value = float(target['target'])
            if not np.isfinite(value):
                raise ValueError('SRO targets must be finite')
            comparisons.append(dict(**{k: target[k] for k in keys}, target=value,
                                    alpha=row['alpha'], residual=row['alpha'] - value))
        if not comparisons:
            raise ValueError('SRO target list must not be empty')
        errors = np.array([row['residual'] for row in comparisons])
        result['sro_comparison'] = dict(entries=comparisons, metrics=dict(
            max_error=float(max(abs(errors))), rms_error=float(np.sqrt(np.mean(errors**2))),
            mean_absolute_error=float(np.mean(abs(errors)))))
    if model is not None:
        if species is None or len(species) != model['species'] or len(set(species)) != len(species):
            raise ValueError('Supply unique species in the figure alphabet order')
        if not np.isfinite(wd) or not np.isfinite(wn) or wd < 0 or wn <= 0:
            raise ValueError('Figure weights require finite wd >= 0 and wn > 0')
        try:
            occupations = [species.index(s) for s in symbols]
        except ValueError as exc:
            raise ValueError('Structure species is absent from figure alphabet') from exc
        vector = correlations(model, occupations)
        if model['groups']:
            for group in set(model['groups']) - {-1}:
                indices = [i for i, g in enumerate(model['groups']) if g == group]
                expected = model['local_counts'][indices[0]]
                if np.bincount(np.array(occupations)[indices], minlength=model['species']).tolist() != expected:
                    raise ValueError(f'Figure composition mismatch in group {group}')
        residual = vector - [f['target'] for f in model['features']]
        scores = metrics(model, vector, wd=wd, wn=wn)
        scores.update(rms_error=float(np.sqrt(np.mean(residual**2))),
                      mean_absolute_error=float(np.mean(abs(residual))))
        result['figures'] = dict(sha256=hashlib.sha256(Path(figures).read_bytes()).hexdigest(),
                                species=species, basis=model['basis'].tolist(), wd=wd, wn=wn,
                                convention='SCRAPS_FIGURES explicit decorated sums; feature order preserved',
                                metrics=scores, features=[dict(index=i, order=f['order'], diameter=f['diameter'],
                                    target=f['target'], correlation=float(vector[i]), residual=float(residual[i]))
                                    for i, f in enumerate(model['features'])])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('structure', type=Path)
    parser.add_argument('--reference', type=Path, help='Full supercell with reference site order')
    parser.add_argument('--sublattices', type=Path, help='JSON site_labels, allowed, optional counts')
    parser.add_argument('--targets', type=Path, help='JSON list of sublattice-pair SRO targets')
    parser.add_argument('--figures', type=Path, help='Native FIGURES file, optionally gzip compressed')
    parser.add_argument('--species', help='Comma-separated figure alphabet, in index order')
    parser.add_argument('--cutoff', type=float, required=True, help='Neighbor cutoff in angstrom')
    parser.add_argument('--shells', type=int, default=2)
    parser.add_argument('--tolerance', type=float, default=0.2, help='Site mapping tolerance in angstrom')
    parser.add_argument('--shell-tolerance', type=float, default=1e-5, help='Shell distance width in angstrom')
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--wn', type=float, default=1)
    parser.add_argument('--output', type=Path, help='JSON output; default stdout')
    parser.add_argument('--csv', type=Path, help='Optional SRO table')
    args = parser.parse_args(argv)
    try:
        result = analyze_structure(read(args.structure), cutoff=args.cutoff, shells=args.shells,
            reference=read(args.reference) if args.reference else None,
            sublattices=json.loads(args.sublattices.read_text()) if args.sublattices else None,
            figures=args.figures, species=args.species.split(',') if args.species else None,
            tolerance=args.tolerance, shell_tolerance=args.shell_tolerance, wd=args.wd, wn=args.wn,
            sro_targets=json.loads(args.targets.read_text()) if args.targets else None)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    result['input'] = dict(path=str(args.structure), sha256=hashlib.sha256(args.structure.read_bytes()).hexdigest())
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')
    if args.csv:
        with args.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['sro'][0]) if result['sro'] else ['alpha'])
            writer.writeheader()
            writer.writerows(result['sro'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
