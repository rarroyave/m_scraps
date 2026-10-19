"""Budgeted supercell-shape and occupation search for explicit pair-SRO targets."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import spglib
from ase.build import make_supercell
from ase.geometry import find_mic, minkowski_reduce
from ase.io import read, write
from ase.neighborlist import neighbor_list

from enumerate_sro import configuration_count, enumerate_sro
from sro_bounds import KEYS, METRICS, sro_bounds
from structure_analysis import analyze_structure


def hnf_matrices(index):
    """Unique row-lattice Hermite normal forms of positive determinant index."""
    for a in range(1, index + 1):
        if index % a:
            continue
        for c in range(1, index // a + 1):
            if index % (a * c):
                continue
            f = index // (a * c)
            for b in range(a):
                for d in range(a):
                    for e in range(c):
                        yield np.array([[a, 0, 0], [b, c, 0], [d, e, f]], dtype=int)


def row_hnf(matrix):
    """Canonical lower triangular row HNF using exact integer row operations."""
    h = np.array(matrix, dtype=object).copy()
    for j in (2, 1, 0):
        while any(h[i, j] for i in range(j)):
            i = next(i for i in range(j) if h[i, j])
            if h[j, j] == 0:
                h[[i, j]] = h[[j, i]]
            q = h[i, j] // h[j, j]
            h[i] -= q * h[j]
            if h[i, j]:
                h[[i, j]] = h[[j, i]]
        if h[j, j] == 0:
            raise ValueError('Singular integer matrix')
        if h[j, j] < 0:
            h[j] *= -1
        for i in range(j + 1, 3):
            h[i] -= (h[i, j] // h[j, j]) * h[j]
    return tuple(int(x) for x in h.flat)


def parent_rotations(atoms, labels):
    """Conservative label-preserving point group, independently checked."""
    types = [sorted(set(labels)).index(label) + 1 for label in labels]
    positions = atoms.get_scaled_positions()
    data = spglib.get_symmetry((atoms.cell, positions, types), symprec=1e-7)
    rotations = {tuple(np.eye(3, dtype=int).flat): np.eye(3, dtype=int)}
    if data is None:
        return list(rotations.values())
    gram = np.asarray(atoms.cell) @ np.asarray(atoms.cell).T
    for rotation, translation in zip(data['rotations'], data['translations']):
        if not np.allclose(rotation.T @ gram @ rotation, gram, rtol=0, atol=1e-9):
            continue
        transformed = positions @ rotation.T + translation
        delta = (transformed[:, None] - positions[None, :]) @ atoms.cell
        _, distances = find_mic(delta.reshape(-1, 3), atoms.cell, pbc=True)
        matches = (distances.reshape(len(atoms), -1) < 1e-8) & (labels[:, None] == labels[None, :])
        if np.all(matches.sum(axis=0) == 1) and np.all(matches.sum(axis=1) == 1):
            rotations[tuple(rotation.flat)] = rotation
    return list(rotations.values())


def cell_geometry(cell):
    """Measure the periodic lattice after removing avoidable basis skew."""
    reduced, operation = minkowski_reduce(cell)
    reduced = np.asarray(reduced)
    lengths = np.linalg.norm(reduced, axis=1)
    shortest = float(min(lengths))
    return dict(shortest_repeat=shortest, aspect_ratio=float(max(lengths) / shortest),
                reduced_lengths=lengths.tolist(), volume=float(abs(np.linalg.det(reduced))),
                reduced_cell=reduced.tolist(), basis_operation=np.asarray(operation).tolist())


def _composition(atoms, sublattices, composition):
    config = sublattices or {'site_labels': ['all'] * len(atoms),
                            'allowed': {'all': sorted(composition['all']) if composition is not None and 'all' in composition
                                               else sorted(set(atoms.get_chemical_symbols()))}}
    labels = np.asarray(config['site_labels'], dtype=str)
    if labels.shape != (len(atoms),) or set(labels) != set(config['allowed']):
        raise ValueError('site_labels and allowed must define every parent site')
    if composition is not None and 'counts' in config:
        raise ValueError('Use composition fractions or sublattice counts, not both')
    if composition is not None and set(composition) != set(labels):
        raise ValueError('composition must specify every sublattice label')
    fractions = {}
    for label in sorted(set(labels)):
        allowed = config['allowed'][label]
        if not allowed or len(set(allowed)) != len(allowed):
            raise ValueError('Each sublattice needs a unique nonempty species alphabet')
        actual = Counter(np.asarray(atoms.get_chemical_symbols())[labels == label])
        if composition is None:
            if set(actual) - set(allowed):
                raise ValueError('Forbidden parent occupation')
            if label in config.get('counts', {}) and config['counts'][label] != {s: actual[s] for s in allowed}:
                raise ValueError('Parent composition differs from sublattice counts')
            values = {s: Fraction(actual[s], sum(actual.values())) for s in allowed}
        else:
            if set(composition[label]) - set(allowed):
                raise ValueError('Composition contains a forbidden species')
            try:
                values = {s: Fraction(str(composition[label].get(s, 0))) for s in allowed}
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError('Composition values must be finite fractions') from exc
        if any(v < 0 for v in values.values()) or sum(values.values()) != 1:
            raise ValueError('Composition fractions must be nonnegative and sum exactly to one; use strings such as 1/3')
        fractions[label] = values
    return labels, config['allowed'], fractions


def _anneal(atoms, config, targets, cutoff, shells, metric, steps, seed, deadline):
    """Small Python SA using the explicit pair-SRO metric, not native fitness."""
    analysis = analyze_structure(atoms, cutoff=cutoff, shells=shells, sublattices=config, sro_targets=targets)
    bound = sro_bounds(atoms, cutoff=cutoff, shells=shells, sublattices=config, targets=targets, metric=metric)
    labels = np.asarray(config['site_labels'])
    ii, jj, distances = neighbor_list('ijd', atoms, cutoff)
    rows = {tuple(row[k] for k in KEYS): row for row in analysis['sro']}
    prepared = []
    for entry, target in zip(bound['entries'], targets):
        key = tuple(target[k] for k in KEYS)
        row = rows[key]
        mask = ((labels[ii] == key[0]) & (labels[jj] == key[1]) &
                (distances >= row['distance_min']) & (distances <= row['distance_max']))
        prepared.append((ii[mask], jj[mask], key[3], key[4], float(Fraction(entry['denominator_exact'])), float(target['target'])))

    def score(symbols):
        residuals = np.array([1 - np.count_nonzero((symbols[i] == a) & (symbols[j] == b)) / d - t
                             for i, j, a, b, d, t in prepared])
        if metric == 'max_error':
            return float(np.max(np.abs(residuals)))
        if metric == 'rms_error':
            return float(np.sqrt(np.mean(residuals ** 2)))
        return float(np.mean(np.abs(residuals)))

    rng = np.random.default_rng(seed)
    current = np.asarray(atoms.get_chemical_symbols())
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))
              if len(set(current[labels == label])) > 1]
    value = score(current)
    best, best_value = current.copy(), value
    completed = 0
    for step in range(steps if groups else 0):
        if time.monotonic() >= deadline:
            break
        group = groups[int(rng.integers(len(groups)))]
        i, j = rng.choice(group, 2, replace=False)
        current[i], current[j] = current[j], current[i]
        proposed = score(current)
        temperature = 0.1 * (0.001 ** (step / max(1, steps - 1)))
        if proposed <= value or rng.random() < math.exp(min(0, (value - proposed) / temperature)):
            value = proposed
            if value < best_value:
                best, best_value = current.copy(), value
        else:
            current[i], current[j] = current[j], current[i]
        completed += 1
    witness = atoms.copy()
    witness.set_chemical_symbols(best.tolist())
    final = sro_bounds(witness, cutoff=cutoff, shells=shells, sublattices=config, targets=targets, metric=metric)
    return dict(method='python_pair_sro_sa', status='certified_minimum' if final['result']['status'] == 'certified_minimum' else 'best_found',
                best_error=final['result']['achieved_upper_bound'], lower_bound=final['result']['lower_bound'],
                steps=completed, seed=seed, temperature_start=0.1, temperature_end=0.0001,
                stop_reason='steps' if completed == steps or not groups else 'time_limit'), witness


def search_cells(atoms, *, cutoff, targets, min_atoms, max_atoms, sublattices=None,
                 composition=None, shells=2, metric='max_error', min_repeat=0.0,
                 max_aspect=2.0, max_hnfs=2000, max_candidates=12,
                 max_configurations=10000, sa_steps=2000, seed=42,
                 max_seconds=60.0, seconds_per_cell=3.0, error_tolerance=None):
    """Return (report, witnesses by candidate id) for a bounded shape search.

    Sizes are multiples of the supplied parent cell. Composition defaults to its
    occupied fractions, or explicit per-label fractions. No rounded compositions.
    Candidate caps and geometry filters preclude a global optimality claim.
    Time limits are cooperative; individual geometry/analysis calls may overrun.
    """
    start = time.monotonic()
    for name, value in dict(min_atoms=min_atoms, max_atoms=max_atoms, max_hnfs=max_hnfs,
                            max_candidates=max_candidates, max_configurations=max_configurations, sa_steps=sa_steps).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if min_atoms > max_atoms or not len(atoms) or not np.all(atoms.pbc) or abs(np.linalg.det(atoms.cell)) < 1e-10:
        raise ValueError('Require a nonempty fully periodic parent and ordered atom-count range')
    if metric not in METRICS:
        raise ValueError('Unknown metric')
    for name, value, minimum in [('min_repeat', min_repeat, 0), ('max_aspect', max_aspect, 1),
                                  ('max_seconds', max_seconds, 0), ('seconds_per_cell', seconds_per_cell, 0)]:
        if not math.isfinite(value) or value < minimum or (name in ('max_seconds', 'seconds_per_cell') and value == 0):
            raise ValueError(f'Invalid {name}')
    if error_tolerance is not None and (not math.isfinite(error_tolerance) or error_tolerance < 0):
        raise ValueError('error_tolerance must be nonnegative and finite')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if shells < 1 or cutoff <= 0 or not math.isfinite(cutoff):
        raise ValueError('Positive cutoff and shell count required')
    labels, allowed, fractions = _composition(atoms, sublattices, composition)
    rotations = parent_rotations(atoms, labels)
    deadline = start + max_seconds
    pool, seen, examined, incompatible = [], set(), 0, []
    stop = 'exhausted'
    for index in range(max(1, math.ceil(min_atoms / len(atoms))), max_atoms // len(atoms) + 1):
        counts = {label: {s: v * int(sum(labels == label)) * index for s, v in values.items()}
                  for label, values in fractions.items()}
        if any(v.denominator != 1 for group in counts.values() for v in group.values()):
            incompatible.append(len(atoms) * index)
            continue
        for matrix in hnf_matrices(index):
            if examined >= max_hnfs or time.monotonic() >= deadline:
                stop = 'hnf_cap' if examined >= max_hnfs else 'time_limit'
                break
            examined += 1
            key = min(row_hnf(matrix @ rotation.T) for rotation in rotations)
            if key in seen:
                continue
            seen.add(key)
            geometry = cell_geometry(matrix @ atoms.cell)
            if geometry['shortest_repeat'] + 1e-10 < min_repeat or geometry['aspect_ratio'] > max_aspect + 1e-10:
                continue
            pool.append(dict(matrix=matrix.tolist(), index=index, atoms=len(atoms) * index,
                             geometry=geometry, counts={g: {s: int(v) for s, v in c.items()} for g, c in counts.items()}))
        if stop != 'exhausted':
            break
    # Round-robin across sizes; compact shapes first within each size.
    grouped = {}
    for candidate in sorted(pool, key=lambda c: (c['geometry']['aspect_ratio'], -c['geometry']['shortest_repeat'], c['matrix'])):
        grouped.setdefault(candidate['atoms'], []).append(candidate)
    ordered = [grouped[n][rank] for rank in range(max(map(len, grouped.values()), default=0))
               for n in sorted(grouped) if rank < len(grouped[n])]
    candidates, witnesses = [], {}
    for candidate in ordered[:max_candidates]:
        if time.monotonic() >= deadline:
            break
        cid = f'cell-{len(candidates):03d}'
        candidate = dict(candidate, id=cid)
        cell = make_supercell(atoms, np.asarray(candidate['matrix']))
        cell.set_cell(candidate['geometry']['reduced_cell'], scale_atoms=False)
        cell.wrap()
        expanded = np.tile(labels, candidate['index'])
        config = dict(site_labels=expanded.tolist(), allowed=allowed, counts=candidate['counts'])
        symbols = np.asarray(cell.get_chemical_symbols(), dtype=object)
        for group, counts in candidate['counts'].items():
            symbols[expanded == group] = [s for s, count in sorted(counts.items()) for _ in range(count)]
        cell.set_chemical_symbols(symbols.tolist())
        bound = sro_bounds(cell, cutoff=cutoff, targets=targets, shells=shells, sublattices=config, metric=metric)
        candidate['sublattices'] = config
        candidate['lower_bound'] = bound['result']['lower_bound']
        candidate['total_configurations'] = configuration_count(bound['composition'])
        if error_tolerance is not None and candidate['lower_bound'] > error_tolerance:
            candidate.update(status='screened_by_bound', best_error=None)
        else:
            remaining = max(1e-9, min(seconds_per_cell, deadline - time.monotonic()))
            if candidate['total_configurations'] <= max_configurations:
                optimization, witness = enumerate_sro(cell, cutoff=cutoff, targets=targets, shells=shells,
                    sublattices=config, metric=metric, max_configurations=max_configurations, max_seconds=remaining)
            else:
                optimization, witness = _anneal(cell, config, targets, cutoff, shells, metric, sa_steps,
                                                seed + len(candidates), min(deadline, time.monotonic() + remaining))
            candidate.update(optimization=optimization, status=optimization['status'],
                             best_error=optimization['best_error'], lower_bound=optimization['lower_bound'])
            witnesses[cid] = witness
        candidates.append(candidate)
    evaluated = [c for c in candidates if c['best_error'] is not None]
    def objectives(c):
        return (c['atoms'], c['best_error'], c['geometry']['aspect_ratio'], -c['geometry']['shortest_repeat'])
    for c in evaluated:
        c['pareto'] = not any(all(a <= b for a, b in zip(objectives(d), objectives(c))) and
                             any(a < b for a, b in zip(objectives(d), objectives(c))) for d in evaluated)
    report = dict(schema_version=1, objective='directed_pair_sro', selected_metric=metric,
        status='bounded_search', global_optimum_certified=False, candidates=candidates,
        pareto_ids=[c['id'] for c in evaluated if c['pareto']],
        generation=dict(stop_reason=stop, hnfs_examined=examined, unique_lattices=len(seen),
                        geometry_accepted=len(pool), parent_rotations=len(rotations),
                        composition_incompatible_sizes=incompatible),
        candidate_limit_reached=len(ordered) > max_candidates,
        time_limit_reached=time.monotonic() >= deadline,
        elapsed_seconds=time.monotonic() - start,
        settings=dict(min_atoms=min_atoms, max_atoms=max_atoms, min_repeat=min_repeat, max_aspect=max_aspect,
                      max_hnfs=max_hnfs, max_candidates=max_candidates, max_configurations=max_configurations,
                      sa_steps=sa_steps, seed=seed, max_seconds=max_seconds, seconds_per_cell=seconds_per_cell,
                      cutoff=cutoff, shells=shells, error_tolerance=error_tolerance),
        fractions={g: {s: str(v) for s, v in values.items()} for g, values in fractions.items()}, targets=targets,
        limitations=['Only integer supercells of the supplied parent are considered; no smaller primitive-cell inference.',
                     'Geometry uses Minkowski-reduced basis lengths; shortest_repeat is a periodic lattice translation, not an interatomic distance.',
                     'Shape, candidate and generation caps restrict the search; no global minimum across shapes is certified.',
                     'Pareto flags compare achieved errors among evaluated candidates, not their unknown true minima.',
                     'Exact certificates apply to one cell only. Larger occupation spaces use a seeded Python pair-SRO SA, not the native SCRAPS solver.',
                     'Pair-SRO and uniform coordination only; no higher-order correlation objective.',
                     'Time budgets are cooperative and include preparation; individual library calls may overrun.'])
    return report, witnesses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('structure', type=Path, help='Parent lattice; occupied fractions are used unless --composition is supplied')
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--sublattices', type=Path)
    parser.add_argument('--composition', type=Path, help='JSON fractions by sublattice and species; rational strings accepted')
    parser.add_argument('--cutoff', type=float, required=True)
    parser.add_argument('--shells', type=int, default=2)
    parser.add_argument('--metric', choices=METRICS, default='max_error')
    parser.add_argument('--min-atoms', type=int, required=True)
    parser.add_argument('--max-atoms', type=int, required=True)
    parser.add_argument('--min-repeat', type=float, default=0)
    parser.add_argument('--max-aspect', type=float, default=2)
    parser.add_argument('--max-hnfs', type=int, default=2000)
    parser.add_argument('--max-candidates', type=int, default=12)
    parser.add_argument('--max-configurations', type=int, default=10000)
    parser.add_argument('--sa-steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--seconds-per-cell', type=float, default=3)
    parser.add_argument('--error-tolerance', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('Output directory already exists; choose a new path')
        kwargs = vars(args).copy()
        for key in ('structure', 'output'):
            kwargs.pop(key)
        for key in ('targets', 'sublattices', 'composition'):
            kwargs[key] = json.loads(kwargs[key].read_text()) if kwargs[key] else None
        report, witnesses = search_cells(read(args.structure), **kwargs)
        report['inputs'] = {key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                            for key in ('structure', 'targets', 'sublattices', 'composition')
                            if (path := getattr(args, key)) is not None}
        args.output.mkdir(parents=True)
        for candidate in report['candidates']:
            cid = candidate['id']
            if cid not in witnesses:
                continue
            directory = args.output / cid
            directory.mkdir()
            for filename, fmt in [('best.extxyz', 'extxyz'), ('best.cif', 'cif'), ('best.POSCAR', 'vasp')]:
                write(directory / filename, witnesses[cid], format=fmt, **({'direct': True, 'sort': False} if fmt == 'vasp' else {}))
            (directory / 'sublattices.json').write_text(json.dumps(candidate['sublattices'], indent=2) + '\n')
            candidate['files'] = {str(p.relative_to(args.output)): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                  for p in sorted(directory.iterdir())}
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('cell       atoms    error        aspect   shortest-repeat(A)  status')
        for c in report['candidates']:
            error = '-' if c['best_error'] is None else f"{c['best_error']:.6g}"
            print(f"{c['id']:10} {c['atoms']:5} {error:>12} {c['geometry']['aspect_ratio']:8.3f} "
                  f"{c['geometry']['shortest_repeat']:19.3f}  {c['status']}" + (' *' if c.get('pareto') else ''))
        print(f"* Pareto candidates among evaluated cells; report: {args.output / 'result.json'}")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
