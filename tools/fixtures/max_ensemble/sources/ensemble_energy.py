#!/usr/bin/env python3
"""Reproducible MAX-phase ensemble preparation and MaterialsFramework energies."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from ase.io import read, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args):
    from ase.spacegroup import crystal
    from ase import Atoms
    from higher_order import export_figures
    from sro_ensemble import generate_ensemble
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Preparation requires a new output directory')
    out.mkdir(parents=True)
    primitive = crystal(['Ti', 'Al', 'C'], [(1/3, 2/3, .08), (1/3, 2/3, .75), (0, 0, 0)],
                        spacegroup=194, cellpar=[3.05, 3.05, 13.6, 90, 90, 120])
    kinds = primitive.arrays['spacegroup_kinds']
    species = ['Ti', 'Zr', 'Al', 'Si', 'C']
    counts = [8, 8, 4, 4, 8]
    model = export_figures(out/'FIGURES', primitive, species, counts, [2, 2, 1], [3.5, 3.5],
        sublattices=[(0, np.flatnonzero(kinds == 0).tolist(), [0, 1], [8, 8]),
                     (1, np.flatnonzero(kinds == 1).tolist(), [2, 3], [4, 4])],
        spectator_basis_species=[(int(i), 4) for i in np.flatnonzero(kinds == 2)])
    labels, symbols = ['']*32, ['']*32
    for gid, group in enumerate(model['sublattice_groups']):
        indices = np.flatnonzero(np.array(model['site_groups']) == gid)
        alphabet = [species[i] for i in group['species']]
        label = 'M' if 'Ti' in alphabet else 'A' if 'Al' in alphabet else 'C'
        values = [species[i] for i in np.repeat(group['species'], group['counts'])]
        for i, symbol in zip(indices, values):
            labels[i], symbols[i] = label, symbol
    atoms = Atoms(symbols, scaled_positions=model['fractional_sites'], cell=model['cell'], pbc=True)
    config = dict(site_labels=labels, allowed={'M': ['Ti', 'Zr'], 'A': ['Al', 'Si'], 'C': ['C']})
    targets = [dict(source=s, destination=d, shell=1, center_species=a, neighbor_species=b, target=0)
               for s, d, a, b in [('M', 'M', 'Ti', 'Zr'), ('A', 'A', 'Al', 'Si'), ('M', 'A', 'Ti', 'Si')]]
    report, structures = generate_ensemble(atoms, cutoff=3.5, shells=2, targets=targets,
        sublattices=config, error_tolerance=args.tolerance, count=args.count,
        method='sa', starts=8, steps=3000, seed=42, max_pool=32, max_seconds=30,
        figures=out/'FIGURES', species=species)
    write(out/'initial.extxyz', atoms)
    for entry, structure in zip(report['selected'], structures):
        path = out/entry['id']; path.mkdir()
        write(path/'structure.extxyz', structure)
        entry['structure_sha256'] = sha(path/'structure.extxyz')
    report['prototype'] = dict(formula='(Ti0.5Zr0.5)2(Al0.5Si0.5)C', repeat=[2,2,1],
        cellpar=[3.05,3.05,13.6,90,90,120], metal_z=.08,
        source='m_scraps capture_analysis_oracle.py MAX211 prototype; prescribed common geometry, not an optimized alloy cell')
    save(out/'result.json', report)
    if len(structures) != args.count:
        raise RuntimeError(f'Only {len(structures)} eligible structures; inspect preparation before energy calculations')
    print(json.dumps(dict(selected=len(structures), errors=[e['metrics']['max_error'] for e in report['selected']],
                          orders=sorted({c['order'] for c in report['diversity']['columns']}))), flush=True)


def energy_summary(rows):
    def stats(values):
        if not values:
            return None
        a = np.array(values) * 1000
        return dict(count=len(a), mean_meV_per_atom=float(a.mean()),
                    range_meV_per_atom=float(np.ptp(a)), population_std_meV_per_atom=float(a.std()))
    converged = [r for r in rows if r['force_converged'] and r['site_mapping_valid']]
    return dict(unrelaxed_all=stats([r['unrelaxed_eV_per_atom'] for r in rows]),
        unrelaxed_matched_converged=stats([r['unrelaxed_eV_per_atom'] for r in converged]),
        relaxed_matched_converged=stats([r['relaxed_eV_per_atom'] for r in converged]),
        matched_ids=[r['id'] for r in converged], excluded_ids=[r['id'] for r in rows if r not in converged],
        interpretation='Descriptive configuration spread under one ML potential and a prescribed common cell. Not uncertainty, formation energy, or equilibrium sampling.')


def evaluate(args):
    import torch
    from materialsframework.calculators import MACECalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.geometry import find_mic
    from structure_analysis import map_sites
    torch.set_num_threads(args.threads)
    root, out = args.ensemble.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = json.loads((root/'result.json').read_text())
    model = args.model.resolve()
    manifest = dict(ensemble_sha256=sha(root/'result.json'), model_sha256=sha(model),
        model_file=str(model), script_sha256=sha(__file__), fmax=args.fmax, steps=args.steps,
        device='cpu', dtype='float64', relax_cell=False, optimizer='FIRE', threads=args.threads,
        site_mapping_tolerance_A=.75,
        versions={name: importlib.metadata.version(name) for name in ('materialsframework','mace-torch','torch','ase','numpy')})
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text()) != manifest:
            raise ValueError('Resume settings/model/source changed; choose a new output directory')
    else:
        save(out/'manifest.json', manifest)
    calc = MACECalculator(model=str(model), device='cpu', default_dtype='float64',
        relax_cell=False, fix_symmetry=False, fmax=args.fmax, steps=args.steps, optimizer='FIRE', verbose=True)
    available = set(int(z) for z in calc.calculator.z_table.zs)
    rows = []
    for index, entry in enumerate(report['selected']):
        directory = out/entry['id']; directory.mkdir(exist_ok=True)
        source = root/entry['id']/'structure.extxyz'
        if sha(source) != entry['structure_sha256']:
            raise ValueError('Ensemble witness checksum changed')
        atoms = read(source)
        if not set(atoms.numbers) <= available:
            raise ValueError('Model does not support all structure elements')
        if (directory/'result.json').exists():
            rows.append(json.loads((directory/'result.json').read_text()))
            continue
        save(out/'status.json', dict(state='running', current=entry['id'], completed=index,
                                     total=len(report['selected']), pid=os.getpid()))
        started = time.monotonic()
        initial = calc.calculate(atoms.copy())
        e0 = float(initial['energy']) / len(atoms)
        f0 = float(np.max(np.linalg.norm(initial['forces'], axis=1)))
        save(directory/'initial.json', dict(energy_eV_per_atom=e0, max_force_eV_A=f0))
        write(directory/'initial.extxyz', atoms)
        calc.traj_file = str(directory/'trajectory.pkl')
        result = calc.relax(atoms.copy(), logfile=str(directory/'relaxation.log'))
        final = AseAtomsAdaptor.get_atoms(result['final_structure'])
        if final.get_chemical_symbols() != atoms.get_chemical_symbols() or not np.allclose(final.cell, atoms.cell, atol=1e-10, rtol=0):
            raise RuntimeError('Fixed-cell relaxation changed cell or site species order')
        final_forces = np.asarray(result['forces'])
        final_energy = float(result['energy']) / len(final)
        f1 = float(np.max(np.linalg.norm(final_forces, axis=1)))
        if not all(np.isfinite(v) for v in [e0, f0, final_energy, f1]):
            raise RuntimeError('Nonfinite model result')
        displacement = find_mic(final.positions-atoms.positions, atoms.cell, pbc=True)[1]
        mapping_valid, mapping_error = True, None
        try:
            mapping, _ = map_sites(final, atoms, tolerance=.75)
            mapping_valid = bool(np.array_equal(mapping, np.arange(len(atoms))))
        except ValueError as exc:
            mapping_valid, mapping_error = False, str(exc)
        write(directory/'relaxed.extxyz', final)
        row = dict(id=entry['id'], atoms=len(atoms), formula=atoms.get_chemical_formula(),
            unrelaxed_eV_per_atom=e0, relaxed_eV_per_atom=final_energy,
            relaxation_meV_per_atom=1000*(final_energy-e0),
            initial_max_force_eV_A=f0, final_max_force_eV_A=f1,
            force_converged=f1 <= args.fmax, framework_converged=bool(calc.converged),
            site_mapping_valid=mapping_valid, mapping_error=mapping_error,
            max_displacement_A=float(displacement.max()), rms_displacement_A=float(np.sqrt(np.mean(displacement**2))),
            initial_sro_metrics=entry['metrics'], initial_correlations=entry['diversity_vector'],
            input_sha256=sha(source), relaxed_sha256=sha(directory/'relaxed.extxyz'),
            seconds=time.monotonic()-started)
        save(directory/'result.json', row); rows.append(row)
        save(out/'summary.json', dict(summary=energy_summary(rows), structures=rows))
        print(json.dumps(row), flush=True)
    summary = energy_summary(rows)
    vectors = np.array([r['initial_correlations'] for r in rows])
    summary['correlation_spread'] = dict(columns=report['diversity']['columns'],
        min=vectors.min(axis=0).tolist(), max=vectors.max(axis=0).tolist(), population_std=vectors.std(axis=0).tolist())
    save(out/'summary.json', dict(summary=summary, structures=rows))
    fields = ['id','unrelaxed_eV_per_atom','relaxed_eV_per_atom','relaxation_meV_per_atom',
              'final_max_force_eV_A','force_converged','site_mapping_valid','max_displacement_A']
    with (out/'energies.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    save(out/'status.json', dict(state='complete', completed=len(rows), total=len(rows),
                                 all_converged=len(summary['matched_ids']) == len(rows)))
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scraps-repo', type=Path, default=Path(os.environ.get('SCRAPS_REPO', Path.home()/'src/tools/scraps-perpair')))
    subs = p.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare-max')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--count', type=int, default=4)
    prep.add_argument('--tolerance', type=float, default=.15)
    run = subs.add_parser('evaluate')
    run.add_argument('--ensemble', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--model', type=Path, required=True, help='Explicit local MACE model file; hash recorded')
    run.add_argument('--fmax', type=float, default=.03)
    run.add_argument('--steps', type=int, default=300)
    run.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.command == 'evaluate' and (not np.isfinite(args.fmax) or args.fmax <= 0 or args.steps < 1 or args.threads < 1):
        p.error('Require positive finite fmax, steps and threads')
    sys.path.insert(0, str(args.scraps_repo.resolve()/'tools'))
    (prepare if args.command == 'prepare-max' else evaluate)(args)


if __name__ == '__main__':
    main()
