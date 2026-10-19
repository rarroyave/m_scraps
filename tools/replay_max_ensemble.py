"""Replay the frozen random MAX ensemble; optional external MACE evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ase import Atoms
from ase.io import read, write

from atat_reference import figure_model
from sro_ensemble import generate_ensemble


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New run directory')
    parser.add_argument('--model', type=Path, help='Optional local MACE checkpoint; requires MaterialsFramework environment')
    args = parser.parse_args(argv)
    fixture = Path(__file__).resolve().parent/'fixtures/max_ensemble'
    if args.output.exists():
        parser.error('Choose a new output directory')
    if args.model is not None and not args.model.is_file():
        parser.error('Model checkpoint does not exist')
    saved = json.loads((fixture/'random32/result.json').read_text())
    figures = fixture/'random32/FIGURES'
    model = figure_model(figures)
    # Recover exact parent coordinates: extxyz rounding can wrap zero to one.
    atoms = Atoms(read(fixture/'initial.extxyz').get_chemical_symbols(),
                  cell=model['cell'], scaled_positions=model['positions'], pbc=True)
    report, structures = generate_ensemble(atoms, **saved['settings'],
        targets=saved['targets'], sublattices=saved['sublattices'],
        error_tolerance=saved['error_tolerance'], count=saved['requested_count'],
        method=saved['search']['method'], figures=figures, species=saved['diversity']['species'])
    out = args.output/'ensemble'; out.mkdir(parents=True)
    for entry, structure in zip(report['selected'], structures):
        path = out/entry['id']; path.mkdir()
        write(path/'structure.extxyz', structure)
        entry['structure_sha256'] = hashlib.sha256((path/'structure.extxyz').read_bytes()).hexdigest()
    report['prototype'] = saved['prototype']
    report['replay'] = dict(reference_selector_commit='da29a9a',
        same_ordered_pool=report['selection']['pool_occupations_sha256'] == saved['selection']['pool_occupations_sha256'],
        same_selection=[r['occupations_sha256'] for r in report['selected']] ==
                       [r['occupations_sha256'] for r in saved['selected']])
    (out/'FIGURES').write_bytes(figures.read_bytes())
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(selected=len(structures), **report['replay'])), flush=True)
    if len(structures) != saved['requested_count']:
        raise RuntimeError('Incomplete ensemble; inspect result.json before evaluating energies')
    if args.model is not None:
        subprocess.run([sys.executable, str(fixture/'sources/ensemble_energy.py'),
            '--scraps-repo', str(Path(__file__).resolve().parents[1]), 'evaluate',
            '--ensemble', str(out), '--output', str(args.output/'energies'),
            '--model', str(args.model.resolve()), '--fmax', '0.03', '--steps', '300', '--threads', '4'], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
