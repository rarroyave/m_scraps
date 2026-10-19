import json, sys
from pathlib import Path
import numpy as np
from ase.io import read, write
root = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home()/'runs/scraps-perpair/ensemble-20261019/source/tools'))
sys.path.insert(0, str(root))
from atat_reference import figure_model
from ase import Atoms
from ase.geometry import find_mic
from sro_ensemble import generate_ensemble
from structure_analysis import analyze_structure
source = Path.home()/'runs/sqs-to-property-demo/max-ensemble-20261019/exact-ensemble'
old = json.loads((source/'result.json').read_text())
saved = read(source/'initial.extxyz')
model = figure_model(source/'FIGURES')
atoms = Atoms(saved.get_chemical_symbols(), scaled_positions=model['positions'], cell=model['cell'], pbc=True)
assert np.max(find_mic(atoms.positions-saved.positions, atoms.cell, pbc=True)[1]) < 1e-7
# Reconstruct the exact parent coordinates; extxyz rounding can wrap a zero to one.

options = dict(old['settings'], targets=old['targets'], sublattices=old['sublattices'],
               error_tolerance=0, count=4, method='sa', figures=source/'FIGURES',
               species=['Ti','Zr','Al','Si','C'])
rows, reports = [], []
for name, mode, seed in [('diversity','diversity',17), ('random17','random',17),
                         ('random23','random',23), ('random41','random',41),
                         ('repeat17','random',17)]:
    report, structures = generate_ensemble(atoms, **options, selection=mode, selection_seed=seed)
    assert len(structures) == 4
    if reports:
        assert report['selection']['pool_occupations_sha256'] == reports[0]['selection']['pool_occupations_sha256']
    directory = root/name; directory.mkdir(exist_ok=True)
    for i, structure in enumerate(structures):
        assert np.allclose(structure.cell, atoms.cell)
        analysis = analyze_structure(structure, cutoff=3.5, shells=2,
            sublattices=old['sublattices'], sro_targets=old['targets'])
        assert analysis['sro_comparison']['metrics']['max_error'] == 0
        for group, composition in analysis['composition'].items():
            assert composition['counts'] == old['sublattices']['counts'][group]
        write(directory/f'{i}.extxyz', structure)
    hashes = [r['occupations_sha256'] for r in report['selected']]
    assert len(set(hashes)) == 4
    if name == 'repeat17': assert hashes == [r['occupations_sha256'] for r in reports[1]['selected']]
    d = np.array(report['diversity']['spread']['pairwise_rms_distances'])
    pair_distances = d[np.triu_indices(4,1)]
    rows.append(dict(name=name, selection=mode, selection_seed=seed,
        atoms=len(atoms), pool_size=report['search']['pool_size'],
        max_target_error=max(r['metrics']['max_error'] for r in report['selected']),
        mean_pairwise_correlation_rms=float(pair_distances.mean()),
        min_pairwise_correlation_rms=float(pair_distances.min()),
        selected_pool_indices=[r['pool_index'] for r in report['selected']],
        elapsed_seconds=report['elapsed_seconds']))
    reports.append(report)
    (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(rows[-1]), flush=True)
(root/'summary.json').write_text(json.dumps(dict(rows=rows, same_pool=True,
    seed17_reproduced=True, all_constraints_passed=True),indent=2)+'\n')
