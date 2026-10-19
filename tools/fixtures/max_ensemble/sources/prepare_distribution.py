import hashlib, json, sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write
root = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home()/'runs/scraps-perpair/ensemble-20261019/source/tools'))
sys.path.insert(0, str(root))
from atat_reference import figure_model
from sro_ensemble import generate_ensemble
from structure_analysis import analyze_structure
source = Path.home()/'runs/sqs-to-property-demo/max-ensemble-20261019/exact-ensemble'
old = json.loads((source/'result.json').read_text())
model = figure_model(source/'FIGURES')
atoms = Atoms(read(source/'initial.extxyz').get_chemical_symbols(), scaled_positions=model['positions'], cell=model['cell'], pbc=True)
out = root/'random32'
if out.exists(): raise ValueError('Use a new output directory')
report, structures = generate_ensemble(atoms, cutoff=3.5, shells=2, targets=old['targets'],
    sublattices=old['sublattices'], error_tolerance=0, count=32, method='sa',
    starts=8, steps=6000, seed=42, max_pool=128, max_seconds=120,
    figures=source/'FIGURES', species=['Ti','Zr','Al','Si','C'], selection='random', selection_seed=17)
assert len(structures) == 32 and report['search']['pool_size'] > 32
assert len({r['occupations_sha256'] for r in report['selected']}) == 32
out.mkdir()
for entry, atoms in zip(report['selected'], structures):
    analysis = analyze_structure(atoms, cutoff=3.5, shells=2, sublattices=old['sublattices'], sro_targets=old['targets'])
    assert analysis['sro_comparison']['metrics']['max_error'] == 0
    for group, composition in analysis['composition'].items():
        assert composition['counts'] == old['sublattices']['counts'][group]
    path = out/entry['id']; path.mkdir()
    write(path/'structure.extxyz', atoms)
    entry['structure_sha256'] = hashlib.sha256((path/'structure.extxyz').read_bytes()).hexdigest()
report['prototype'] = old['prototype']
(out/'FIGURES').write_bytes((source/'FIGURES').read_bytes())
(out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(selected=len(structures), pool_size=report['search']['pool_size'], stop_reason=report['stop_reason'], max_error=0)), flush=True)
