"""Run outside the checkout using an isolated installation without icet/ATAT."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

assert importlib.util.find_spec('icet') is None, 'Use a clean environment without optional icet'
import scraps_ase  # Exercise compatibility import before the package CLI.
from m_scraps import analyze_structure, run_sqs_from_atoms
from ase.build import bulk

root = Path(tempfile.mkdtemp(prefix='m_scraps_installed_'))
command = str(Path(sys.executable).parent / 'm_scraps')
subprocess.run([command, 'compare-atat', '--output', str(root/'oracle.json')], check=True)
assert json.loads((root/'oracle.json').read_text())['passed']
subprocess.run([command, 'benchmark', '--report-only', '--output', str(root/'archived')], check=True)
assert len(json.loads((root/'archived/comparison.json').read_text())['archived_reference']) == 8
cell = run_sqs_from_atoms(bulk('Cu', 'fcc', a=3.6, cubic=True), species=['Cu','Zn'],
    element_counts=[16,16], cell_dim=(2,2,2), work_dir=root/'native', ranks=2,
    max_shellnum=2, weights=[1,1], rng_seed=42, nests=8, mcstep=32, total_iter=2, auto_mcstep=0, extra_lines="GITER 1\n")
assert len(cell) == 32
assert analyze_structure(cell, cutoff=3.7)['sites'] == 32
cell.write(root/'bounds-cell.extxyz')
(root/'targets.json').write_text(json.dumps([dict(source='all', destination='all',
    shell=1, center_species='Cu', neighbor_species='Zn', target=0)]))
subprocess.run([command, 'bounds', str(root/'bounds-cell.extxyz'), '--cutoff', '3.7',
    '--targets', str(root/'targets.json'), '--output', str(root/'bounds.json')], check=True)
bound = json.loads((root/'bounds.json').read_text())['result']
assert 0 <= bound['lower_bound'] <= bound['achieved_upper_bound']

small = bulk('Cu', 'bcc', a=2, cubic=True)
small[1].symbol = 'Zn'
small.write(root/'enumeration-cell.extxyz')
subprocess.run([command, 'enumerate', str(root/'enumeration-cell.extxyz'), '--cutoff', '1.8',
    '--targets', str(root/'targets.json'), '--output', str(root/'enumeration')], check=True)
exact = json.loads((root/'enumeration/result.json').read_text())
assert exact['status'] == 'certified_minimum' and exact['best_error_exact'] == '1'
assert exact['covered_configurations'] == 2
assert exact['evaluated_configurations'] == 1


subprocess.run([command, 'benchmark', '--cases', 'bcc_128', '--seconds', '3', '--ranks', '2',
                '--output', str(root/'fresh')], check=True)
fresh = json.loads((root/'fresh/comparison.json').read_text())['fresh_native']
assert len(fresh) == 2
assert all(row['candidates'] and not row['errors'] for row in fresh)
for row in fresh:
    assert all('rms_error' in c['metrics'] for c in row['candidates'])
print('Installed API, native MPI, frozen oracle, and portable benchmark passed:', root)

subprocess.run([command, 'search-cells', str(root/'enumeration-cell.extxyz'),
    '--cutoff', '1.8', '--targets', str(root/'targets.json'), '--min-atoms', '2',
    '--max-atoms', '4', '--max-candidates', '2', '--max-aspect', '1.5',
    '--output', str(root/'cell-search')], check=True)
search = json.loads((root/'cell-search/result.json').read_text())
assert search['candidates'] and not search['global_optimum_certified']
assert all(c['geometry']['aspect_ratio'] <= 1.5 + 1e-10 for c in search['candidates'])
from m_scraps import search_cells
assert callable(search_cells)
