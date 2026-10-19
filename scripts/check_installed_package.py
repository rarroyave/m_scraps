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
subprocess.run([command, 'benchmark', '--cases', 'bcc_128', '--seconds', '3', '--ranks', '2',
                '--output', str(root/'fresh')], check=True)
fresh = json.loads((root/'fresh/comparison.json').read_text())['fresh_native']
assert len(fresh) == 2
assert all(row['candidates'] and not row['errors'] for row in fresh)
for row in fresh:
    assert all('rms_error' in c['metrics'] for c in row['candidates'])
print('Installed API, native MPI, frozen oracle, and portable benchmark passed:', root)
