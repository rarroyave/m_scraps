"""Public input routes for independently constrained shared species."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from ase import Atoms
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scraps_ase import make_scraps_inputs


def primitive():
    return Atoms('FeAl', scaled_positions=[[0, 0, 0], [.5, .5, .5]],
                 cell=[2.9]*3, pbc=True)


def test_local_counts_require_figures(tmp_path):
    with pytest.raises(ValueError, match='require figure_cutoffs'):
        make_scraps_inputs(tmp_path, primitive(), [8, 8], (2, 2, 2),
                           sublattices=[(0, [0], [0, 1], [6, 2]),
                                        (1, [1], [0, 1], [2, 6])])


def test_shared_counts_api_and_cli(tmp_path):
    pytest.importorskip('icet', minversion='4.0')
    groups = [(0, [0], [0, 1], [6, 2]), (1, [1], [0, 1], [2, 6])]
    make_scraps_inputs(tmp_path/'api', primitive(), [8, 8], (2, 2, 2),
                       sublattices=groups, figure_species=['Fe', 'Al'],
                       figure_cutoffs=[4.2]*3)
    content = (tmp_path/'api'/'INPUT').read_text()
    assert 'SUBLATTICE 0 BASIS 0 SPECIES 0 1 COUNTS 6 2' in content
    assert 'SUBLATTICE 1 BASIS 1 SPECIES 0 1 COUNTS 2 6' in content
    write(tmp_path/'POSCAR', primitive(), format='vasp')
    (tmp_path/'partition.json').write_text(json.dumps(groups))
    script = Path(__file__).resolve().parents[1]/'run_sqs.py'
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([
        sys.executable, str(script), str(tmp_path/'POSCAR'),
        '--species', 'Fe,Al', '--composition', '0.5,0.5', '--target-atoms', '16',
        '--sublattices', str(tmp_path/'partition.json'), '--cell-dim', '2,2,2',
        '--figure-cutoffs', '4.2,4.2,4.2', '--dry-run',
        '--work-dir', str(tmp_path/'cli')], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'COUNTS 6 2' in (tmp_path/'cli'/'INPUT').read_text()
