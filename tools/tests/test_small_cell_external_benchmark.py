"""Archived all-method witnesses checked against ATAT without optional engines."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from ase.io import read

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from atat_reference import figure_model, correlations
from structure_analysis import analyze_structure

ROOT = TOOLS / 'fixtures' / 'small_cells_external'
ROWS = json.loads((ROOT/'summary.json').read_text())
REFERENCES = {c['id']:c for c in json.loads((TOOLS/'fixtures/small_cells/cases.json').read_text())['cases']}
ORACLES = json.loads((ROOT/'oracles.json').read_text())


def test_complete_small_cell_method_matrix():
    assert len(ROWS) == 48
    assert {(r['case'],r['method']) for r in ROWS} == {
        (case, method) for case in REFERENCES
        for method in ['scraps_cuckoo','scraps_sa','mcsqs','icet_atat']}
    assert all(r['seed']==42 and r['workers']==1 and r['budget_seconds']==5 for r in ROWS)
    manifest = json.loads((ROOT/'fixture_manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest


@pytest.fixture(scope='module')
def native_probe(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler unavailable')
    binary = tmp_path_factory.mktemp('small-cell-probe')/'probe'
    subprocess.run([compiler,'-std=c++17','-O2','-I',str(TOOLS.parent/'SCRAPs'),
                    str(TOOLS/'benchmark/figure_probe.cpp'),'-o',str(binary)],check=True)
    return binary


@pytest.mark.parametrize('case', list(REFERENCES))
def test_each_method_witness_against_atat(case, native_probe, tmp_path):
    ref = REFERENCES[case]; parsed = figure_model(ROOT/(case+'.figures.gz'))
    metadata = json.loads((ROOT/(case+'.figures.json')).read_text())
    occupations=[]; oracles=[]
    for row in [r for r in ROWS if r['case']==case]:
        assert row['status'] == 'verified_output' and not row['errors']
        assert hashlib.sha256(row['witness_extxyz'].encode()).hexdigest() == row['witness_sha256']
        atoms = read(io.StringIO(row['witness_extxyz']),format='extxyz')
        config = dict(site_labels=row['site_labels'],allowed=ref['sublattices']['allowed'])
        for group, allowed in config['allowed'].items():
            symbols = [a.symbol for a,label in zip(atoms,config['site_labels']) if label==group]
            assert set(symbols)==set(allowed)
            assert all(symbols.count(s)==len(symbols)//len(allowed) for s in allowed)
        actual = analyze_structure(atoms,cutoff=ref['cutoff'],shells=1,
                                   sublattices=config,sro_targets=ref['targets'])['sro_comparison']['metrics']
        occ = [metadata['species'].index(s) for s in atoms.get_chemical_symbols()]
        oracle = ORACLES[case+'/'+row['method']]
        np.testing.assert_allclose(correlations(parsed,occ),oracle,atol=2e-10,rtol=0)
        error = abs(oracle[0]-ref['targets'][0]['target'])
        for metric in ['max_error','rms_error','mean_absolute_error']:
            assert actual[metric] == pytest.approx(error,abs=2e-10)
            assert actual[metric] == pytest.approx(row[metric],abs=2e-10)
        assert actual['max_error']+2e-10 >= ref['optimum']
        assert row['gap_to_optimum'] == pytest.approx(actual['max_error']-ref['optimum'],abs=2e-10)
        occupations.append(occ); oracles.append(oracle)
    figures = tmp_path/'FIGURES'; figures.write_bytes(gzip.decompress((ROOT/(case+'.figures.gz')).read_bytes()))
    run = subprocess.run([str(native_probe),str(figures),str(parsed['sites']),str(parsed['species']),'8'],
                         input='\n'.join(' '.join(map(str,occ)) for occ in occupations)+'\n',
                         text=True,capture_output=True,check=True,timeout=30)
    observed=np.array([np.fromstring(line,sep=' ') for line in run.stdout.splitlines()])
    np.testing.assert_allclose(observed[:,:-1],oracles,atol=2e-10,rtol=0)
    assert max(observed[:,-1]) < 2e-10
