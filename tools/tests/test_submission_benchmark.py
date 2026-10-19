"""Frozen submission results validate Python and native figure accounting."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atat_reference import figure_model, correlations, metrics
from benchmark_atat_reference import validate_counts

ROOT=Path(__file__).resolve().parents[1]/'fixtures/submission_benchmark'
MANIFEST=json.loads((ROOT/'manifest.json').read_text()) if (ROOT/'manifest.json').exists() else None
pytestmark=pytest.mark.skipif(MANIFEST is None,reason='submission fixtures not installed')


@pytest.fixture(scope='module')
def native_probe(tmp_path_factory):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    repo=Path(__file__).resolve().parents[2]
    binary=tmp_path_factory.mktemp('submission-probe')/'probe'
    subprocess.run([compiler,'-std=c++17','-O2','-I',str(repo/'SCRAPs'),str(repo/'tools/benchmark/figure_probe.cpp'),'-o',str(binary)],check=True)
    return binary


def test_submission_matrix_and_fixture_integrity():
    assert MANIFEST['trials']==40
    assert len(MANIFEST['cases'])==8
    total=0
    for name,digest in MANIFEST['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for ident in MANIFEST['cases']:
        data=json.loads(gzip.decompress((ROOT/(ident+'.json.gz')).read_bytes()))
        c=data['case'];examples=data['examples'];total+=len(examples)
        expected_seeds={20261019,21261019} if c['sites']==128 and c['target_mode']=='reference' else {20261019}
        assert {(x['method'],x['seed']) for x in examples}=={(m,s) for m in MANIFEST['methods'] for s in expected_seeds}
    assert total==40


@pytest.mark.parametrize('ident',MANIFEST['cases'] if MANIFEST else [])
def test_saved_outputs_against_atat_and_native(ident,native_probe,tmp_path):
    data=json.loads(gzip.decompress((ROOT/(ident+'.json.gz')).read_bytes()))
    model=figure_model(ROOT/(ident+'.figures.gz'));case=data['case']
    assert {f['order'] for f in model['features']}=={2,3,4}
    valid=[]
    for example in data['examples']:
        if example['outcome']=='no_output_within_budget':
            assert example['budget_seconds']==30
            assert all(p['right_censored'] and p['seconds'] is None for rows in example['passage'].values() for p in rows)
            continue
        occ=np.asarray(example['occupations']);validate_counts(model,case,occ)
        oracle=np.asarray(example['oracle_correlations']);actual=correlations(model,occ)
        np.testing.assert_allclose(actual,oracle,atol=2e-10,rtol=0)
        values=metrics(model,actual,wd=0,wn=1);expected=example['metrics']
        for key in ['max_error','weighted_l1']:
            assert values[key]==pytest.approx(expected[key],abs=2e-10)
        residual=actual-np.array([f['target'] for f in model['features']])
        assert np.sqrt(np.mean(residual**2))==pytest.approx(expected['rms_error'],abs=2e-10)
        for order in ['2','3','4']:
            for key in ['max_error','rms']:
                assert values['by_order'][order][key]==pytest.approx(expected['by_order'][order][key],abs=2e-10)
        valid.append(example)
    if case['target_mode']=='reference':
        witness=np.array(data['target_reference_occupations'])
        validate_counts(model,case,witness)
        np.testing.assert_allclose(correlations(model,witness),[f['target'] for f in model['features']],atol=2e-10,rtol=0)
    if not valid:return
    figure_file=tmp_path/'FIGURES';figure_file.write_bytes(gzip.decompress((ROOT/(ident+'.figures.gz')).read_bytes()))
    # A bounded swap sample keeps large-cell native regressions fast.
    result=subprocess.run([str(native_probe),str(figure_file),str(model['sites']),str(model['species']),'16'],
                          input='\n'.join(' '.join(map(str,x['occupations'])) for x in valid)+'\n',text=True,capture_output=True,check=True,timeout=60)
    observed=np.array([np.fromstring(line,sep=' ') for line in result.stdout.splitlines()])
    np.testing.assert_allclose(observed[:,:-1],[x['oracle_correlations'] for x in valid],atol=2e-10,rtol=0)
    assert max(observed[:,-1])<2e-10
