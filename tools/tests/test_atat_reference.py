"""Saved ATAT oracle regression: no ATAT executable, source, network or icet."""
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atat_reference import FIXTURES, manifest, figure_model, correlations


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    repo = Path(__file__).resolve().parents[2]
    binary = tmp_path_factory.mktemp('atat-reference')/'probe'
    subprocess.run([compiler,'-std=c++17','-O2','-I',str(repo/'SCRAPs'),
                    str(repo/'tools/benchmark/figure_probe.cpp'),'-o',str(binary)],check=True)
    return binary


def test_fixture_integrity():
    assert len(manifest()['cases']) == 5


def test_saved_atat_objectives(probe):
    data=json.loads(gzip.decompress((FIXTURES/'objective-reference.json.gz').read_bytes()))
    lines=[]
    for r in data['fixtures']:
        lines.append(' '.join(map(str,[len(r['corr']),r['tol'],r['wr'],r['wn'],r['wd']])))
        lines.extend(' '.join(map(str,v)) for v in zip(r['order'],r['diameter'],r['target'],r['corr']))
    result=subprocess.run([str(probe)],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
    actual=np.fromstring(result.stdout,sep=' ')
    expected=np.array([r['atat_score'] for r in data['fixtures']])
    assert len(actual)==160
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-12)
    assert any(r['atat_terminal'] and not r['all_within_tolerance'] for r in data['fixtures'])


@pytest.mark.parametrize('case',['rocksalt_ternary_binary','diboride_fixed_b'])
def test_saved_atat_vectors_and_incremental_swaps(case,probe,tmp_path):
    manifest()
    reference=json.loads(gzip.decompress((FIXTURES/(case+'.reference.json.gz')).read_bytes()))
    model=figure_model(FIXTURES/(case+'.figures.gz'))
    path=tmp_path/'FIGURES';path.write_bytes(gzip.decompress((FIXTURES/(case+'.figures.gz')).read_bytes()))
    configs=reference['configurations']
    result=subprocess.run([str(probe),str(path),str(model['sites']),str(model['species'])],
                          input='\n'.join(' '.join(map(str,r['occupations'])) for r in configs)+'\n',
                          text=True,capture_output=True,check=True)
    native=np.array([np.fromstring(line,sep=' ') for line in result.stdout.splitlines()])
    expected=np.array([r['correlations'] for r in configs])
    np.testing.assert_allclose(native[:,:-1],expected,atol=2e-12,rtol=0)
    assert max(native[:,-1])<2e-12
    for r in configs:
        np.testing.assert_allclose(correlations(model,r['occupations']),r['correlations'],atol=2e-12,rtol=0)
    np.testing.assert_allclose([f['target'] for f in model['features']],reference['targets'],atol=2e-12,rtol=0)


def test_shared_species_analytic_targets():
    model=figure_model(FIXTURES/'b2_shared_species.figures.gz')
    for f in model['features']:
        target=sum(c*np.prod([.5 if site<8 else -.5 for site,_ in factors]) for c,factors in f['terms'])
        assert f['target']==pytest.approx(target,abs=2e-12)
