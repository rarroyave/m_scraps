"""ATAT-audited higher-order outputs from all four search methods."""
import gzip,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atat_reference import figure_model,correlations,metrics
from benchmark_atat_reference import validate_counts
ROOT=Path(os.environ.get('HIGHER_ORDER_FIXTURES',str(Path(__file__).resolve().parents[1]/'fixtures/higher_order_screen')))
MANIFEST=json.loads((ROOT/'manifest.json').read_text())
def test_fixture_integrity():
    for name,digest in MANIFEST['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
@pytest.mark.parametrize('case',MANIFEST['cases'])
def test_all_methods_higher_order_correlations_against_atat(case):
    data=json.loads(gzip.decompress((ROOT/(case+'.json.gz')).read_bytes()));model=figure_model(ROOT/(case+'.figures.gz'))
    assert {x['method'] for x in data['examples']}==set(MANIFEST['methods'])
    assert {f['order'] for f in model['features']}=={2,3,4}
    for example in data['examples']:
        if example['outcome']=='no_output_within_budget':
            assert example['budget_seconds']==30
            for rows in example['passage'].values():
                assert all(not r['success'] and r['right_censored'] and r['seconds'] is None for r in rows)
            continue
        occ=np.array(example['occupations']);validate_counts(model,data['case'],occ)
        vector=correlations(model,occ);oracle=np.array(example['oracle_correlations'])
        np.testing.assert_allclose(vector,oracle,atol=2e-10,rtol=0)
        expected=example['metrics'];actual=metrics(model,vector,wd=0,wn=1)
        for key in ['max_error','weighted_l1']:assert abs(actual[key]-expected[key])<2e-10
        errors=oracle-[f['target'] for f in model['features']]
        assert abs(np.sqrt(np.mean(errors**2))-expected['rms_error'])<2e-10
        for order in ['2','3','4']:
            for key in ['max_error','rms']:assert abs(actual['by_order'][order][key]-expected['by_order'][order][key])<2e-10
