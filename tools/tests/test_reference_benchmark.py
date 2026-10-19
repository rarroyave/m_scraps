"""Time-to-target accounting and a real ATAT-free CLI smoke run."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_atat_reference import first_passage, summary, validate_counts, REPO
from atat_reference import FIXTURES, figure_model, manifest
from scraps_ase import _find_mpirun


def test_first_passage_includes_timeouts_and_earlier_candidates():
    candidates=[dict(available_seconds=1.2,metrics={'max_error':.01}),
                dict(available_seconds=2.1,metrics={'max_error':.04})]
    assert first_passage(candidates,'max_error',.025,5)['seconds']==1.2
    assert first_passage(candidates,'max_error',.001,5)['right_censored']
    assert first_passage([],'max_error',.025,5)['seconds'] is None


def test_summary_does_not_discard_failed_trials():
    records=[]
    for success in [True,False,False]:
        p=dict(threshold=.01,success=success,seconds=1 if success else None)
        records.append(dict(case='example',profile='test',passage={m:[p]*4 for m in ['max_error','weighted_l1']}))
    for row in summary(records):
        assert row['trials']==3 and row['successes']==1 and row['right_censored']==2
        assert row['median_observed_passage_seconds'] is None
        assert row['wilson95'][0]<1/3<row['wilson95'][1]


def test_reject_global_correct_but_local_wrong_composition():
    model=figure_model(FIXTURES/'b2_shared_species.figures.gz')
    case=next(c for c in manifest()['cases'] if c['id']=='b2_shared_species')
    with pytest.raises(ValueError,match='Local composition'):
        validate_counts(model,case,np.array([0]*4+[1]*4+[0]*4+[1]*4))


@pytest.mark.skipif(not (REPO/'SCRAPs/scraps').exists() or _find_mpirun() is None,
                    reason='native executable or matched MPI launcher unavailable')
def test_benchmark_cli_no_atat_or_icet(tmp_path):
    result=subprocess.run([sys.executable,str(REPO/'tools/benchmark_atat_reference.py'),
                           '--output',str(tmp_path/'run'),'--cases','fcc_binary_32',
                           '--seconds','5','--trials','1','--ranks','2','--seed','42'],
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    record=json.loads((tmp_path/'run/fcc_binary_32/0/result.json').read_text())
    assert record['candidates']  # Fixed seed exercises a real output, not only timeout handling.
    assert all(c['available_seconds']<=5 for c in record['candidates'])
    assert max(c['native_correlation_discrepancy'] for c in record['candidates'])<2e-12
    provenance=json.loads((tmp_path/'run/provenance.json').read_text())
    assert provenance['binary_sha256'] and provenance['fixture_manifest_sha256']
    comparison=json.loads((tmp_path/'run/historical-comparison.json').read_text())
    assert comparison['rows'] and not comparison['rows'][0]['same_allocated_cores']
