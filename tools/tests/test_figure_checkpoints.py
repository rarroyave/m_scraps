"""Native snapshots preserve search trajectories and survive a hard deadline."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_atat_reference as bench
from atat_reference import FIXTURES, manifest, figure_model, correlations
from scraps_ase import _find_mpirun

pytestmark = pytest.mark.skipif(not (bench.REPO/'SCRAPs/scraps').exists() or not _find_mpirun(),
                                reason='native binary and matched MPI required')
PROFILE = dict(name='test', wd=0, wn=1)


def prepare(directory, interval, kernel="sa"):
    directory.mkdir()
    case=next(c for c in manifest()['cases'] if c['id']=='fcc_binary_32')
    (directory/'FIGURES').write_bytes(gzip.decompress((FIXTURES/(case['id']+'.figures.gz')).read_bytes()))
    (directory/'POSCAR').write_bytes((FIXTURES/(case['id']+'.poscar')).read_bytes())
    text=bench.input_text(case,PROFILE,42)+f'MC_KERNEL {kernel}\n'
    if interval is not None: text+=f'FIGURE_CHECKPOINT_SECONDS {interval}\n'
    (directory/'INPUT').write_text(text)
    return case


def run(directory):
    return subprocess.run([_find_mpirun(),'-np','2',str(bench.REPO/'SCRAPs/scraps')],
                          cwd=directory,capture_output=True,text=True,timeout=60)


@pytest.mark.parametrize("kernel", ["sa", "cuckoo"])
def test_snapshots_match_independent_correlations_and_preserve_rng(tmp_path,kernel):
    plain=tmp_path/'plain'; enabled=tmp_path/'enabled'
    prepare(plain,None,kernel);case=prepare(enabled,.001,kernel)
    for d in [plain,enabled]:
        result=run(d)
        assert result.returncode==0,result.stdout+result.stderr
    assert not list(plain.glob('FIGURE_CHECKPOINT*'))
    assert (plain/'FIGURE_OCCUPATIONS.txt').read_bytes()==(enabled/'FIGURE_OCCUPATIONS.txt').read_bytes()
    model=figure_model(enabled/'FIGURES')
    paths=list(enabled.glob('FIGURE_CHECKPOINT.rank*.json'))
    assert len(paths)==2
    for path in paths:
        data=json.loads(path.read_text());occ=np.array(data['occupations'])
        bench.validate_counts(model,case,occ)
        np.testing.assert_allclose(correlations(model,occ),data['correlations'],atol=2e-12,rtol=0)


@pytest.mark.parametrize('interval',['-1','nan','garbage'])
def test_invalid_interval(tmp_path,interval):
    d=tmp_path/'invalid';prepare(d,interval)
    result=run(d)
    assert result.returncode!=0
    assert 'FIGURE_CHECKPOINT_SECONDS' in result.stdout+result.stderr


def test_timeout_retains_independently_valid_checkpoint(tmp_path,monkeypatch):
    case=next(c for c in manifest()['cases'] if c['id']=='fcc_binary_32')
    original=bench.input_text
    monkeypatch.setattr(bench,'input_text',lambda *a: original(*a).replace('TOTALIter 2','TOTALIter 100000')+'MC_KERNEL sa\n')
    args=SimpleNamespace(seconds=3,ranks=2,mpirun=_find_mpirun(),binary=bench.REPO/'SCRAPs/scraps',checkpoint_seconds=.05)
    record=bench.run_trial(case,PROFILE,42,args,tmp_path/'timed')
    assert record['unfinished_restart_at_deadline']
    assert not record['errors']
    assert record['candidates']
    assert all(c['available_seconds']<=3 for c in record['candidates'])
    assert max(c['native_correlation_discrepancy'] for c in record['candidates'])<2e-12
