"""Frozen external reference checks: no ATAT or icet needed in normal CI."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_atat import FIXTURES, compare_case
from atat_reference import figure_model


@pytest.mark.parametrize('case', ['diboride', 'max211', 'heusler'])
def test_analysis_against_atat_golden_results(case):
    reports = compare_case(FIXTURES, case)
    assert len(reports) == 2
    assert all(row['passed'] for row in reports)
    assert all(row['sro_entries'] > 0 for row in reports)
    model = figure_model(FIXTURES / f'{case}.figures.gz')
    assert {f['order'] for f in model['features']} == {2, 3, 4}
    data = json.loads((FIXTURES / f'{case}.json').read_text())
    for config in data['configurations']:
        raw = np.loadtxt(FIXTURES / f"{case}-{config['seed']}.corrdump.txt")
        np.testing.assert_array_equal(raw, config['correlations'])
        residual = raw - [f['target'] for f in model['features']]
        assert config['metrics']['max_error'] == pytest.approx(max(abs(residual)), abs=1e-14)
        assert config['metrics']['rms_error'] == pytest.approx(np.sqrt(np.mean(residual**2)), abs=1e-14)


def test_custom_structure_requires_live_oracle():
    with pytest.raises(ValueError, match='require --corrdump'):
        compare_case(FIXTURES, 'max211', structure='own.cif')


@pytest.fixture(scope='module')
def native_probe(tmp_path_factory):
    import shutil
    import subprocess
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    root = Path(__file__).resolve().parents[2]
    binary = tmp_path_factory.mktemp('analysis-probe') / 'probe'
    subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(root/'SCRAPs'),
                    str(root/'tools/benchmark/figure_probe.cpp'), '-o', str(binary)], check=True)
    return binary


@pytest.mark.parametrize('case', ['diboride', 'max211', 'heusler'])
def test_native_correlations_and_swap_updates_against_atat(case, native_probe, tmp_path):
    import gzip
    import subprocess
    path = tmp_path/'FIGURES'
    path.write_bytes(gzip.decompress((FIXTURES/f'{case}.figures.gz').read_bytes()))
    model = figure_model(path)
    configurations = json.loads((FIXTURES/f'{case}.json').read_text())['configurations']
    result = subprocess.run([str(native_probe), str(path), str(model['sites']), str(model['species']), '16'],
        input='\n'.join(' '.join(map(str, row['occupations'])) for row in configurations)+'\n',
        text=True, capture_output=True, check=True)
    native = np.array([np.fromstring(line, sep=' ') for line in result.stdout.splitlines()])
    np.testing.assert_allclose(native[:, :-1], [row['correlations'] for row in configurations], atol=2e-10, rtol=0)
    assert max(native[:, -1]) < 2e-10
