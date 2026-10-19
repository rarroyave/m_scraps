"""Frozen benchmark witnesses and finite-cell certificates, without timing gates."""
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enumerate_sro import enumerate_sro
from sro_bounds import sro_bounds
from structure_analysis import analyze_structure

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'small_cells'
CASES = json.loads((FIXTURES / 'cases.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_frozen_small_cell_optimum(case):
    serialized = case['witness_extxyz']
    assert hashlib.sha256(serialized.encode()).hexdigest() == case['witness_sha256']
    witness = read(io.StringIO(serialized), format='extxyz')
    initial = read(io.StringIO(case['initial_extxyz']), format='extxyz')
    kwargs = dict(cutoff=case['cutoff'], sublattices=case['sublattices'], shells=1)
    labels = case['sublattices']['site_labels']
    for label in set(labels):
        indices = [i for i, value in enumerate(labels) if value == label]
        assert sorted(witness[i].symbol for i in indices) == sorted(initial[i].symbol for i in indices)
    metrics = analyze_structure(witness, sro_targets=case['targets'], **kwargs)['sro_comparison']['metrics']
    for name in ['max_error', 'rms_error', 'mean_absolute_error']:
        assert metrics[name] == pytest.approx(case['optimum'], abs=1e-12)
    bound = sro_bounds(witness, targets=case['targets'], **kwargs)['result']['lower_bound']
    assert bound == pytest.approx(case['lower_bound'], abs=1e-12)
    if len(witness) <= 16:
        # Re-establish the archived optimum by full labeled traversal, not SA.
        result, _ = enumerate_sro(initial, targets=case['targets'], symmetry='none',
                                  stop_at_bound=False, max_configurations=20000, **kwargs)
        assert result['status'] == 'certified_minimum'
        assert result['best_error'] == pytest.approx(case['optimum'], abs=1e-12)
    else:
        # A feasible witness reaching a rigorous lower bound proves optimality.
        assert metrics['max_error'] == pytest.approx(bound, abs=1e-12)
