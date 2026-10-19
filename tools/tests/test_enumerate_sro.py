"""Exact small-cell ground truths and certification/limit regressions."""
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import enumerate_sro as module
from structure_analysis import analyze_structure


def target(value, source='all', destination='all', a='Cu', b='Zn'):
    return dict(source=source, destination=destination, shell=1,
                center_species=a, neighbor_species=b, target=value)


def ring():
    return Atoms('Cu2Zn2', positions=[[i, 0, 0] for i in range(4)], cell=[4, 8, 8], pbc=True)


def test_multinomial_count_and_unique_generation():
    composition = {'A': {'sites': 3, 'counts': {'Cu': 1, 'Zn': 1, 'Al': 1}},
                   'B': {'sites': 2, 'counts': {'Cu': 1, 'Zn': 1}},
                   'fixed': {'sites': 1, 'counts': {'C': 1, 'Si': 0}}}
    labels = np.array(['A', 'B', 'A', 'fixed', 'A', 'B'])
    configurations = list(module.occupations(labels, composition))
    assert module.configuration_count(composition) == len(configurations) == len(set(configurations)) == 12
    assert all(c[3] == 'C' for c in configurations)
    assert module.configuration_count({'all': {'sites': 16, 'counts': {'Cu': 8, 'Zn': 8}}}) == 12870
    assert module.configuration_count({'all': {'sites': 32, 'counts': {'Cu': 16, 'Zn': 16}}}) == 601080390


def test_unattainable_zero_bound_and_periodic_image_multiplicity():
    atoms = bulk('Cu', 'bcc', a=2, cubic=True)
    atoms[1].symbol = 'Zn'
    result, witness = module.enumerate_sro(atoms, cutoff=1.8, targets=[target(0)])
    assert result['total_configurations'] == result['evaluated_configurations'] == 2
    assert result['relaxed_lower_bound'] == 0
    assert result['best_error_exact'] == result['lower_bound_exact'] == '1'
    assert result['status'] == 'certified_minimum'
    assert result['entries'][0]['pair_count'] == 8
    assert sorted(witness.get_chemical_symbols()) == ['Cu', 'Zn']


@pytest.mark.parametrize('metric', module.METRICS)
@pytest.mark.parametrize('multi', [False, True])
def test_matches_independent_bruteforce(metric, multi):
    atoms = ring()
    config = None
    targets = [target(.3), target(-.2, a='Cu', b='Cu')]
    valid = [v for v in product(['Cu', 'Zn'], repeat=4) if v.count('Cu') == 2]
    if multi:
        config = {'site_labels': ['A', 'B', 'A', 'B'],
                  'allowed': {'A': ['Cu', 'Zn'], 'B': ['Cu', 'Zn']},
                  'counts': {'A': {'Cu': 1, 'Zn': 1}, 'B': {'Cu': 1, 'Zn': 1}}}
        valid = [v for v in valid if v[0] != v[2] and v[1] != v[3]]
        targets = [target(.3, 'A', 'B'), target(-.4, 'B', 'A', a='Zn', b='Cu')]
    measured = []
    for values in valid:
        atoms.set_chemical_symbols(values)
        measured.append(analyze_structure(atoms, cutoff=1.1, sublattices=config,
                         sro_targets=targets)['sro_comparison']['metrics'][metric])
    result, witness = module.enumerate_sro(atoms, cutoff=1.1, targets=targets,
                                          sublattices=config, metric=metric)
    assert result['complete'] and result['optimality_gap'] == 0
    assert result['evaluated_configurations'] == len(valid)
    assert result['best_error'] == pytest.approx(min(measured), abs=1e-14)
    actual = analyze_structure(witness, cutoff=1.1, sublattices=config,
                               sro_targets=targets)['sro_comparison']['metrics']
    for name in module.METRICS:
        assert result['metrics_at_best'][name]['value'] == pytest.approx(actual[name], abs=1e-14)


def test_reference_order_fixed_spectator_and_unequal_coordination():
    ref = Atoms('TiCuZn', positions=[[i, 0, 0] for i in range(3)], cell=[3, 8, 8], pbc=True)
    atoms = ref[[2, 0, 1]]
    atoms.positions += [.01, 0, 0]
    config = {'site_labels': ['A', 'B', 'B'], 'allowed': {'A': ['Ti'], 'B': ['Cu', 'Zn']}}
    result, witness = module.enumerate_sro(atoms, reference=ref, cutoff=1.1,
        sublattices=config, targets=[target(.1, 'A', 'B', a='Ti')])
    assert result['evaluated_configurations'] == 2
    assert result['best_error_exact'] == '1/10'
    assert witness[0].symbol == 'Ti'
    np.testing.assert_allclose(witness.positions, ref.positions)


def test_cap_refuses_before_enumeration(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Search should not begin')
    monkeypatch.setattr(module, 'occupations', forbidden)
    with pytest.raises(ValueError, match='6 configurations exceed'):
        module.enumerate_sro(ring(), cutoff=1.1, targets=[target(0)], max_configurations=5)


def test_time_limit_and_interrupt_never_claim_completion(monkeypatch):
    ticks = iter([0., 10., 11.])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    result, _ = module.enumerate_sro(ring(), cutoff=1.1, targets=[target(0)], max_seconds=1)
    assert result['stop_reason'] == 'time_limit'
    assert result['status'] == 'best_found' and not result['complete']
    assert result['evaluated_configurations'] == 0
    assert result['best_error'] >= result['lower_bound']
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.)
    original = module.occupations
    def interrupted(*args):
        yield next(original(*args))
        raise KeyboardInterrupt
    monkeypatch.setattr(module, 'occupations', interrupted)
    result, _ = module.enumerate_sro(ring(), cutoff=1.1, targets=[target(0)])
    assert result['evaluated_configurations'] == 1
    assert result['status'] == 'best_found' and result['stop_reason'] == 'interrupted'


def test_cli_outputs_roundtrip_and_public_api(tmp_path):
    path = tmp_path / 'cell.extxyz'
    ring().write(path)
    targets = tmp_path / 'targets.json'
    targets.write_text(json.dumps([target(.25)]))
    output = tmp_path / 'exact'
    args = [str(path), '--targets', str(targets), '--cutoff', '1.1', '--output', str(output)]
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT.parent / 'python'))
    command = [sys.executable, '-c', 'from m_scraps.cli import main; main()', 'enumerate', *args]
    subprocess.run(command, check=True, env=env)
    result = json.loads((output / 'result.json').read_text())
    assert result['best_error_exact'] == '1/4'
    assert result['total_configurations'] == 6
    for name in ['best.cif', 'best.POSCAR', 'best.extxyz']:
        witness = read(output / name)
        actual = analyze_structure(witness, cutoff=1.1, sro_targets=[target(.25)])
        assert actual['sro_comparison']['metrics']['max_error'] == .25
    fail = subprocess.run(command, env=env, capture_output=True, text=True)
    assert fail.returncode != 0 and 'already exists' in fail.stderr
    sys.path.insert(0, str(ROOT.parent / 'python'))
    from m_scraps import enumerate_sro
    assert enumerate_sro(ring(), cutoff=1.1, targets=[target(.25)])[0]['best_error'] == .25
