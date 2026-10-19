"""Independent geometry, finite-search and witness checks for cell discovery."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_cells import cell_geometry, hnf_matrices, row_hnf, search_cells
from sro_bounds import sro_bounds


def target(source='all', destination='all', a='Cu', b='Zn', value=0):
    return dict(source=source, destination=destination, shell=1,
                center_species=a, neighbor_species=b, target=value)


def run(**kwargs):
    args = dict(cutoff=1.01, targets=[target()], min_atoms=2, max_atoms=4,
                composition={'all': {'Cu': '1/2', 'Zn': '1/2'}},
                max_candidates=20, max_seconds=30, seconds_per_cell=10)
    args.update(kwargs)
    return search_cells(Atoms('Cu', cell=np.eye(3), positions=[[0, 0, 0]], pbc=True), **args)


def test_hnf_counts_and_unimodular_basis_invariance():
    # Known sublattice counts sigma_1,2(p)=1+p+p^2 for prime p.
    assert len(list(hnf_matrices(2))) == 7
    assert len(list(hnf_matrices(3))) == 13
    assert len(list(hnf_matrices(4))) == 35
    transforms = [np.array([[1, 7, 0], [0, 1, -3], [0, 0, 1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])]
    for matrix in hnf_matrices(4):
        assert row_hnf(matrix) == tuple(matrix.flat)
        for transform in transforms:
            assert row_hnf(transform @ matrix) == tuple(matrix.flat)


def test_basis_skew_removed_but_thin_lattice_retained():
    compact = cell_geometry([[3, 0, 0], [30, 3, 0], [-9, 18, 3]])
    assert compact['aspect_ratio'] == pytest.approx(1)
    assert compact['shortest_repeat'] == pytest.approx(3)
    thin = cell_geometry(np.diag([1, 5, 5]))
    assert thin['aspect_ratio'] == 5
    assert thin['shortest_repeat'] == 1
    assert thin['volume'] == pytest.approx(25)


def test_cubic_equivalence_and_geometry_constraints():
    report, _ = run(min_atoms=2, max_atoms=2)
    assert report['generation']['hnfs_examined'] == 7
    assert report['generation']['unique_lattices'] == 3
    assert report['generation']['parent_rotations'] == 48
    assert len(report['candidates']) == 3
    filtered, _ = run(min_repeat=1.4, max_aspect=1.5)
    assert filtered['candidates']
    for c in filtered['candidates']:
        assert c['geometry']['shortest_repeat'] >= 1.4 - 1e-10
        assert c['geometry']['aspect_ratio'] <= 1.5 + 1e-10
    empty, _ = run(min_repeat=10)
    assert empty['candidates'] == empty['pareto_ids'] == []


@pytest.mark.parametrize('metric', ['max_error', 'rms_error', 'mean_absolute_error'])
def test_exportable_witnesses_and_exact_optima(metric):
    report, witnesses = run(metric=metric)
    assert not report['global_optimum_certified']
    assert report['pareto_ids']
    for c in report['candidates']:
        assert c['status'] == 'certified_minimum'
        witness = witnesses[c['id']]
        independent = sro_bounds(witness, cutoff=1.01, targets=[target()],
                                 sublattices=c['sublattices'], metric=metric)
        assert independent['result']['achieved_upper_bound'] == pytest.approx(c['best_error'])
        assert c['best_error'] == c['lower_bound']
        assert witness.get_chemical_symbols().count('Cu') == len(witness) // 2
        assert np.linalg.det(witness.cell) == pytest.approx(len(witness))


def test_multisublattice_shared_species_and_fixed_spectator():
    parent = Atoms('CuCuC', scaled_positions=[[0, 0, 0], [.5, .5, .5], [.25, .25, .25]],
                   cell=np.eye(3) * 3, pbc=True)
    config = dict(site_labels=['A', 'B', 'fixed'], allowed={'A': ['Cu', 'Zn'], 'B': ['Cu', 'Zn'], 'fixed': ['C']})
    composition = {'A': {'Cu': '.5', 'Zn': '.5'}, 'B': {'Cu': '.5', 'Zn': '.5'}, 'fixed': {'C': 1}}
    report, witnesses = search_cells(parent, cutoff=2.7, targets=[target('A', 'B')],
        min_atoms=3, max_atoms=6, sublattices=config, composition=composition, max_candidates=3,
        max_seconds=30, seconds_per_cell=10)
    assert report['generation']['composition_incompatible_sizes'] == [3]
    for c in report['candidates']:
        symbols = np.array(witnesses[c['id']].get_chemical_symbols())
        labels = np.array(c['sublattices']['site_labels'])
        assert list(symbols[labels == 'fixed']) == ['C', 'C']
        for g in ['A', 'B']:
            assert sorted(symbols[labels == g]) == ['Cu', 'Zn']
        check = sro_bounds(witnesses[c['id']], cutoff=2.7, targets=[target('A', 'B')], sublattices=c['sublattices'])
        assert check['result']['achieved_upper_bound'] == pytest.approx(c['best_error'])
    assert witnesses


def test_sa_determinism_and_same_metric():
    targets = [target(value=0), target(a='Zn', b='Cu', value=-1)]
    one, witnesses = run(max_configurations=1, sa_steps=80, targets=targets)
    two, _ = run(max_configurations=1, sa_steps=80, targets=targets)
    assert any(c['optimization']['method'] == 'python_pair_sro_sa' for c in one['candidates'])
    assert [c['best_error'] for c in one['candidates']] == [c['best_error'] for c in two['candidates']]
    for c in one['candidates']:
        assert c['optimization']['method'] in ('python_pair_sro_sa', 'initial_bound_witness')
        check = sro_bounds(witnesses[c['id']], cutoff=1.01, targets=targets, sublattices=c['sublattices'])
        assert c['best_error'] == check['result']['achieved_upper_bound']


def test_caps_and_fraction_validation():
    report, _ = run(max_hnfs=1, max_candidates=1)
    assert report['generation']['stop_reason'] == 'hnf_cap'
    assert report['generation']['hnfs_examined'] == 1
    report, _ = run(max_candidates=1)
    assert report['candidate_limit_reached']
    assert len(report['candidates']) == 1
    for bad in [dict(max_aspect=.5), dict(min_repeat=-1), dict(max_candidates=0),
                dict(max_seconds=0), dict(error_tolerance=-1), dict(seed=-1),
                dict(composition={'all': {'Cu': '1/0', 'Zn': 0}}),
                dict(composition={'all': {'Cu': '.3', 'Zn': '.3'}})]:
        with pytest.raises(ValueError):
            run(**bad)
    timed, _ = run(max_seconds=1e-12)
    assert timed['time_limit_reached']
    assert not timed['global_optimum_certified']


def test_bound_screening():
    report, witnesses = run(targets=[target(value=.123)], error_tolerance=.00001)
    assert report['candidates']
    assert not witnesses
    assert all(c['status'] == 'screened_by_bound' for c in report['candidates'])


def test_cli_roundtrip(tmp_path):
    parent = tmp_path / 'parent.extxyz'
    write(parent, bulk('Cu', 'sc', a=1))
    targets = tmp_path / 'targets.json'
    targets.write_text(json.dumps([target()]))
    composition = tmp_path / 'composition.json'
    composition.write_text(json.dumps({'all': {'Cu': '.5', 'Zn': '.5'}}))
    output = tmp_path / 'results'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'search_cells.py'), str(parent),
        '--cutoff', '1.01', '--targets', str(targets), '--composition', str(composition),
        '--min-atoms', '2', '--max-atoms', '2', '--output', str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads((output / 'result.json').read_text())
    for candidate in report['candidates']:
        folder = output / candidate['id']
        witness = read(folder / 'best.extxyz')
        config = json.loads((folder / 'sublattices.json').read_text())
        check = sro_bounds(witness, cutoff=1.01, targets=[target()], sublattices=config)
        assert check['result']['achieved_upper_bound'] == pytest.approx(candidate['best_error'])
        assert len(candidate['files']) == 4
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_stronger_bound_screens_small_cells_without_optimization(monkeypatch):
    import search_cells as module
    def forbidden(*args, **kwargs):
        raise AssertionError('A bound-screened cell must not be optimized')
    monkeypatch.setattr(module, 'enumerate_sro', forbidden)
    monkeypatch.setattr(module, '_anneal', forbidden)
    parent = bulk('Cu', 'bcc', a=3, cubic=True)
    report, witnesses = module.search_cells(parent, cutoff=2.7, targets=[target(value=-.1)],
        min_atoms=16, max_atoms=16, composition={'all': {'Cu': '.5', 'Zn': '.5'}},
        error_tolerance=.01, max_candidates=8)
    assert report['candidates'] and not witnesses
    assert all(c['status'] == 'screened_by_bound' and c['lower_bound'] >= .025 for c in report['candidates'])


def test_sa_progress_preserves_search_and_measures_improvements():
    from search_cells import _anneal
    import time
    parent = bulk('Cu', 'bcc', a=2, cubic=True).repeat((2, 1, 1))
    parent.symbols[2:] = 'Zn'
    config = {'site_labels': ['all'] * 4, 'allowed': {'all': ['Cu', 'Zn']}}
    events = []
    observed, witness = _anneal(parent, config, [target()], 1.8, 1, 'max_error', 100, 42,
                                time.monotonic()+10, progress=events.append)
    reference, other = _anneal(parent, config, [target()], 1.8, 1, 'max_error', 100, 42, time.monotonic()+10)
    assert observed['best_error'] == reference['best_error']
    assert witness.get_chemical_symbols() == other.get_chemical_symbols()
    assert events[0]['steps'] == 0
    assert all(a['best_error'] > b['best_error'] for a, b in zip(events, events[1:]))
    assert all(a['elapsed_seconds'] <= b['elapsed_seconds'] for a, b in zip(events, events[1:]))
    assert np.isclose(events[-1]['best_error'], observed['best_error'])
