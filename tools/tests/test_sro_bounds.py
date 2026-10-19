"""Analytic and exhaustive small-cell checks of the SRO relaxation."""
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import pytest
from ase import Atoms
from ase.build import bulk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sro_bounds import pair_count_bound, sro_bounds
from structure_analysis import analyze_structure


def target(value, *, source='all', destination='all', a='Cu', b='Zn', shell=1):
    return dict(source=source, destination=destination, center_species=a,
                neighbor_species=b, shell=shell, target=value)


def test_analytic_32_site_bcc_bound():
    atoms = bulk('Cu', 'bcc', a=2, cubic=True).repeat((2, 2, 4))
    atoms.symbols[16:] = 'Zn'
    result = sro_bounds(atoms, cutoff=1.8, shells=1, targets=[target(-.1)])
    row = result['entries'][0]
    assert row['denominator_exact'] == '64'
    assert row['target_pair_count_exact'] == '352/5'
    assert row['nearest_relaxed_count'] == 70
    assert row['component_lower_bound_exact'] == '1/160'
    assert result['result']['lower_bound'] == .00625


def test_capacity_and_parity_bounds():
    # A+B>E forces at least six bonds, rather than rounding to zero.
    row = pair_count_bound(8, 8, 10, Fraction(32, 5), Fraction(1))
    assert row[:3] == (6, 8, 1)
    assert row[-1] == Fraction(15, 16)
    even = pair_count_bound(6, 6, 12, Fraction(3), Fraction(0), same_species=True)
    assert even[2] == 2
    assert even[-1] == Fraction(1, 3)


def test_periodic_multiplicity_closes_integer_relaxation_gap():
    # Periodic images all connect the same two occupation variables.
    atoms = bulk('Cu', 'bcc', a=2, cubic=True)
    atoms[1].symbol = 'Zn'
    result = sro_bounds(atoms, cutoff=1.8, targets=[target(0)])
    assert result['result']['lower_bound'] == 1
    assert result['result']['achieved_upper_bound'] == 1
    assert result['result']['status'] == 'certified_minimum'
    # A valid supplied witness can certify a nonzero optimum too.
    fixed = {'site_labels': ['A', 'B'], 'allowed': {'A': ['Cu'], 'B': ['Zn']}}
    result = sro_bounds(atoms, cutoff=1.8, sublattices=fixed,
                        targets=[target(.2, source='A', destination='B')])
    assert result['result']['lower_bound'] == .2
    assert result['result']['status'] == 'certified_minimum'


@pytest.mark.parametrize('multisublattice', [False, True])
def test_bounds_do_not_exceed_exhaustive_lattice_minima(multisublattice):
    atoms = Atoms('Cu2Zn2', positions=[[i, 0, 0] for i in range(4)],
                  cell=[4, 8, 8], pbc=True)
    config = None
    occupations = []
    if multisublattice:
        config = {'site_labels': ['A', 'B', 'A', 'B'],
                  'allowed': {'A': ['Cu', 'Zn'], 'B': ['Cu', 'Zn']},
                  'counts': {'A': {'Cu': 1, 'Zn': 1}, 'B': {'Cu': 1, 'Zn': 1}}}
        occupations = [(a, b) for a in (0, 2) for b in (1, 3)]
        targets = [target(.3, source='A', destination='B'),
                   target(-.4, source='B', destination='A', a='Zn', b='Cu')]
    else:
        occupations = list(combinations(range(4), 2))
        targets = [target(.5), target(-.2, a='Cu', b='Cu'), target(.1, a='Zn', b='Cu')]
    minima = {k: float('inf') for k in ('max_error', 'rms_error', 'mean_absolute_error')}
    results = []
    for cu in occupations:
        atoms.set_chemical_symbols(['Cu' if i in cu else 'Zn' for i in range(4)])
        actual = analyze_structure(atoms, cutoff=1.1, sublattices=config,
                                   sro_targets=targets)['sro_comparison']['metrics']
        for k in minima:
            minima[k] = min(minima[k], actual[k])
        results.append(sro_bounds(atoms, cutoff=1.1, targets=targets, sublattices=config))
    for result in results:
        for k in minima:
            assert result['metrics'][k]['lower_bound'] <= minima[k] + 1e-14
            assert result['metrics'][k]['achieved_upper_bound'] >= minima[k] - 1e-14
    # The lower bound depends on geometry/composition, not the starting decoration.
    for k in minima:
        assert len({r['metrics'][k]['lower_bound_exact'] for r in results}) == 1


def test_nonuniform_geometry_and_invalid_inputs():
    atoms = Atoms('CuZnCu', positions=[[0, 0, 0], [1, 0, 0], [2.05, 0, 0]],
                  cell=[8, 8, 8], pbc=True)
    with pytest.raises(ValueError, match='Nonuniform coordination'):
        sro_bounds(atoms, cutoff=1.1, targets=[target(0)])
    with pytest.raises(ValueError, match='nonempty'):
        sro_bounds(atoms, cutoff=1.1, targets=[])
    with pytest.raises(ValueError, match='finite'):
        sro_bounds(atoms, cutoff=1.1, targets=[target(float('nan'))])
    with pytest.raises(ValueError, match='Duplicate'):
        sro_bounds(atoms, cutoff=1.1, targets=[target(0), target(0)])


def test_cli_and_public_api(tmp_path):
    atoms = bulk('Cu', 'bcc', a=2, cubic=True)
    atoms[1].symbol = 'Zn'
    path = tmp_path / 'cell.extxyz'
    atoms.write(path)
    targets = tmp_path / 'targets.json'
    targets.write_text(json.dumps([target(0)]))
    output = tmp_path / 'bounds.json'
    command = [str(path), '--cutoff', '1.8', '--targets', str(targets),
               '--metric', 'rms_error', '--output', str(output)]
    subprocess.run([sys.executable, str(ROOT / 'sro_bounds.py'), *command], check=True)
    payload = json.loads(output.read_text())
    assert payload['selected_metric'] == 'rms_error'
    assert payload['result']['status'] == 'certified_minimum'
    assert set(payload['inputs']) == {'structure', 'targets'}
    sys.path.insert(0, str(ROOT.parent / 'python'))
    from m_scraps import sro_bounds as api
    assert api(atoms, cutoff=1.8, targets=[target(0)])['result']['lower_bound'] == 1
    # Exercise installed-command dispatch using the source package without building MPI.
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT.parent / 'python'))
    subprocess.run([sys.executable, '-c', 'from m_scraps.cli import main; main()',
                    'bounds', *command], env=env, check=True)


def test_unequal_coordination_and_reference_mapping():
    ref = Atoms('TiCuZn', positions=[[i, 0, 0] for i in range(3)],
                cell=[3, 8, 8], pbc=True)
    config = {'site_labels': ['A', 'B', 'B'],
              'allowed': {'A': ['Ti'], 'B': ['Cu', 'Zn']}}
    occupied = ref[[2, 0, 1]]
    occupied.positions += [.01, 0, 0]
    result = sro_bounds(occupied, reference=ref, cutoff=1.1, sublattices=config,
                       targets=[target(.1, source='A', destination='B', a='Ti', b='Zn')])
    row = result['entries'][0]
    assert row['coordination'] == 2
    assert row['reverse_coordination'] == 1
    assert row['admissible_count_min'] == row['admissible_count_max'] == 1
    assert result['result']['status'] == 'certified_minimum'
    assert result['result']['lower_bound'] == .1


def test_binary_cut_parity_strengthens_16_site_bound():
    atoms = bulk('Cu', 'bcc', a=2, cubic=True).repeat((2, 2, 2))
    atoms.symbols[8:] = 'Zn'
    result = sro_bounds(atoms, cutoff=1.8, targets=[target(-.1)])
    row = result['entries'][0]
    assert row['integer_only_lower_bound_exact'] == '1/160'
    assert row['component_lower_bound_exact'] == '1/40'
    assert row['count_step'] == 2


def test_graph_congruences_against_every_small_graph_decoration():
    from itertools import permutations
    import numpy as np
    from sro_bounds import graph_count_bound
    graphs = [np.array([[4, 2, 4], [2, 2, 6], [4, 6, 0]]),
              np.array([[2, 2, 0, 2], [2, 2, 2, 0], [0, 2, 2, 2], [2, 0, 2, 2]]),
              np.eye(4, dtype=int) * 6]
    for graph in graphs:
        n = len(graph)
        symbols = ['Cu', 'Zn', 'Al'] if n == 3 else ['Cu', 'Cu', 'Zn', 'Zn']
        counts = {s: symbols.count(s) for s in set(symbols)}
        composition = {'all': {'sites': n, 'counts': counts}}
        ii, jj = [], []
        for i in range(n):
            for j in range(n):
                ii.extend([i] * graph[i, j]); jj.extend([j] * graph[i, j])
        for a, b in [('Cu', 'Cu'), ('Cu', 'Zn')]:
            actual = [sum(graph[i, j] for i in range(n) for j in range(n) if values[i] == a and values[j] == b)
                      for values in set(permutations(symbols))]
            z = int(graph.sum(axis=1)[0])
            denominator = Fraction(counts[a] * z * counts[b], n)
            for value in [Fraction(-1, 10), Fraction(0), Fraction(1, 3)]:
                lo, hi, _, _, _, _ = pair_count_bound(counts[a]*z, counts[b]*z, int(graph.sum()), denominator, value, same_species=a == b)
                low, high, step, offset, nearest, error = graph_count_bound(np.array(ii), np.array(jj), n, composition,
                    'all', 'all', a, b, lo, hi, denominator, value)
                assert error <= min(abs(1-Fraction(v)/denominator-value) for v in actual)
                for v in actual:
                    assert low <= v <= high
                    assert (v-offset) % step == 0 if step else v == offset


def test_joint_reciprocity_conflict_and_safe_cap():
    atoms = Atoms('Cu2Zn2', positions=[[i, 0, 0] for i in range(4)], cell=[4, 8, 8], pbc=True)
    targets = [target(0), target(-1, a='Zn', b='Cu')]
    result = sro_bounds(atoms, cutoff=1.1, targets=targets)
    assert result['independent_lower_bounds_exact']['max_error'] == '0'
    assert result['result']['lower_bound_exact'] == '1'
    assert result['metrics']['mean_absolute_error']['lower_bound_exact'] == '1/2'
    assert result['metrics']['rms_error']['lower_bound_exact'] == '1/2'
    assert result['joint']['blocks'][0]['strengthened']
    capped = sro_bounds(atoms, cutoff=1.1, targets=targets, max_joint_states=1)
    assert capped['result']['lower_bound_exact'] == '0'
    assert capped['joint']['blocks'][0]['status'] == 'state_cap'
    independent = sro_bounds(atoms, cutoff=1.1, targets=targets, joint=False)
    assert independent['result']['lower_bound_exact'] == '0'


def test_joint_ternary_neighbor_totals_against_all_occupations():
    from itertools import permutations
    atoms = Atoms('Cu2Zn2Al2', positions=[[i, 0, 0] for i in range(6)], cell=[6, 8, 8], pbc=True)
    targets = [target(1, a='Cu', b=b) for b in ('Cu', 'Zn', 'Al')]
    report = sro_bounds(atoms, cutoff=1.1, targets=targets)
    assert all(v == '0' for v in report['independent_lower_bounds_exact'].values())
    assert report['result']['lower_bound'] > 0
    minima = {k: float('inf') for k in report['metrics']}
    for symbols in set(permutations('Cu Cu Zn Zn Al Al'.split())):
        atoms.set_chemical_symbols(symbols)
        actual = analyze_structure(atoms, cutoff=1.1, sro_targets=targets)['sro_comparison']['metrics']
        for k in minima:
            minima[k] = min(minima[k], actual[k])
    for k in minima:
        assert report['metrics'][k]['lower_bound'] <= minima[k] + 1e-14


def test_joint_multisublattice_reversed_graph_conflict():
    atoms = Atoms('CuZnCuZn', positions=[[i, 0, 0] for i in range(4)], cell=[4, 8, 8], pbc=True)
    config = dict(site_labels=['A', 'A', 'B', 'B'], allowed={'A': ['Cu', 'Zn'], 'B': ['Cu', 'Zn']})
    targets = [target(-1, source='A', destination='B'),
               target(1, source='B', destination='A', a='Zn', b='Cu')]
    report = sro_bounds(atoms, cutoff=1.1, targets=targets, sublattices=config)
    assert report['independent_lower_bounds_exact']['max_error'] == '0'
    assert report['result']['lower_bound_exact'] == '2'
    assert report['joint']['blocks'][0]['target_indices'] == [0, 1]


def test_joint_unequal_coordination_and_compositions():
    atoms = Atoms('Cu6', positions=[[i, 0, 0] for i in range(6)], cell=[6, 8, 8], pbc=True)
    config = dict(site_labels=['A', 'B', 'B', 'A', 'B', 'B'],
                  allowed={'A': ['Cu', 'Zn'], 'B': ['Cu', 'Zn']},
                  counts={'A': {'Cu': 1, 'Zn': 1}, 'B': {'Cu': 3, 'Zn': 1}})
    targets = [target(.4, source='A', destination='B'),
               target(-.2, source='B', destination='A', a='Zn', b='Cu'),
               target(.1, source='A', destination='B', b='Cu')]
    minima = {k: float('inf') for k in ('max_error', 'rms_error', 'mean_absolute_error')}
    bounds = []
    for a in (0, 3):
        for b in (1, 2, 4, 5):
            atoms.set_chemical_symbols(['Zn' if i in (a, b) else 'Cu' for i in range(6)])
            result = sro_bounds(atoms, cutoff=1.1, sublattices=config, targets=targets)
            assert len(result['joint']['blocks']) == 1
            assert result['entries'][0]['coordination'] == 2
            assert result['entries'][0]['reverse_coordination'] == 1
            for k in minima:
                minima[k] = min(minima[k], result['metrics'][k]['achieved_upper_bound'])
            bounds.append(result)
    for result in bounds:
        for k in minima:
            assert result['metrics'][k]['lower_bound'] <= minima[k] + 1e-14
