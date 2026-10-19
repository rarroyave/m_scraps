"""Configuration diversity, exact acceptance, and real periodic-cell regressions."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sro_ensemble import generate_ensemble
from sro_stopping import SROBoundStop
from structure_analysis import analyze_structure
from enumeration_symmetry import site_permutations


def ring():
    return Atoms('Cu4Zn4', positions=[[i, 0, 0] for i in range(8)], cell=[8, 10, 10], pbc=True)


def targets(value=0):
    return [dict(source='all', destination='all', shell=1,
                 center_species='Cu', neighbor_species='Zn', target=value)]


def run(atoms=None, **kwargs):
    options = dict(cutoff=2.1, shells=2, targets=targets(), error_tolerance=0,
                   count=8, max_seconds=20)
    options.update(kwargs)
    return generate_ensemble(ring() if atoms is None else atoms, **options)


def test_exact_pool_matches_exhaustive_orbits_and_reports_variability():
    atoms = ring()
    report, structures = run()
    checker = SROBoundStop(atoms, cutoff=2.1, shells=2, targets=targets())
    permutations = site_permutations(atoms, checker.labels, checker.prepared, 1e-5)
    eligible = set()
    for cu in itertools.combinations(range(8), 4):
        candidate = atoms.copy()
        symbols = np.array(['Cu' if i in cu else 'Zn' for i in range(8)])
        candidate.set_chemical_symbols(symbols)
        if checker(candidate):
            eligible.add(min(tuple(symbols[p]) for p in permutations))
    assert {tuple(a.get_chemical_symbols()) for a in structures} == eligible
    assert len(structures) == 4  # Four distinct first-shell-perfect configurations.
    assert report['search']['complete'] and report['search']['covered_configurations'] == 70
    assert report['status'] == 'insufficient_candidates'
    vectors = np.array([r['diversity_vector'] for r in report['selected']])
    np.testing.assert_allclose(report['diversity']['spread']['population_std'], vectors.std(axis=0))
    assert np.max(vectors.std(axis=0)) > 0  # Untargeted second-shell SRO varies.
    assert all(checker(a) for a in structures)
    assert all(r['metrics_exact']['max_error'] == '0' for r in report['selected'])
    # Equal pair vectors can still belong to different symmetry orbits.
    assert len(np.unique(vectors, axis=0)) < len(structures)


def test_greedy_selection_anchor_and_distance():
    full, _ = run()
    report, _ = run(count=2)
    vectors = np.array([r['diversity_vector'] for r in full['selected']])
    chosen = np.array([r['diversity_vector'] for r in report['selected']])
    assert report['status'] == 'requested_count_reached'
    assert np.mean((chosen[0] - chosen[1]) ** 2) == pytest.approx(
        max(np.mean((v - chosen[0]) ** 2) for v in vectors))


@pytest.mark.parametrize('metric', ['max_error', 'rms_error', 'mean_absolute_error'])
def test_acceptance_uses_exact_tolerance_and_screens_impossible_targets(metric):
    report, structures = run(targets=targets(.1), error_tolerance=.09999999999999999, metric=metric)
    assert report['status'] == 'infeasible_by_bound' and not structures
    report, structures = run(targets=targets(.1), error_tolerance=.1, metric=metric)
    assert structures
    assert all(r['metrics'][metric] <= .1 for r in report['selected'])


def test_seeded_sa_reproducible_and_continues_after_first_optimum():
    a, structures = run(method='sa', starts=3, steps=100, seed=42, count=4)
    b, again = run(method='sa', starts=3, steps=100, seed=42, count=4)
    assert len(structures) > 1
    assert [r['occupations_sha256'] for r in a['selected']] == [r['occupations_sha256'] for r in b['selected']]
    assert a['stop_reason'] == b['stop_reason'] == 'starts_exhausted'
    assert not a['search']['complete']
    assert all(analyze_structure(s, cutoff=2.1, shells=2, sro_targets=targets())['sro_comparison']['metrics']['max_error'] == 0 for s in structures)


@pytest.mark.parametrize('selection', ['diversity', 'random'])
def test_multisublattice_rocksalt_counts_and_symmetry(selection):
    atoms = bulk('TiC', 'rocksalt', a=4.4, cubic=True)
    labels = ['M' if a.symbol == 'Ti' else 'X' for a in atoms]
    metal, nonmetal = [i for i, g in enumerate(labels) if g == 'M'], [i for i, g in enumerate(labels) if g == 'X']
    for i in metal[2:]: atoms[i].symbol = 'Zr'
    for i in nonmetal[2:]: atoms[i].symbol = 'N'
    config = dict(site_labels=labels, allowed={'M': ['Ti', 'Zr'], 'X': ['C', 'N']})
    target = [dict(source='M', destination='X', shell=1, center_species='Ti', neighbor_species='N', target=0)]
    report, structures = generate_ensemble(atoms, cutoff=2.3, shells=1, targets=target,
        sublattices=config, error_tolerance=.34, count=8, selection=selection)
    assert len(structures) > 1 and report['search']['complete']
    for candidate in structures:
        analysis = analyze_structure(candidate, cutoff=2.3, shells=1, sublattices=config, sro_targets=target)
        assert analysis['composition']['M']['counts'] == {'Ti': 2, 'Zr': 2}
        assert analysis['composition']['X']['counts'] == {'C': 2, 'N': 2}
        assert analysis['sro_comparison']['metrics']['max_error'] <= .34


def test_pool_cap_and_time_cap_are_explicit():
    report, structures = run(count=2, max_pool=2)
    assert report['stop_reason'] == 'pool_limit' and len(structures) == 2
    assert not report['search']['complete']
    report, structures = run(max_seconds=1e-12)
    assert report['stop_reason'] == 'time_limit' and not structures
    with pytest.raises(ValueError, match='max_pool'): run(count=9, max_pool=8)
    with pytest.raises(ValueError, match='Occupation space'): run(method='enumerate', max_configurations=1)


def write_ring_figures(path):
    atoms = ring()
    lines = ['SCRAPS_FIGURES 1 8 2 1 2']
    lines += [' '.join(map(str, row)) for row in atoms.cell.array]
    lines += [' '.join(map(str, row)) for row in atoms.get_scaled_positions()]
    lines += ['-1 1']
    for offsets in [(0, 2), (0, 1, 2)]:
        lines += [f'{len(offsets)} 2 0 8']
        for i in range(8):
            lines += ['0.125 ' + ' '.join(f'{(i+j)%8} 0' for j in offsets)]
    path.write_text('\n'.join(lines) + '\n')


@pytest.mark.parametrize('selection', ['diversity', 'random'])
def test_pair_triplet_diversity_against_direct_spin_products(selection, tmp_path):
    path = tmp_path / 'FIGURES'
    write_ring_figures(path)
    report, structures = run(figures=path, species=['Cu', 'Zn'], selection=selection)
    assert report['diversity']['kind'] == 'decorated_figure_correlations'
    assert [r['order'] for r in report['diversity']['columns']] == [2, 3]
    vectors = np.array([entry['diversity_vector'] for entry in report['selected']])
    assert len(np.unique(vectors, axis=0)) == 4
    assert report['diversity']['spread']['population_std'][1] > 0
    for entry, candidate in zip(report['selected'], structures):
        spins = np.array([-1 if s == 'Cu' else 1 for s in candidate.get_chemical_symbols()])
        direct = [np.mean(spins * np.roll(spins, -2)),
                  np.mean(spins * np.roll(spins, -1) * np.roll(spins, -2))]
        np.testing.assert_allclose(entry['diversity_vector'], direct, atol=1e-15)
    with pytest.raises(ValueError, match='alphabet'): run(figures=path)
    shifted = ring(); shifted.positions += [.1, 0, 0]
    with pytest.raises(ValueError, match='site order'): run(shifted, figures=path, species=['Cu', 'Zn'])


@pytest.mark.parametrize('selection', ['diversity', 'random'])
def test_cli_exports_and_public_api(selection, tmp_path):
    structure, target_path = tmp_path/'input.extxyz', tmp_path/'targets.json'
    write(structure, ring()); target_path.write_text(json.dumps(targets()))
    output = tmp_path/'ensemble'
    subprocess.run([sys.executable, '-c',
        'import sys; sys.path.insert(0, sys.argv.pop(1)); from m_scraps.cli import main; sys.argv[0]="m_scraps"; main()',
        str(ROOT.parent/'python'), 'ensemble', str(structure), '--targets', str(target_path),
        '--cutoff', '2.1', '--shells', '2', '--error-tolerance', '0', '--count', '3', '--selection', selection, '--selection-seed', '17', '--output', str(output)], check=True)
    report = json.loads((output/'result.json').read_text())
    assert report['selected_count'] == 3
    assert report['selection']['method'] == selection
    assert report['selection']['seed'] == 17
    for entry in report['selected']:
        for name in ['structure.extxyz', 'structure.cif', 'POSCAR']:
            atoms = read(output/entry['id']/name)
            assert atoms.get_chemical_symbols() == entry['symbols']
    sys.path.insert(0, str(ROOT.parent/'python'))
    from m_scraps import generate_ensemble as public
    assert public is generate_ensemble


def test_frozen_48_site_ensembles_independently(tmp_path):
    import hashlib
    import io
    fixtures = json.loads((ROOT/'fixtures/ensemble_48.json').read_text())
    for case in fixtures['cases']:
        report = case['report']
        vectors, seen = [], set()
        assert report['selected_count'] == 4
        for entry, saved in zip(report['selected'], case['witnesses']):
            assert hashlib.sha256(saved['extxyz'].encode()).hexdigest() == saved['sha256']
            atoms = read(io.StringIO(saved['extxyz']), format='extxyz')
            checker = SROBoundStop(atoms, cutoff=3.2, shells=2, targets=report['targets'],
                                   sublattices=report['sublattices'])
            permutations = site_permutations(atoms, checker.labels, checker.prepared, 1e-5)
            symbols = np.array(atoms.get_chemical_symbols())
            canonical = min(tuple(symbols[p]) for p in permutations)
            assert canonical not in seen
            seen.add(canonical)
            analysis = analyze_structure(atoms, cutoff=3.2, shells=2, sublattices=report['sublattices'],
                                         sro_targets=report['targets'])
            assert analysis['sro_comparison']['metrics']['max_error'] <= .02
            assert analysis['sro_comparison']['metrics'] == pytest.approx(entry['metrics'], abs=1e-14)
            vector = [row['alpha'] for row in analysis['sro'] if row['alpha'] is not None]
            np.testing.assert_allclose(vector, entry['diversity_vector'], atol=1e-14)
            vectors.append(vector)
        np.testing.assert_allclose(np.std(vectors, axis=0), report['diversity']['spread']['population_std'], atol=1e-14)
        assert np.max(np.std(vectors, axis=0)) > 0


@pytest.mark.parametrize('method', ['enumerate', 'sa'])
def test_random_selection_reproducible_without_changing_candidate_pool(method):
    options = dict(method=method, starts=3, steps=100, count=2, seed=42)
    diverse, _ = run(**options)
    random, structures = run(**options, selection='random', selection_seed=17)
    repeated, _ = run(**options, selection='random', selection_seed=17)
    pool = random['selection']['pool_occupations_sha256']
    assert len(pool) > 2
    assert pool == diverse['selection']['pool_occupations_sha256']
    assert random['selected'] == repeated['selected']
    assert len({r['occupations_sha256'] for r in random['selected']}) == 2
    assert all(SROBoundStop(ring(), cutoff=2.1, shells=2, targets=targets())(a) for a in structures)
    # Fixed selection seeds exercise every member, without an obligatory anchor.
    subsets = []
    for seed in range(12):
        report, _ = run(**options, selection='random', selection_seed=seed)
        assert report['selection']['pool_occupations_sha256'] == pool
        subsets.append({r['pool_index'] for r in report['selected']})
    assert set.union(*subsets) == set(range(len(pool)))
    assert not set.intersection(*subsets)


def test_random_empty_and_short_pools_and_validation():
    report, structures = run(selection='random', count=8)
    assert len(structures) == 4
    assert report['status'] == 'insufficient_candidates'
    report, structures = run(selection='random', targets=targets(.1), error_tolerance=0)
    assert not structures and report['selection']['pool_occupations_sha256'] == []
    report, _ = run(selection='random', seed=7)
    assert report['selection']['seed'] == 7
    with pytest.raises(ValueError, match='selection must'):
        run(selection='uniform')
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match='selection_seed'):
            run(selection_seed=bad)
