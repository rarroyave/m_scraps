"""Symmetry reduction must preserve the exhaustive answer and cover every orbit."""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from enumeration_symmetry import permutation_group, site_permutations, symmetry_occupations
from enumerate_sro import enumerate_sro, occupations


def target(value=0, source='all', destination='all', a='Cu', b='Zn'):
    return dict(source=source, destination=destination, shell=1,
                center_species=a, neighbor_species=b, target=value)


def ring():
    return Atoms('Cu2Zn2', positions=[[i, 0, 0] for i in range(4)], cell=[4, 8, 8], pbc=True)


def test_ring_six_decorations_two_representatives():
    reduced, _ = enumerate_sro(ring(), stop_at_bound=False, cutoff=1.1, targets=[target(.25)])
    exhaustive, _ = enumerate_sro(ring(), stop_at_bound=False, cutoff=1.1, targets=[target(.25)], symmetry='none')
    assert reduced['evaluated_configurations'] == 2
    assert exhaustive['evaluated_configurations'] == 6
    assert reduced['covered_configurations'] == 6
    assert reduced['best_error_exact'] == exhaustive['best_error_exact'] == '1/4'
    assert reduced['complete'] and reduced['symmetry']['site_permutations'] == 8


@pytest.mark.parametrize('counts', [{'Cu': 2, 'Zn': 2}, {'Cu': 2, 'Zn': 1, 'Al': 1}, {'Cu': 4}])
def test_subset_stabilizers_cover_exact_orbits(counts):
    group = permutation_group([(1, 2, 3, 0), (0, 3, 2, 1)], 4)
    labels = np.array(['all'] * 4)
    composition = {'all': dict(sites=4, counts=counts)}
    exhaustive = list(occupations(labels, composition))
    canonical = lambda c: min(tuple(c[i] for i in p) for p in group)
    expected = {canonical(c) for c in exhaustive}
    representatives = list(symmetry_occupations(labels, composition, group))
    assert {canonical(c) for c, _ in representatives} == expected
    assert len(representatives) == len(expected)
    assert sum(m for _, m in representatives) == len(exhaustive)
    for c, multiplicity in representatives:
        assert multiplicity == len({tuple(c[i] for i in p) for p in group})


@pytest.mark.parametrize('metric', ['max_error', 'rms_error', 'mean_absolute_error'])
def test_shared_species_distinct_targets_and_constraints(metric):
    config = {'site_labels': ['A', 'B', 'A', 'B'],
              'allowed': {'A': ['Cu', 'Zn'], 'B': ['Cu', 'Zn']},
              'counts': {'A': {'Cu': 1, 'Zn': 1}, 'B': {'Cu': 1, 'Zn': 1}}}
    targets = [target(.3, 'A', 'B'), target(-.4, 'B', 'A', a='Zn', b='Cu')]
    auto, _ = enumerate_sro(ring(), stop_at_bound=False, cutoff=1.1, sublattices=config, targets=targets, metric=metric)
    full, _ = enumerate_sro(ring(), stop_at_bound=False, cutoff=1.1, sublattices=config, targets=targets, metric=metric, symmetry='none')
    assert auto['best_error_exact'] == full['best_error_exact']
    assert auto['covered_configurations'] == full['total_configurations'] == 4
    assert auto['evaluated_configurations'] < full['evaluated_configurations']
    # Unequal sublattice compositions must also remain fixed.
    atoms = ring(); atoms.set_chemical_symbols(['Cu', 'Cu', 'Cu', 'Zn'])
    config['counts']['A'] = {'Cu': 2, 'Zn': 0}
    auto, _ = enumerate_sro(atoms, stop_at_bound=False, cutoff=1.1, sublattices=config,
                            targets=[target(.2, 'A', 'B')], metric=metric)
    full, _ = enumerate_sro(atoms, stop_at_bound=False, cutoff=1.1, sublattices=config,
                            targets=[target(.2, 'A', 'B')], metric=metric, symmetry='none')
    assert auto['best_error_exact'] == full['best_error_exact']
    assert auto['covered_configurations'] == 2


def test_exact_bond_graph_rejects_invalid_geometric_reduction():
    # Deliberately supply a target graph that distinguishes one edge of the ring.
    # Geometry alone would allow eight permutations, but the target must not.
    atoms = ring()
    prepared = [(np.array([0, 1]), np.array([1, 0]), 'Cu', 'Zn', Fraction(1), Fraction(0))]
    labels = np.array(['all'] * 4)
    permutations = site_permutations(atoms, labels, prepared, symprec=.1)
    assert len(permutations) == 2
    assert all(set(p[[0, 1]]) == {0, 1} for p in permutations)
    unique_labels = np.array(['A', 'B', 'C', 'D'])
    assert len(site_permutations(atoms, unique_labels, prepared, symprec=1e-5)) == 1


def test_bcc_parent_symmetry_ignores_initial_chemical_decoration():
    atoms = bulk('Cu', 'bcc', a=2, cubic=True).repeat((2, 1, 1))
    atoms.symbols[2:] = 'Zn'
    reduced, _ = enumerate_sro(atoms, stop_at_bound=False, cutoff=1.8, targets=[target(-.1)])
    full, _ = enumerate_sro(atoms, stop_at_bound=False, cutoff=1.8, targets=[target(-.1)], symmetry='none')
    assert reduced['covered_configurations'] == 6
    assert reduced['evaluated_configurations'] < 6
    assert reduced['best_error_exact'] == full['best_error_exact']


def test_symmetry_group_closure_and_invalid_options():
    group = permutation_group([(1, 2, 0)], 3)
    assert len(group) == 3
    with pytest.raises(ValueError, match='exceeds'):
        permutation_group([(1, 2, 0)], 3, limit=2)
    with pytest.raises(ValueError, match='symprec'):
        enumerate_sro(ring(), stop_at_bound=False, cutoff=1.1, targets=[target()], symprec=0)


def test_interrupted_symmetry_search_reports_only_covered_orbits(monkeypatch):
    import enumeration_symmetry
    original = enumeration_symmetry.symmetry_occupations
    def interrupted(*args):
        yield next(original(*args))
        raise KeyboardInterrupt
    monkeypatch.setattr(enumeration_symmetry, 'symmetry_occupations', interrupted)
    result, _ = enumerate_sro(ring(), stop_at_bound=False, cutoff=1.1, targets=[target(.25)])
    assert result['status'] == 'best_found' and not result['complete']
    assert result['stop_reason'] == 'interrupted'
    assert result['evaluated_configurations'] == 1
    assert 1 < result['covered_configurations'] < result['total_configurations']
