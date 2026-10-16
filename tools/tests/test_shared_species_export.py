"""Independent checks of shared-species, independently constrained groups."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('icet', minversion='4.0')
from ase import Atoms
from icet import ClusterSpace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from higher_order import export_figures, evaluate_figures, scraps_supercell


def b2():
    return Atoms('FeAl', scaled_positions=[[0,0,0],[.5,.5,.5]], cell=[3,3,3], pbc=True)


def test_identical_alphabets_remain_independent(tmp_path):
    # Physical atomic-number order is Al,Fe; gamma(Al)=-1, gamma(Fe)=+1.
    groups = [(10,[0],[0,1],[7,1]), (20,[1],[0,1],[3,5])]
    model = export_figures(tmp_path/'FIGURES', b2(), ['Fe','Al'], [10,6],
                           (2,2,2), [4.3,4.3,4.3], sublattices=groups)
    assert model['version'] == 3
    assert model['uses_auxiliary_labels']
    np.testing.assert_allclose(model['basis_moments'], [.75,-.25])
    expected_targets = set()
    for f in model['features']:
        target = sum(t['coefficient']*np.prod([.75 if s < 8 else -.25
                                             for s,b in t['factors']]) for t in f['terms'])
        assert f['target'] == pytest.approx(target)
        if f['order'] == 2:
            expected_targets.add(round(target,8))
    assert {.5625, -.1875, .0625} <= expected_targets
    # Independently construct group-preserving symmetry with physical-basis
    # signs mapped onto distinct chemical alphabets on the two basis sites.
    primitive = b2()
    primitive.symbols = ['H','Li']
    cs = ClusterSpace(primitive,[4.3,4.3,4.3],[['H','He'],['Li','Be']])
    cell = scraps_supercell(primitive,(2,2,2))
    rng = np.random.default_rng(271)
    for _ in range(5):
        occ = np.concatenate([rng.permutation([0]*7+[1]),rng.permutation([0]*3+[1]*5)])
        cell.symbols = [('He' if s == 0 else 'H') if i < 8 else
                        ('Be' if s == 0 else 'Li') for i,s in enumerate(occ)]
        expected = cs.get_cluster_vector(cell)[[f['index'] for f in model['features']]]
        np.testing.assert_allclose(evaluate_figures(model,occ),expected,atol=1e-12)


@pytest.mark.parametrize('case', ['partial', 'fixed'])
def test_partially_shared_and_fixed_alphabets(tmp_path, case):
    if case == 'partial':
        species, counts = ['Ni','Co','Fe'], [6,4,6]
        constraints = dict(sublattices=[(0,[0],[0,1],[6,2]), (1,[1],[1,2],[2,6])])
        # Ni-Co: (+1,-1), Co-Fe: (+1,-1).
        moments = [.5,-.5]
    else:
        species, counts = ['Fe','Al'], [12,4]
        constraints = dict(sublattices=[(0,[0],[0,1],[4,4])],spectator_basis_species=[(1,0)])
        moments = [0]
    model = export_figures(tmp_path/'FIGURES',b2(),species,counts,(2,2,2),[4.3,4.3,4.3],**constraints)
    assert model['uses_auxiliary_labels']
    np.testing.assert_allclose(model['basis_moments'],moments,atol=1e-12)
    if case == 'fixed':
        assert all(s < 8 for f in model['features'] for t in f['terms'] for s,b in t['factors'])
        assert model['site_allowed_species'][8:] == [[0]]*8
        assert all(abs(f['target']) < 1e-12 for f in model['features'])


@pytest.mark.parametrize('groups,counts,message', [
    ([(0,[0],[0,1]),(1,[1],[0,1])],[8,8],'explicit counts'),
    ([(0,[0],[0,1],[6,2]),(1,[1],[0,1])],[8,8],'every declared'),
    ([(0,[0],[0,1],[6,2]),(1,[1],[0,1],[6,2])],[8,8],'global ELEMENT'),
    ([(0,[0],[0,1],[7,2]),(1,[1],[0,1],[1,6])],[8,8],'capacity'),
    ([(0,[0],[0,1],[6.5,1.5]),(1,[1],[0,1],[1.5,6.5])],[8,8],'integers'),
])
def test_invalid_local_budgets(tmp_path, groups, counts, message):
    with pytest.raises(ValueError,match=message):
        export_figures(tmp_path/'FIGURES',b2(),['Fe','Al'],counts,(2,2,2),[4.3,4.3],sublattices=groups)


def test_zero_local_count_preserves_nominal_alphabet(tmp_path):
    model = export_figures(tmp_path/'FIGURES',b2(),['Fe','Al'],[8,8],(2,2,2),
                           [4.3,4.3,4.3],
                           sublattices=[(0,[0],[0,1],[8,0]),(1,[1],[0,1],[0,8])])
    np.testing.assert_allclose(model['basis_moments'],[1,-1],atol=1e-12)
    assert model['site_allowed_species'] == [[0,1]]*16
    occupations = [0]*8+[1]*8
    np.testing.assert_allclose(evaluate_figures(model,occupations),
                               [f['target'] for f in model['features']],atol=1e-12)
