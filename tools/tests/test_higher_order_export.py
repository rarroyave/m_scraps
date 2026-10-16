"""Exporter checks against icet and analytic independent-site moments."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('icet', minversion='4.0')
from ase.build import bulk
from icet import ClusterSpace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from higher_order import export_figures, evaluate_figures, scraps_supercell
from scraps_ase import make_scraps_inputs, run_sqs_from_atoms, _find_mpirun


@pytest.mark.parametrize('species', [['Ni','Cu'], ['Ni','Cu','Co'], ['Ni','Cu','Co','Fe'], ['Ni','Cu','Co','Fe','Cr']])
def test_all_decorations_and_targets(tmp_path, species):
    primitive = bulk('Cu','fcc',a=3.6, cubic=True)
    counts = [1]*(len(species)-1)+[32-len(species)+1]
    model = export_figures(tmp_path/'FIGURES', primitive, species, counts, (2,2,2), [3,3,3])
    cs = ClusterSpace(primitive,[3,3,3],species)
    cell = scraps_supercell(primitive,(2,2,2))
    rng = np.random.default_rng(42)
    occ = rng.permutation(np.repeat(np.arange(len(species)), counts))
    cell.symbols = [species[i] for i in occ]
    indices = [f['index'] for f in model['features']]
    np.testing.assert_allclose(evaluate_figures(model,occ),cs.get_cluster_vector(cell)[indices],atol=1e-12)
    moments = np.asarray(model['basis']) @ (np.asarray(counts)/32)
    for f in model['features']:
        assert f['target'] == pytest.approx(np.prod(moments[f['multicomponent_vector']]))
    assert {f['order'] for f in model['features']} == {2,3,4}
    assert (tmp_path/'FIGURES.json').exists()


def test_self_interaction_and_skew_ordering(tmp_path):
    primitive = bulk('Cu','fcc',a=3.6)
    cell = scraps_supercell(primitive,(2,1,1))
    np.testing.assert_allclose(cell.positions[1], primitive.cell[0])
    model = export_figures(tmp_path/'FIGURES', primitive, ['Ni','Cu'], [1,1], (2,1,1), [3,3,3])
    assert model['self_interacting']
    assert any(len({s for s,b in t['factors']}) < f['order']
               for f in model['features'] for t in f['terms'])
    cs = ClusterSpace(primitive,[3,3,3],['Ni','Cu'])
    cell.symbols = ['Ni','Cu']
    np.testing.assert_allclose(evaluate_figures(model,[0,1]),cs.get_cluster_vector(cell)[[f['index'] for f in model['features']]],atol=1e-12)


def test_api_and_rejections(tmp_path):
    primitive = bulk('Cu','fcc',a=3.6,cubic=True)
    kwargs = dict(figure_cutoffs=[3,3,3],figure_species=['Ni','Cu'],figure_wd=1,figure_wn=.5)
    make_scraps_inputs(tmp_path,primitive,[2,2],**kwargs)
    text = (tmp_path/'INPUT').read_text()
    assert 'FIGURES_FILE FIGURES' in text
    assert 'FIGURE_WN 0.5' in text
    with pytest.raises(ValueError,match='random-alloy'):
        make_scraps_inputs(tmp_path,primitive,[2,2],sro_targets=[(0,1,0,.1)],**kwargs)
    with pytest.raises(ValueError,match='finite'):
        make_scraps_inputs(tmp_path,primitive,[2,2],figure_wr=float('nan'),**kwargs)


@pytest.mark.parametrize('lattice', ['bcc', 'hcp'])
def test_non_fcc_figures(tmp_path, lattice):
    primitive = bulk('Cu', lattice, a=3.6)
    n = len(primitive)*8
    model = export_figures(tmp_path/'FIGURES', primitive, ['Cu','Co','Ni'],
                           [n//3,n//3,n-2*(n//3)], (2,2,2), [5,5,5])
    assert {f['order'] for f in model['features']} == {2,3,4}


def test_figure_parameter_validation(tmp_path):
    primitive = bulk('Cu','fcc',a=3.6,cubic=True)
    with pytest.raises(ValueError, match='require figure_cutoffs'):
        make_scraps_inputs(tmp_path, primitive, [2,2], figure_wd=1)
    with pytest.raises(ValueError, match='wn <= 1'):
        make_scraps_inputs(tmp_path, primitive, [2,2], figure_cutoffs=[3,3,3],
                           figure_species=['Ni','Cu'], figure_wn=2)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2]/'SCRAPs'/'scraps').exists()
                    or _find_mpirun() is None, reason='scraps binary not built or mpirun missing')
def test_native_optimization_matches_icet(tmp_path):
    import json
    primitive = bulk('Cu','fcc',a=3.6,cubic=True)
    # Fixed seed checks native output consistency, not stochastic success.
    run_sqs_from_atoms(primitive, ['Ni','Cu','Co'], [12,10,10], (2,2,2),
                       work_dir=tmp_path, ranks=2, figure_cutoffs=[3,3,3],
                       figure_wd=1, figure_wn=.5, nests=4, mcstep=10,
                       total_iter=1, auto_mcstep=0, rng_seed=42)
    model = json.loads((tmp_path/'FIGURES.json').read_text())
    occ = np.loadtxt(tmp_path/'FIGURE_OCCUPATIONS.txt',dtype=int)
    cell = scraps_supercell(primitive,(2,2,2))
    cell.symbols = [model['species'][i] for i in occ]
    cs = ClusterSpace(primitive,[3,3,3],model['species'])
    expected = cs.get_cluster_vector(cell)[[f['index'] for f in model['features']]]
    actual = np.genfromtxt(tmp_path/'FIGURE_CORRELATIONS.csv',delimiter=',',names=True)['correlation']
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual, evaluate_figures(model,occ), atol=1e-12)


def multisublattice_case(name):
    if name == 'rocksalt':
        return (bulk('NaCl', 'rocksalt', a=4.5), ['Ti','Zr','Hf','C','N'],
                [4,3,1,6,2], [4,4,4],
                dict(sublattices=[(0,[0],[0,1,2]), (1,[1],[3,4])]))
    from ase import Atoms
    primitive = Atoms('AlBB', scaled_positions=[[0,0,0],[1/3,2/3,.5],[2/3,1/3,.5]],
                      cell=[[3,0,0],[-1.5,3*np.sqrt(3)/2,0],[0,0,3.2]], pbc=True)
    return (primitive, ['Ti','Zr','Hf','Ta','Nb','B'], [2,2,2,1,1,16], [4.5]*3,
            dict(spectator_basis_species=[(1,5),(2,5)]))


@pytest.mark.parametrize('name', ['rocksalt','diboride'])
def test_multisublattice_export_and_local_targets(tmp_path, name):
    primitive, species, counts, cutoffs, constraints = multisublattice_case(name)
    model = export_figures(tmp_path/'FIGURES', primitive, species, counts,
                           (2,2,2), cutoffs, **constraints)
    assert model['version'] == 2
    assert {f['order'] for f in model['features']} == {2,3,4}
    allowed = [[species[s] for s in a] for a in model['basis_allowed_species']]
    cs = ClusterSpace(primitive, cutoffs, allowed)
    sublattices = cs.get_sublattices(cs.primitive_structure)
    local = {s.symbol: s.chemical_symbols for s in sublattices}
    from ase.data import atomic_numbers
    for f in model['features']:
        row = cs.as_list[f['index']]
        target = 1.0
        for symbol, function in zip(row['sublattices'], row['multicomponent_vector']):
            alphabet = sorted(local[symbol], key=atomic_numbers.get)
            total = sum(counts[species.index(s)] for s in alphabet)
            target *= sum(counts[species.index(s)]/total *
                          cs.evaluate_cluster_function(len(alphabet),function,i)
                          for i,s in enumerate(alphabet))
        assert f['target'] == pytest.approx(target, abs=1e-12)
    if name == 'rocksalt':
        for order in (2,3,4):
            assert any(f['order'] == order and len(set(cs.as_list[f['index']]['sublattices'])) == 2
                       for f in model['features'])
    else:
        assert len(species) == 6
        assert all(site < 8 for f in model['features'] for t in f['terms'] for site,_ in t['factors'])


@pytest.mark.parametrize('constraints,counts,message', [
    ({'sublattices': [(0,[0],[0,1]),(1,[1],[1,2])]}, [4,4,8], 'disjoint'),
    ({'sublattices': [(0,[0],[0,1]),(1,[1],[2])]}, [6,4,6], 'capacity'),
    ({'sublattices': [(0,[0],[0,1,2])]}, [4,4,8], 'capacity'),
    ({'spectator_basis_species': [(1,2)]}, [4,3,9], 'exactly'),
])
def test_multisublattice_invalid_partition(tmp_path, constraints, counts, message):
    with pytest.raises(ValueError,match=message):
        export_figures(tmp_path/'FIGURES', bulk('NaCl','rocksalt',a=4.5),
                       ['Ti','Zr','C'],counts,(2,2,2),[4]*3,**constraints)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2]/'SCRAPs'/'scraps').exists()
                    or _find_mpirun() is None, reason='scraps binary not built or mpirun missing')
@pytest.mark.parametrize('name', ['rocksalt','diboride'])
@pytest.mark.parametrize('fast', [0,1])
def test_native_multisublattice_preserves_constraints(tmp_path, name, fast):
    import json
    primitive, species, counts, cutoffs, constraints = multisublattice_case(name)
    make_scraps_inputs(tmp_path,primitive,counts,(2,2,2),figure_species=species,
                       figure_cutoffs=cutoffs,nests=4,mcstep=20,total_iter=2,
                       auto_mcstep=0,rng_seed=42,**constraints)
    with (tmp_path/'INPUT').open('a') as stream:
        stream.write(f'FAST_SWAPS {fast}\nCLUSTER_SWAP_K 3\nMC_KERNEL sa\n')
    import subprocess
    result = subprocess.run([_find_mpirun(),'-np','2',str(Path(__file__).resolve().parents[2]/'SCRAPs'/'scraps')],
                            cwd=tmp_path,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    model = json.loads((tmp_path/'FIGURES.json').read_text())
    occ = np.loadtxt(tmp_path/'FIGURE_OCCUPATIONS.txt',dtype=int)
    assert np.bincount(occ,minlength=len(species)).tolist() == counts
    assert all(int(s) in allowed for s,allowed in zip(occ,model['site_allowed_species']))
    cell = scraps_supercell(primitive,(2,2,2))
    cell.symbols = [species[i] for i in occ]
    cs = ClusterSpace(primitive,cutoffs,[[species[s] for s in a] for a in model['basis_allowed_species']])
    reference = cs.get_cluster_vector(cell)[[f['index'] for f in model['features']]]
    actual = np.genfromtxt(tmp_path/'FIGURE_CORRELATIONS.csv',delimiter=',',names=True)['correlation']
    np.testing.assert_allclose(actual,reference,atol=1e-12)
    np.testing.assert_allclose(actual,evaluate_figures(model,occ),atol=1e-12)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2]/'SCRAPs'/'scraps').exists()
                    or _find_mpirun() is None, reason='scraps binary not built or mpirun missing')
@pytest.mark.parametrize('damage', ['missing_constraints','wrong_allowed','version1'])
def test_native_rejects_figure_partition_mismatch(tmp_path, damage):
    import subprocess
    primitive,species,counts,cutoffs,constraints = multisublattice_case('rocksalt')
    make_scraps_inputs(tmp_path,primitive,counts,(2,2,2),figure_species=species,
                       figure_cutoffs=cutoffs,**constraints)
    if damage == 'missing_constraints':
        p = tmp_path/'INPUT'
        p.write_text('\n'.join(line for line in p.read_text().splitlines()
                               if not line.startswith('SUBLATTICE'))+'\n')
    else:
        p = tmp_path/'FIGURES'
        lines = p.read_text().splitlines()
        # Two header lines, 3 cell rows, 16 site rows, 3 local-function rows.
        start = 2+3+16+3
        if damage == 'wrong_allowed':
            lines[start] = '2 3 4'
        else:
            lines[0] = 'SCRAPS_FIGURES 1'
            del lines[start:start+16]
        p.write_text('\n'.join(lines)+'\n')
    result = subprocess.run([_find_mpirun(),'-np','1',str(Path(__file__).resolve().parents[2]/'SCRAPs'/'scraps')],
                            cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode != 0
    assert ('allowed species disagree' if damage != 'version1' else 'requires format version 2') in result.stdout+result.stderr


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2]/'SCRAPs'/'scraps').exists(),
                    reason='scraps binary not built')
def test_multisublattice_cli_dry_run(tmp_path):
    import json
    import subprocess
    primitive, species, counts, cutoffs, constraints = multisublattice_case('rocksalt')
    primitive.write(tmp_path/'POSCAR',format='vasp')
    (tmp_path/'partition.json').write_text(json.dumps(constraints['sublattices']))
    command = [sys.executable,str(Path(__file__).resolve().parents[1]/'run_sqs.py'),
               str(tmp_path/'POSCAR'),'--composition','0.25,0.1875,0.0625,0.375,0.125',
               '--target-atoms','16','--cell-dim','2,2,2','--sublattices',str(tmp_path/'partition.json'),
               '--species',','.join(species),'--figure-cutoffs','4,4,4','--dry-run',
               '--work-dir',str(tmp_path/'run')]
    result = subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    model = json.loads((tmp_path/'run'/'FIGURES.json').read_text())
    assert model['element_counts'] == counts
    assert model['basis_allowed_species'] == [[0,1,2],[3,4]]
    assert 'SUBLATTICE 1 BASIS 1 SPECIES 3 4' in (tmp_path/'run'/'INPUT').read_text()
