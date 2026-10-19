"""Scientific checks for evaluation of externally supplied structures."""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atat_reference import figure_model
from structure_analysis import analyze_structure, map_sites

ROOT = Path(__file__).resolve().parents[1]


def test_ordered_b2_has_known_sro_and_periodic_coordination():
    atoms = Atoms('CuZn', scaled_positions=[[0, 0, 0], [.5, .5, .5]], cell=[3, 3, 3], pbc=True)
    report = analyze_structure(atoms, cutoff=3.1, shells=2)
    rows = report['sro']
    first = [r for r in rows if r['shell'] == 1]
    assert all(r['coordination'] == 8 for r in first)
    for row in first:
        assert row['alpha'] == (1 if row['center_species'] == row['neighbor_species'] else -1)
    assert all(r['coordination'] == 6 for r in rows if r['shell'] == 2)


def test_sublattice_normalization_fixed_sites_and_composition():
    atoms = Atoms('TiZrBB', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], cell=[4, 5, 5], pbc=True)
    config = {'site_labels': ['M', 'M', 'B', 'B'], 'allowed': {'M': ['Ti', 'Zr'], 'B': ['B']},
              'counts': {'M': {'Ti': 1, 'Zr': 1}, 'B': {'B': 2}}}
    result = analyze_structure(atoms, cutoff=1.1, sublattices=config)
    cross = next(r for r in result['sro'] if r['source'] == 'M' and r['destination'] == 'M' and r['center_species'] == 'Ti' and r['neighbor_species'] == 'Zr')
    assert cross['alpha'] == -1  # c_Zr on M is 1/2, not global 1/4.
    fixed = [r for r in result['sro'] if r['destination'] == 'B']
    assert all(r['alpha'] == 0 and r['fixed_pair'] for r in fixed)
    atoms[2].symbol = 'Ti'
    with pytest.raises(ValueError, match='Forbidden'):
        analyze_structure(atoms, cutoff=1.1, sublattices=config)


def test_mapping_permutation_relaxation_and_ambiguity():
    ref = Atoms('CuZn', positions=[[0, 0, 0], [1, 0, 0]], cell=[4, 4, 4], pbc=True)
    relaxed = ref[[1, 0]]
    relaxed.positions += [.03, 0, 0]
    order, distance = map_sites(relaxed, ref)
    assert list(order) == [1, 0]
    np.testing.assert_allclose(distance, .03)
    with pytest.raises(ValueError, match='ambiguous'):
        map_sites(relaxed, ref, tolerance=2)
    with pytest.raises(ValueError, match='missing'):
        map_sites(relaxed, ref, tolerance=.01)
    rounded = ref.copy()
    rounded.cell[0, 0] += 3e-6
    assert list(map_sites(rounded, ref)[0]) == [0, 1]
    relaxed.cell *= 1.01
    with pytest.raises(ValueError, match='matching cells'):
        map_sites(relaxed, ref)


@pytest.mark.parametrize('case', ['bcc_128', 'rocksalt_128'])
def test_analysis_reproduces_atat_pair_triplet_quadruplet_vectors(case):
    root = ROOT / 'fixtures/higher_order_screen'
    data = json.loads(gzip.decompress((root / f'{case}.json.gz').read_bytes()))
    path = root / f'{case}.figures.gz'
    model = figure_model(path)
    for example in data['examples']:
        if 'occupations' not in example:
            continue
        atoms = Atoms([data['case']['species'][i] for i in example['occupations']],
                      scaled_positions=model['positions'], cell=model['cell'], pbc=True)
        result = analyze_structure(atoms[::-1], cutoff=3.5, figures=path, species=data['case']['species'])
        values = [f['correlation'] for f in result['figures']['features']]
        np.testing.assert_allclose(values, example['oracle_correlations'], atol=2e-10, rtol=0)
        for metric in ['max_error', 'rms_error', 'weighted_l1']:
            assert result['figures']['metrics'][metric] == pytest.approx(example['metrics'][metric], abs=2e-10)


def test_cli_json_csv_and_python_api(tmp_path):
    atoms = Atoms('CuZn', scaled_positions=[[0, 0, 0], [.5, .5, .5]], cell=[3, 3, 3], pbc=True)
    path = tmp_path / 'cell.extxyz'
    atoms.write(path)
    output, table = tmp_path / 'analysis.json', tmp_path / 'sro.csv'
    subprocess.run([sys.executable, str(ROOT / 'structure_analysis.py'), str(path),
                    '--cutoff', '3.1', '--output', str(output), '--csv', str(table)], check=True)
    assert len(json.loads(output.read_text())['sro']) == 8
    assert len(table.read_text().splitlines()) == 9
    sys.path.insert(0, str(ROOT.parent / 'python'))
    from m_scraps import analyze_structure as api
    assert api(atoms, cutoff=3.1)['sites'] == 2


def test_sro_target_residual_and_unavailable_target():
    atoms = Atoms('CuZn', scaled_positions=[[0, 0, 0], [.5, .5, .5]], cell=[3, 3, 3], pbc=True)
    target = dict(source='all', destination='all', shell=1, center_species='Cu', neighbor_species='Zn', target=-.75)
    report = analyze_structure(atoms, cutoff=3.1, sro_targets=[target])
    assert report['sro_comparison']['metrics'] == dict(max_error=.25, rms_error=.25, mean_absolute_error=.25)
    assert report['shell_coverage'][0]['reported_shells'] == 2
    with pytest.raises(ValueError, match='no evaluable'):
        analyze_structure(atoms, cutoff=1, sro_targets=[target])
    with pytest.raises(ValueError, match='Duplicate'):
        analyze_structure(atoms, cutoff=3.1, sro_targets=[target, target])


def test_materials_notebook_runs_without_external_engines(tmp_path, monkeypatch):
    import tempfile
    # Keep notebook artifacts under pytest's scratch directory.
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path))
    notebook = json.loads((ROOT.parent / 'notebooks/06_structure_analysis.ipynb').read_text())
    namespace = {}
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), f'notebook-cell-{i}', 'exec'), namespace)
    assert {name: report['sites'] for name, report in namespace['reports'].items()} == {
        'diboride': 24, 'max211': 64, 'heusler': 128}
    assert len(list(tmp_path.glob('*_analysis.json'))) == 3
