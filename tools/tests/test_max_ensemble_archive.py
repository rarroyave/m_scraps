"""Offline audit of saved MAX structures and energy arithmetic; no ML calculator."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ase.io import read

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from atat_reference import figure_model, correlations
from enumeration_symmetry import site_permutations
from sro_stopping import SROBoundStop
from structure_analysis import analyze_structure, map_sites

FIXTURE = TOOLS/'fixtures/max_ensemble'


def test_archive_integrity_and_model_provenance():
    checksums = json.loads((FIXTURE/'checksums.json').read_text())
    for name, expected in checksums['files'].items():
        assert hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest() == expected, name
    manifest = json.loads((FIXTURE/'random32-energies/manifest.json').read_text())
    assert manifest['ensemble_sha256'] == checksums['files']['random32/result.json']
    assert manifest['script_sha256'] == checksums['files']['sources/ensemble_energy.py']
    assert manifest['relax_cell'] is False
    assert manifest['fmax'] == .03


def test_random_max_constraints_correlations_and_saved_energy_statistics():
    report = json.loads((FIXTURE/'random32/result.json').read_text())
    results = json.loads((FIXTURE/'random32-energies/summary.json').read_text())
    stats = json.loads((FIXTURE/'distribution-summary.json').read_text())
    assert report['selection']['method'] == 'random'
    assert report['selection']['seed'] == 17
    assert report['search']['pool_size'] == 128
    assert len(report['selected']) == len(results['structures']) == 32
    pool = report['selection']['pool_occupations_sha256']
    indices = np.random.default_rng(17).choice(len(pool), size=32, replace=False).tolist()
    assert [e['pool_index'] for e in report['selected']] == indices
    model = figure_model(FIXTURE/'random32/FIGURES')
    alphabet = report['diversity']['species']
    seen = set()
    for entry, row in zip(report['selected'], results['structures']):
        assert entry['id'] == row['id']
        path = FIXTURE/'random32'/entry['id']/'structure.extxyz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['structure_sha256'] == row['input_sha256']
        assert entry['occupations_sha256'] == pool[entry['pool_index']]
        atoms = read(path)
        checker = SROBoundStop(atoms, cutoff=3.5, shells=2, targets=report['targets'], sublattices=report['sublattices'])
        permutations = site_permutations(atoms, checker.labels, checker.prepared, 1e-5)
        symbols = np.array(atoms.get_chemical_symbols())
        assert hashlib.sha256(','.join(symbols).encode()).hexdigest() == entry['occupations_sha256']
        vector = correlations(model, [alphabet.index(s) for s in symbols])
        np.testing.assert_allclose(vector, entry['diversity_vector'], atol=1e-14)
        np.testing.assert_allclose(vector, row['initial_correlations'], atol=1e-14)
        canonical = min(tuple(symbols[p]) for p in permutations)
        assert canonical not in seen
        seen.add(canonical)
        analysis = analyze_structure(atoms, cutoff=3.5, shells=2,
            sublattices=report['sublattices'], sro_targets=report['targets'])
        assert analysis['sro_comparison']['metrics']['max_error'] == 0
        for group, composition in analysis['composition'].items():
            assert composition['counts'] == report['sublattices']['counts'][group]
        final_path = FIXTURE/'random32-energies'/entry['id']/'relaxed.extxyz'
        assert hashlib.sha256(final_path.read_bytes()).hexdigest() == row['relaxed_sha256']
        final = read(final_path)
        np.testing.assert_allclose(final.cell, atoms.cell, atol=1e-10)
        assert final.get_chemical_symbols() == atoms.get_chemical_symbols()
        mapping, _ = map_sites(final, atoms, tolerance=.75)
        np.testing.assert_array_equal(mapping, np.arange(len(atoms)))
        assert row['force_converged'] and row['final_max_force_eV_A'] <= .03
    for name, key in [('Unrelaxed','unrelaxed_eV_per_atom'), ('Fixed-cell relaxed','relaxed_eV_per_atom')]:
        energies = np.array([r[key] for r in results['structures']])
        expected = stats['stats'][name]
        np.testing.assert_allclose(energies.mean(), expected['mean_eV_per_atom'], atol=1e-12)
        np.testing.assert_allclose(1000*energies.std(), expected['population_std_meV_per_atom'], atol=1e-10)
        np.testing.assert_allclose(1000*np.ptp(energies), expected['range_meV_per_atom'], atol=1e-10)
