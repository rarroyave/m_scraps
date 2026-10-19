from fractions import Fraction
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from ase import Atoms
from sro_stopping import SROBoundStop
from search_cells import _anneal


def problem():
    atoms = Atoms('Cu2Zn2', positions=[[i, 0, 0] for i in range(4)], cell=[4, 8, 8], pbc=True)
    config = dict(site_labels=['all'] * 4, allowed={'all': ['Cu', 'Zn']})
    targets = [dict(source='all', destination='all', shell=1,
                    center_species='Cu', neighbor_species='Zn', target=0)]
    return atoms, config, targets


@pytest.mark.parametrize('metric', ['max_error', 'rms_error', 'mean_absolute_error'])
def test_exact_stopping_and_invalid_witnesses(metric):
    atoms, config, targets = problem()
    stop = SROBoundStop(atoms, cutoff=1.1, targets=targets, sublattices=config, metric=metric)
    assert stop(atoms)
    assert Fraction(stop.certificate['achieved_error_exact']) == 0
    worse = atoms.copy(); worse.set_chemical_symbols('CuZnCuZn')
    assert not stop(worse)
    wrong = atoms.copy(); wrong[0].symbol = 'Zn'
    with pytest.raises(ValueError, match='compositions'): stop(wrong)
    wrong = atoms.copy(); wrong.positions[0, 0] += .001
    with pytest.raises(ValueError, match='geometry'): stop(wrong)


def test_sa_stops_initial_and_during_search():
    atoms, config, targets = problem()
    report, witness = _anneal(atoms, config, targets, 1.1, 1, 'max_error', 1000, 42, time.monotonic()+10)
    assert report['stop_reason'] == 'bound_attained' and report['steps'] == 0
    atoms.set_chemical_symbols('CuZnCuZn')
    report, witness = _anneal(atoms, config, targets, 1.1, 1, 'max_error', 1000, 42, time.monotonic()+10)
    assert report['stop_reason'] == 'bound_attained' and 0 < report['steps'] < 1000
    assert SROBoundStop(atoms, cutoff=1.1, targets=targets)(witness)


def test_frozen_all_solver_stopping_witnesses():
    import io
    import json
    import hashlib
    from ase.io import read
    archive = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/bound_stopping/results.json').read_text())
    rows = archive['trials']
    assert len(rows) == 16
    assert {r['method'] for r in rows} == {'scraps_cuckoo', 'scraps_sa', 'mcsqs', 'icet_atat'}
    for row in rows:
        text = row['witness_extxyz']
        assert hashlib.sha256(text.encode()).hexdigest() == row['witness_sha256']
        atoms = read(io.StringIO(text), format='extxyz')
        stop = SROBoundStop(atoms, cutoff=row['cutoff'], targets=row['targets'],
                            shells=1, sublattices=row['sublattices'])
        attained = stop(atoms)
        # This graph's count relaxation is strictly below its realizable minimum.
        # Reaching a known optimum must NOT substitute for attaining the bound.
        if row['case'] == 'rocksalt_8_sro':
            assert not attained and row['stop_reason'] == 'time_limit'
            assert row['certificate'] is None
        else:
            assert attained and row['stop_reason'] == 'bound_attained'
            for key in ('lower_bound_exact', 'achieved_error_exact', 'pair_counts', 'occupations_sha256'):
                assert stop.certificate[key] == row['certificate'][key]
        assert not row['errors']
        assert row['atat_discrepancy'] < 2e-10


def test_public_stopping_api():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
    from m_scraps import SROBoundStop as public_stop
    assert public_stop is SROBoundStop
