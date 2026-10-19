"""Compare m_scraps with frozen ATAT results or a user's corrdump executable."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from ase import Atoms
from ase.io import read

from atat_reference import figure_model
from structure_analysis import analyze_structure

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'analysis_oracle'


def compare_case(root, case, *, corrdump=None, structure=None, tolerance=2e-10):
    """Return numeric discrepancies; custom structures need live ATAT.

    Bundled SRO conversions are for binary, equiatomic sublattices only.
    Composition checks in the analyzer enforce this assumption.
    """
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if case not in manifest['cases']:
        raise ValueError(f'Unknown case {case}; available: {manifest["cases"]}')
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Fixture checksum mismatch: {name}')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Comparison tolerance must be positive and finite')
    if structure is not None and corrdump is None:
        raise ValueError('Custom structures require --corrdump')
    data = json.loads((root / f'{case}.json').read_text())
    model = figure_model(root / f'{case}.figures.gz')
    species = data['case']['species']
    configurations = data['configurations'][:1] if structure is not None else data['configurations']
    results = []
    for config in configurations:
        atoms = read(structure) if structure is not None else Atoms(
            [species[i] for i in config['occupations']], cell=model['cell'],
            scaled_positions=model['positions'], pbc=True)
        report = analyze_structure(atoms, cutoff=max(data['case']['cutoffs'])+1e-6,
            shells=100, figures=root / f'{case}.figures.gz', species=species)
        vector = np.array([f['correlation'] for f in report['figures']['features']])
        oracle = np.array(config['correlations'])
        if corrdump is not None:
            with tempfile.TemporaryDirectory(prefix='m_scraps_atat_') as work:
                work = Path(work)
                for name in ['rndstr.in', 'clusters.out']:
                    shutil.copy2(root / f'{case}.{name}', work / name)
                labels = np.array(atoms.get_chemical_symbols())[report['mapping']['input_indices']]
                positions = model['positions'] @ model['cell']
                lines = ['1 0 0', '0 1 0', '0 0 1'] + [' '.join(f'{v:.17g}' for v in row) for row in model['cell']]
                lines += [' '.join(f'{v:.17g}' for v in row)+' '+symbol for row, symbol in zip(positions, labels)]
                (work / 'structure.out').write_text('\n'.join(lines)+'\n')
                proc = subprocess.run([str(corrdump), '-l=rndstr.in', '-ro', '-c', '-noe', '-nop',
                                       '-s=structure.out', '-sig=17'], cwd=work,
                                      text=True, capture_output=True, check=True, timeout=120)
                try:
                    oracle = np.array([float(x) for x in proc.stdout.split()])
                except ValueError as exc:
                    raise ValueError('ATAT returned nonnumeric correlation output') from exc
        if oracle.shape != vector.shape or not np.all(np.isfinite(oracle)):
            raise ValueError('ATAT returned an invalid correlation vector')
        sro_errors = []
        for expected in config['sro']:
            matches = [row for row in report['sro'] if all(row[k] == expected[k] for k in
                ['source', 'destination', 'center_species', 'neighbor_species']) and
                abs(row['distance_min']-expected['distance']) < 1e-6]
            if len(matches) != 1 or matches[0]['alpha'] is None:
                raise ValueError('ATAT SRO shell does not map uniquely to the analysis report')
            value = expected['spin_factor'] * (oracle[expected['feature_indices']] @ expected['weights'])
            sro_errors.append(abs(matches[0]['alpha'] - value))
        residual = oracle - [f['target'] for f in model['features']]
        expected_metrics = dict(max_error=float(max(abs(residual))), rms_error=float(np.sqrt(np.mean(residual**2))),
                                mean_absolute_error=float(np.mean(abs(residual))), weighted_l1=float(np.mean(abs(residual))))
        metric_errors = {key: abs(report['figures']['metrics'][key]-value) for key, value in expected_metrics.items()}
        error = float(max(abs(vector-oracle)))
        sro_error = float(max(sro_errors, default=0))
        archived_error = float(max(abs(oracle-config['correlations']))) if structure is None else None
        passed = max([error, sro_error, *metric_errors.values(), archived_error or 0]) <= tolerance
        results.append(dict(case=case, seed=config['seed'] if structure is None else None,
            source='live ATAT' if corrdump else 'frozen ATAT', custom_structure=str(structure) if structure else None,
            correlations=len(vector), sro_entries=len(sro_errors), max_correlation_discrepancy=error,
            max_sro_discrepancy=sro_error, metric_discrepancies=metric_errors,
            live_vs_archived_discrepancy=archived_error if corrdump else None,
            tolerance=tolerance, passed=passed, atat_correlations=oracle.tolist()))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--case', help='Default: all bundled cases')
    parser.add_argument('--corrdump', help='Path to ATAT corrdump; omit to use frozen results')
    parser.add_argument('--structure', type=Path, help='Own structure matching the selected case cell/composition')
    parser.add_argument('--tolerance', type=float, default=2e-10)
    parser.add_argument('--output', type=Path, help='Write JSON report; default stdout')
    args = parser.parse_args(argv)
    if args.structure and (not args.case or not args.corrdump):
        parser.error('--structure requires --case and --corrdump')
    executable = None
    if args.corrdump:
        executable = shutil.which(args.corrdump)
        if executable is None or not os.access(executable, os.X_OK):
            parser.error('corrdump executable not found')
        executable = str(Path(executable).resolve())
    try:
        manifest = json.loads((args.fixtures / 'manifest.json').read_text())
        rows = [row for case in ([args.case] if args.case else manifest['cases'])
                for row in compare_case(args.fixtures, case, corrdump=executable,
                    structure=args.structure, tolerance=args.tolerance)]
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    result = dict(passed=all(row['passed'] for row in rows), comparisons=rows,
                  corrdump_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest() if executable else None,
                  reference_atat_version=manifest['atat_version'])
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
