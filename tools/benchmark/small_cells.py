#!/usr/bin/env python3
"""Bounded, self-contained small-cell pair-SRO comparison (maximum 32 atoms)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import numpy as np
import ase
import spglib
from ase.build import bulk
from ase.io import read, write
from enumerate_sro import enumerate_sro, configuration_count
from sro_bounds import sro_bounds
from search_cells import _anneal, cell_geometry
from structure_analysis import analyze_structure


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def cases():
    """Fixed geometries and equiatomic fractions; only target SRO varies."""
    for system in ['bcc', 'rocksalt']:
        for size in [8, 16, 32]:
            if system == 'bcc':
                repeat = {8: (2, 2, 1), 16: (2, 2, 2), 32: (2, 2, 4)}[size]
                atoms = bulk('Cu', 'bcc', a=3, cubic=True).repeat(repeat)
                labels = ['all'] * size
                allowed = {'all': ['Cu', 'Zn']}
                source, destination, a, b, cutoff = 'all', 'all', 'Cu', 'Zn', 2.7
            else:
                repeat = {8: (2, 2, 1), 16: (2, 2, 2), 32: (2, 2, 4)}[size]
                atoms = bulk('TiC', 'rocksalt', a=4.4).repeat(repeat)
                labels = ['M', 'X'] * (size // 2)
                allowed = {'M': ['Ti', 'Zr'], 'X': ['C', 'N']}
                source, destination, a, b, cutoff = 'M', 'X', 'Ti', 'N', 2.3
            rng = np.random.default_rng(20261019 + size)
            symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
            for group, species in allowed.items():
                positions = np.flatnonzero(np.array(labels) == group)
                symbols[positions] = rng.permutation(np.repeat(species, len(positions) // 2))
            atoms.set_chemical_symbols(symbols.tolist())
            config = dict(site_labels=labels, allowed=allowed)
            for alpha in [0, -.1]:
                yield dict(id=f'{system}_{size}_{"random" if alpha == 0 else "sro"}',
                    system=system, size=size, alpha=alpha, atoms=atoms.copy(), config=config,
                    cutoff=cutoff, targets=[dict(source=source, destination=destination, shell=1,
                        center_species=a, neighbor_species=b, target=alpha)])


def render(root, records):
    columns = ['case', 'method', 'seed', 'status', 'max_error', 'rms_error', 'mean_absolute_error',
               'lower_bound', 'elapsed_seconds', 'time_to_0.01', 'time_to_0.05',
               'evaluations', 'total_configurations']
    rows = []
    for record in records:
        row = {k: record.get(k) for k in columns}
        rows.append(row)
    with (root / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    def value(v):
        return '—' if v is None else (f'{v:.6g}' if isinstance(v, float) else str(v))
    header = ''.join('<th>' + html.escape(k) + '</th>' for k in columns)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(value(row[k])) + '</td>' for k in columns) + '</tr>' for row in rows)
    (root / 'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>Small-cell m_scraps benchmark</title>
<style>body{font:15px system-ui;margin:32px;color:#172333}h1{font-size:26px}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border-bottom:1px solid #ccd4dd;text-align:right}th{background:#eaf0f6;position:sticky;top:0}tr:nth-child(even){background:#f6f8fa}p{max-width:1000px}</style>
<h1>Small-cell pair-SRO benchmark: 8–32 atoms</h1>
<p>Same fixed cells, compositions, initial occupations, targets and metrics across methods. Full enumeration, symmetry reduction and bound-aware stopping are compared with the cell-search Python SA, not native SCRAPS SA. One directed first-shell component per case makes max, RMS and MAE numerically identical. No higher-order correlation claims.</p>
<p>Enumeration is capped before symmetry reduction; skipped rows are explicit, not failed optimizations. SA has three seeds. Times include per-method preparation but exclude interpreter startup and validation. Threshold times are observed improvement callback times; a dash means not observed, not zero or the budget. Each final witness is independently reanalyzed. Geometry is held fixed: this is not a supercell-shape search. Bounds/certificates concern one cell only.</p>
<p>This limited deterministic screen does not establish broad statistical superiority. See protocol.json, environment.json, results.json and case witness files for reproducibility.</p>
<div style="overflow:auto"><table><thead><tr>''' + header + '</tr></thead><tbody>' + body + '</tbody></table></div>\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--max-configurations', type=int, default=20000)
    parser.add_argument('--sa-steps', type=int, default=10000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output directory must be new')
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.max_configurations < 1 or args.sa_steps < 1:
        parser.error('Budgets must be positive and finite')
    root = args.output.resolve(); root.mkdir(parents=True)
    methods = ['labeled_exhaustive', 'symmetry_exhaustive', 'symmetry_bound_stop', 'python_pair_sro_sa']
    protocol = dict(schema_version=1, systems=['bcc_CuZn', 'rocksalt_TiZr_CN'], sizes=[8,16,32], targets=[0,-.1],
        methods=methods, sa_seeds=[42,43,44], selected_metric='max_error', reported_metrics=['max_error','rms_error','mean_absolute_error'],
        components='One explicit first-shell directed component per case; no implicit targets.',
        max_configurations=args.max_configurations, seconds_per_trial=args.seconds, sa_steps=args.sa_steps,
        threads=1, initialization='Shared per-case seed 20261019 + atom count; exact fixed fractions per sublattice.',
        timing='Per-method preparation included; interpreter startup and post-trial verification excluded. Monotonic improvement callbacks record threshold observations.',
        thresholds=[.01,.05], limits='Time limits cooperative; library preparation may overrun. No global shape certificate; pair-SRO only.',
        objective='All methods optimize the identical explicit max pair-SRO error; native SCRAPS/ATAT/icet are outside this focused comparison.')
    save(root/'protocol.json', protocol)
    environment = dict(python=sys.version, platform=platform.platform(), hostname=platform.node(),
        numpy=np.__version__, ase=ase.__version__, spglib=spglib.__version__, cpu_count=os.cpu_count(),
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
        source_sha256={str(p.relative_to(TOOLS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), TOOLS/'enumerate_sro.py',TOOLS/'enumeration_symmetry.py',TOOLS/'sro_bounds.py',TOOLS/'search_cells.py',TOOLS/'structure_analysis.py']})
    save(root/'environment.json', environment)
    records = []
    for case in cases():
        directory = root/case['id']; directory.mkdir()
        write(directory/'initial.extxyz', case['atoms'])
        save(directory/'targets.json', case['targets']); save(directory/'sublattices.json', case['config'])
        start = time.monotonic()
        bound = sro_bounds(case['atoms'], cutoff=case['cutoff'], targets=case['targets'], sublattices=case['config'], shells=1)
        bound['benchmark_elapsed_seconds'] = time.monotonic()-start
        save(directory/'bounds.json',bound)
        total = configuration_count(bound['composition'])
        for method in methods:
            for seed in ([42,43,44] if method=='python_pair_sro_sa' else [None]):
                events = []
                record = dict(case=case['id'], system=case['system'], size=case['size'], target=case['alpha'],
                    method=method, seed=seed, lower_bound=bound['result']['lower_bound'], total_configurations=total,
                    geometry=cell_geometry(case['atoms'].cell))
                prefix = method + (f'-{seed}' if seed is not None else '')
                start = time.monotonic()
                try:
                    if method == 'python_pair_sro_sa':
                        result, witness = _anneal(case['atoms'],case['config'],case['targets'],case['cutoff'],1,
                            'max_error',args.sa_steps,seed,start+args.seconds,progress=events.append)
                    else:
                        result, witness = enumerate_sro(case['atoms'],cutoff=case['cutoff'],targets=case['targets'],
                            sublattices=case['config'],shells=1,max_configurations=args.max_configurations,
                            max_seconds=args.seconds,symmetry='none' if method=='labeled_exhaustive' else 'auto',
                            stop_at_bound=method=='symmetry_bound_stop',progress=events.append)
                except ValueError as exc:
                    if method=='python_pair_sro_sa' or 'configurations exceed' not in str(exc): raise
                    record.update(status='configuration_cap',elapsed_seconds=time.monotonic()-start,reason=str(exc),events=events)
                else:
                    elapsed = time.monotonic()-start
                    path = directory/(prefix+'.extxyz');write(path,witness)
                    check = analyze_structure(read(path),cutoff=case['cutoff'],sublattices=case['config'],shells=1,sro_targets=case['targets'])
                    metrics=check['sro_comparison']['metrics']
                    assert abs(metrics['max_error']-result['best_error'])<1e-12
                    assert result['best_error']+1e-12>=bound['result']['lower_bound']
                    record.update(status=result['status'],elapsed_seconds=elapsed,**metrics,
                        evaluations=result.get('evaluated_configurations',result.get('steps')),
                        certificate_basis=result.get('certificate_basis'),events=events,
                        witness=str(path.relative_to(root)),witness_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        verification='Independent analyze_structure of exported witness passed',result=result)
                    for threshold in protocol['thresholds']:
                        record[f'time_to_{threshold}']=next((e['elapsed_seconds'] for e in events if e['best_error']<=threshold+1e-12),None)
                records.append(record)
                save(directory/(prefix+'.json'),record)
                save(root/'results.json',records)
                print(json.dumps({k:record.get(k) for k in ['case','method','seed','status','max_error','elapsed_seconds']}),flush=True)
        exact = [r for r in records if r['case']==case['id'] and r['method']!='python_pair_sro_sa' and r['status']=='certified_minimum']
        if exact:
            optimum=exact[0]['max_error']
            assert all(abs(r['max_error']-optimum)<1e-12 for r in exact)
            assert all(r.get('max_error',float('inf'))+1e-12>=optimum for r in records if r['case']==case['id'])
    render(root,records)
    save(root/'completed.json',dict(trials=len(records),verified_witnesses=sum('witness' in r for r in records),status='complete'))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
