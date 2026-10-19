#!/usr/bin/env python3
"""Bounded native SCRAPS, ATAT and icet extension of the small-cell benchmark."""
from __future__ import annotations
import argparse
import fcntl
import gzip
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read, write

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'submission'))
from atat_reference import figure_model, correlations
from higher_order import export_figures
from scraps_ase import write_scraps_poscar
from structure_analysis import analyze_structure
from sro_bounds import sro_bounds
from sro_targets import write_figures
from small_cells import save

REFERENCES = TOOLS / 'fixtures' / 'small_cells'
METHODS = ['scraps_cuckoo', 'scraps_sa', 'mcsqs', 'icet_atat']
PROFILE = dict(name='single_pair_uniform', wd=0, wn=1, wr=1)


def pair_analysis(atoms, case, model):
    labels = ['all'] * len(atoms) if case['kind'] == 'bcc' else [
        'M' if gid == 0 else 'X' for gid in model['site_groups']]
    config = dict(site_labels=labels, allowed=case['reference']['sublattices']['allowed'])
    return analyze_structure(atoms, cutoff=case['cutoffs'][0], shells=1,
                             sublattices=config, sro_targets=case['reference']['targets'])


def atoms_for(model, occ):
    return Atoms([model['species'][int(i)] for i in occ], cell=model['cell'],
                 scaled_positions=model['fractional_sites'], pbc=True)


def prepare(root, m):
    inputs = root / 'inputs'; inputs.mkdir(exist_ok=True)
    cases = []
    references = json.loads((REFERENCES / 'cases.json').read_text())['cases']
    for ref in references:
        initial = read(io.StringIO(ref['initial_extxyz']), format='extxyz')
        n = len(initial); kind = ref['id'].split('_')[0]
        dims = {8: [2,2,1], 16: [2,2,2], 32: [2,2,4], 48: [2,3,4]}[n]
        if kind == 'bcc':
            primitive = bulk('Cu', 'bcc', a=3, cubic=True)
            species = ['Cu','Zn']; counts = [n//2]*2; kwargs = {}; constraints = []
        else:
            primitive = bulk('TiC', 'rocksalt', a=4.4)
            species = ['Ti','Zr','C','N']; counts = [n//4]*4
            kwargs = dict(sublattices=[(0,[0],[0,1]), (1,[1],[2,3])])
            constraints = ['SUBLATTICE 0 BASIS 0 SPECIES 0 1', 'SUBLATTICE 1 BASIS 1 SPECIES 2 3']
        case = dict(id=ref['id'], kind=kind, sites=n, cell_dim=dims, species=species,
                    counts=counts, cutoffs=[ref['cutoff']], constraints=constraints,
                    target_mode='explicit', reference=ref)
        path = inputs / (case['id'] + '.figures')
        model = export_figures(path, primitive, species, counts, dims, case['cutoffs'], **kwargs)
        assert len(model['features']) == 1 and model['features'][0]['order'] == 2
        assert np.allclose(model['cell'], initial.cell, atol=1e-12)
        target = ref['targets'][0]['target']
        model['features'][0]['target'] = target
        parsed = figure_model(path); parsed['version'] = model['version']
        parsed['features'][0]['target'] = target
        write_figures(path, parsed)
        save(path.with_suffix('.figures.json'), model)
        path.with_suffix('.figures.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        write_scraps_poscar(primitive, inputs / (case['id'] + '.poscar'))
        # m.prepare compares three independent correlation vectors with corrdump.
        m.prepare(case)
        parsed = figure_model(path)
        rng = np.random.default_rng(20261019)
        discrepancies = []
        for _ in range(3):
            occ = np.empty(n, int)
            for gid, group in enumerate(model['sublattice_groups']):
                sites = np.flatnonzero(np.asarray(model['site_groups']) == gid)
                occ[sites] = rng.permutation(np.repeat(group['species'], group['counts']))
            actual = pair_analysis(atoms_for(model, occ), case, model)['sro_comparison']['metrics']['max_error']
            error = abs(float(correlations(parsed, occ)[0]) - target)
            discrepancies.append(abs(actual - error))
        assert max(discrepancies) < 2e-10
        save(inputs / (case['id'] + '.pair-sro-audit.json'), dict(
            correlation_equals_sro_for_equiatomic_binary_groups=True,
            max_error_discrepancies=discrepancies,
            explanation='With zero spin mean on each group, the binary pair correlation equals the selected unlike-species Warren-Cowley alpha.'))
        cases.append(case)
    save(root / 'cases.json', cases)
    return cases


def assess(record, case, root, m):
    model = m.model_for(case); parsed = m.parsed_for(case)
    row = dict(case=case['id'], method=record['method'], seed=record['seed'],
               budget_seconds=record['budget_seconds'], workers=record['ranks'],
               elapsed_seconds=record['search_wall_seconds'],
               lower_bound=case['reference']['lower_bound'], certified_optimum=case['reference']['optimum'],
               errors=record['errors'], timing_basis='process launch through observed output; imports included, input preparation excluded')
    if 'stop_reason' in record:
        row['stop_reason'] = record['stop_reason']
        row['stopping_certificate'] = record.get('stopping_certificate')
    no_output = {'No completed native output within budget', 'No valid mcsqs output within budget', 'No completed icet output within budget'}
    if any(error.get('error') not in no_output for error in record['errors']):
        raise RuntimeError(f"Invalid trial {row['case']} {row['method']}: {record['errors']}")
    candidates = record['candidates']
    if not candidates:
        row['status'] = 'no_output_within_budget'
        return row
    best = min(candidates, key=lambda c: c['metrics']['max_error'])
    path = Path(best['structure'])
    occ = (m.occupations(path, model, case) if row['method'] == 'mcsqs'
           else np.loadtxt(path.parent / 'FIGURE_OCCUPATIONS.txt', dtype=int, ndmin=1))
    atoms = atoms_for(model, occ)
    metrics = pair_analysis(atoms, case, model)['sro_comparison']['metrics']
    assert abs(metrics['max_error'] - best['metrics']['max_error']) < 2e-10
    if row['certified_optimum'] is not None:
        assert metrics['max_error'] + 2e-10 >= row['certified_optimum']
    audit = root / 'audit' / case['id'] / row['method']; audit.mkdir(parents=True, exist_ok=True)
    for name in ['rndstr.in', 'clusters.out']:
        if not (audit/name).exists(): (audit/name).symlink_to(m.input_dir(case)/name)
    m.write_structure(audit/'probe.out', model, occ)
    oracle = m.atat_vector(audit, 'probe.out')
    delta = float(np.max(np.abs(oracle-correlations(parsed, occ))))
    assert delta < 2e-10
    witness = audit / 'witness.extxyz'; write(witness, atoms)
    row.update(metrics)
    row.update(status='verified_output', gap_to_optimum=(max(0.,metrics['max_error']-row['certified_optimum']) if row['certified_optimum'] is not None else None),
               atat_discrepancy=delta, witness=str(witness.relative_to(root)),
               witness_extxyz=witness.read_text(), witness_sha256=hashlib.sha256(witness.read_bytes()).hexdigest(),
               site_labels=['all']*len(atoms) if case['kind']=='bcc' else ['M' if g==0 else 'X' for g in model['site_groups']],
               observed_candidates=len(candidates))
    for threshold in [.01,.05]:
        row[f'time_to_{threshold}'] = next((c['available_seconds'] for c in sorted(candidates, key=lambda c:c['available_seconds'])
                                          if c['metrics']['max_error'] <= threshold+1e-12), None)
    return row


def main():
    global REFERENCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCES, help='Frozen case directory with cases.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--mpirun', type=Path, required=True)
    parser.add_argument('--atat-bin', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stop-at-bound', action='store_true', help='Verify exact pair-SRO optimum before stopping workers')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    REFERENCES = args.reference.resolve()
    if not np.isfinite(args.seconds) or args.seconds < 2:
        parser.error('Use a finite budget of at least 2 seconds for subprocess startup and observations')
    for path in [args.binary,args.mpirun,args.atat_bin/'mcsqs',args.atat_bin/'corrdump']:
        if not path.is_file(): parser.error(f'Missing executable: {path}')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(SCRAPS_BENCHMARK_ROOT=str(root), OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    import icet
    if icet.__version__ != '4.0': parser.error('The recording adapter requires icet==4.0')
    import matched_mcsqs as m
    import icet_extension as e
    import benchmark_atat_reference as bench
    m.BIN.mkdir(parents=True, exist_ok=True)
    for name in ['mcsqs','corrdump']:
        dst = m.BIN/name
        if not dst.exists(): shutil.copy2(args.atat_bin/name, dst)
        assert dst.read_bytes() == (args.atat_bin/name).read_bytes()
    fingerprints = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),args.binary,args.mpirun,*m.BIN.iterdir(),
                     TOOLS/'sro_stopping.py',TOOLS/'sro_bounds.py',TOOLS/'sro_joint_bounds.py',TOOLS/'higher_order.py',TOOLS/'structure_analysis.py',TOOLS/'benchmark_atat_reference.py',
                     Path(m.__file__),Path(e.__file__),REFERENCES/'cases.json']}
    if (root/'manifest.json').exists(): assert json.loads((root/'manifest.json').read_text()) == fingerprints, 'Source changed; use a new output directory'
    else: save(root/'manifest.json',fingerprints)
    cases = json.loads((root/'cases.json').read_text()) if (root/'cases.json').exists() else prepare(root,m)
    protocol = dict(cases=len(cases), trials=len(cases)*len(METHODS), methods=METHODS, seed=args.seed, seconds=args.seconds, workers=1,
        objective='One binary pair correlation per case, equal to the selected pair-SRO alpha; ATAT score has the same ranking for these attainable discrete errors. icet uses the matched ATAT calculator, not its stock score.',
        initialization='Native engine initialization; seeds do not imply identical initial occupations across engines. Native SCRAPS restarts increment the seed.',
        timing='Process launch to observed output, including executable/import startup; input preparation and post-search verification excluded. ATAT/icet snapshots at 1-second intervals and budget-0.02; native checkpoints/completed outputs. First-observed times are upper bounds.',
        comparison='Earlier in-process Python SA timings exclude interpreter startup and include method preparation: no direct speedup claim against those times.',
        thresholds=[.01,.05], threshold_tolerance=1e-12, units='maximum absolute pair-SRO error; RMS and MAE identical for this single component',
        scope='Fixed cells, no shape search, no higher-order correlations; one-seed screen, not statistical ranking.')
    protocol['stop_at_bound'] = args.stop_at_bound
    if args.stop_at_bound:
        protocol['timing'] += ' Live certificate verification is included in the search budget.'
    if (root/'protocol.json').exists(): assert json.loads((root/'protocol.json').read_text()) == protocol, 'Protocol changed; use a new output directory'
    else: save(root/'protocol.json',protocol)
    save(root/'environment.json',dict(python=sys.version,platform=platform.platform(),hostname=platform.node(),
        dependencies=sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),
        icet=icet.__version__,numpy=np.__version__))
    if args.prepare_only: return
    jobs = [(c,method) for c in cases for method in METHODS]; random.Random(args.seed).shuffle(jobs)
    rows=[]; bench.FIXTURES=root/'inputs'
    for index,(case,method) in enumerate(jobs):
        save(root/'status.json',dict(state='running',completed=index,total=len(jobs),case=case['id'],method=method,pid=os.getpid()))
        out=root/'results'/case['id']/method/str(args.seed)
        stop_check = None
        if args.stop_at_bound:
            from sro_stopping import SROBoundStop
            model = m.model_for(case)
            occ = np.empty(case['sites'], int)
            for gid, group in enumerate(model['sublattice_groups']):
                occ[np.asarray(model['site_groups']) == gid] = np.repeat(group['species'], group['counts'])
            initial = atoms_for(model, occ)
            labels = ['all'] * len(initial) if case['kind'] == 'bcc' else ['M' if g == 0 else 'X' for g in model['site_groups']]
            stopper = SROBoundStop(initial, cutoff=case['cutoffs'][0], shells=1,
                targets=case['reference']['targets'],
                sublattices=dict(site_labels=labels, allowed=case['reference']['sublattices']['allowed']))
            def stop_check(occ):
                return stopper(atoms_for(model, occ))
        if (out/'result.json').exists(): record=json.loads((out/'result.json').read_text())
        else:
            if out.exists(): out.rename(out.with_name(out.name+'-interrupted-'+str(time.time_ns())))
            if method=='mcsqs': record=m.run_mcsqs(case,PROFILE,args.seed,out,budget=args.seconds,workers=1,stop_check=stop_check)
            elif method=='icet_atat': record=e.trial(case,PROFILE,args.seed,out,budget=args.seconds,workers=1,stop_check=stop_check)
            else:
                original=bench.input_text
                def native_input(c,p,s):
                    content=original(c,p,s)
                    if method=='scraps_sa': content=content.replace('TOTALIter 2\n','TOTALIter 40\n')+'MC_KERNEL sa\nSA_T_INITIAL 2.0\nSA_T_FINAL 0.001\n'
                    else: content+='MC_KERNEL cuckoo\n'
                    return content
                bench.input_text=native_input
                try: record=bench.run_trial(case,PROFILE,args.seed,SimpleNamespace(seconds=args.seconds,ranks=1,
                    binary=args.binary.resolve(),mpirun=str(args.mpirun.resolve()),checkpoint_seconds=.25),out,stop_check=stop_check)
                finally: bench.input_text=original
            if args.stop_at_bound:
                record['stopping_certificate'] = stopper.certificate
            record['method']=method; save(out/'result.json',record)
        row=assess(record,case,root,m); rows.append(row); save(root/'summary.json',rows)
        print(json.dumps({k:row.get(k) for k in ['case','method','max_error','gap_to_optimum','time_to_0.01']}),flush=True)
    save(root/'status.json',dict(state='complete',completed=len(rows),total=len(jobs)))

if __name__=='__main__': main()
