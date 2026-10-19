#!/usr/bin/env python3
"""Benchmark current SCRAPs against immutable ATAT data; never execute ATAT."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import time

import numpy as np
from atat_reference import FIXTURES, manifest, figure_model, correlations, metrics
from scraps_ase import _find_mpirun

REPO = Path(__file__).resolve().parents[1]


def first_passage(candidates, metric, threshold, budget):
    hits=[c['available_seconds'] for c in candidates if c['metrics'][metric]<=threshold]
    return dict(threshold=threshold,success=bool(hits),seconds=min(hits) if hits else None,
                right_censored=not hits,censor_seconds=budget if not hits else None)


def validate_counts(model, case, occ):
    if len(occ)!=case['sites'] or np.any(occ<0) or np.any(occ>=len(case['species'])):
        raise ValueError('Invalid occupation indices')
    if np.bincount(occ,minlength=len(case['species'])).tolist()!=case['counts']:
        raise ValueError('Global composition changed')
    if model['groups']:
        for group in set(model['groups']):
            if group<0:
                continue  # Fixed sites are checked by their singleton alphabet.
            sites=[i for i,g in enumerate(model['groups']) if g==group]
            actual=np.bincount(occ[sites],minlength=model['species']).tolist()
            if actual!=model['local_counts'][sites[0]]:
                raise ValueError('Local composition changed')


def input_text(case, profile, seed):
    return ('TYP '+str(len(case['species']))+'\nELEMENT '+' '.join(map(str,case['counts']))+
            '\nCell_DIM '+' '.join(map(str,case['cell_dim']))+'\n'+
            '\n'.join(case['constraints'])+'\n'+
            f'MAXSHELLNUM 4\nWEIGHT 1 1 1 1\nNESTS 8\nMCSTEP 32\nGITER 1\nTOTALIter 2\n'
            f'EARLY_STOP_PATIENCE 0\nRNG_SEED {seed}\nFIGURES_FILE FIGURES\nFIGURE_OBJECTIVE atat\n'
            f'FIGURE_WD {profile["wd"]}\nFIGURE_WN {profile["wn"]}\nFIGURE_WR 1\nFIGURE_TOL 0.001\n')


def run_trial(case, profile, seed, args, directory, stop_check=None):
    directory.mkdir(parents=True)
    prepared=time.perf_counter()
    figures=directory/'FIGURES';figures.write_bytes(gzip.decompress((FIXTURES/(case['id']+'.figures.gz')).read_bytes()))
    shutil.copyfile(FIXTURES/(case['id']+'.poscar'),directory/'POSCAR')
    model=figure_model(figures)
    preparation=time.perf_counter()-prepared
    snapshots=[]
    checkpoint_seconds=getattr(args, "checkpoint_seconds", 0.0)
    events=[];errors=[];restart=0;start=time.perf_counter();timed_out=False
    attained=False
    while not attained and time.perf_counter()-start<args.seconds:
        d=directory/f'restart-{restart:05d}';d.mkdir()
        for name in ['FIGURES','POSCAR']:(d/name).symlink_to(Path('..')/name)
        (d/'INPUT').write_text(input_text(case,profile,seed+restart) +
            (f'FIGURE_CHECKPOINT_SECONDS {checkpoint_seconds}\n' if checkpoint_seconds else ''))
        remaining=args.seconds-(time.perf_counter()-start)
        if remaining<=0:break
        with (d/'stdout.log').open('w') as log:
            proc=subprocess.Popen([args.mpirun,'-np',str(args.ranks),str(args.binary)],cwd=d,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                if not checkpoint_seconds:
                    proc.wait(timeout=remaining)
                else:
                    grid=sorted(set([float(x) for x in range(1,11)]+[15.,20.,25.,max(0.,float(args.seconds)-0.02),float(args.seconds)]))
                    while proc.poll() is None:
                        elapsed=time.perf_counter()-start
                        if elapsed>=args.seconds:
                            raise subprocess.TimeoutExpired(proc.args,args.seconds)
                        deadline=min([x for x in grid if x>elapsed]+[float(args.seconds)])
                        try:
                            proc.wait(timeout=max(0.001,deadline-elapsed))
                        except subprocess.TimeoutExpired:
                            pass
                        for path in sorted(d.glob('FIGURE_CHECKPOINT.rank*.json')):
                            raw=path.read_bytes()
                            observed=time.perf_counter()-start
                            if observed<=args.seconds:
                                snapshots.append((observed,raw))
                                if stop_check is not None and stop_check(np.asarray(json.loads(raw)["occupations"], dtype=int)):
                                    attained=True
                                    break
                        if attained:
                            break
            except subprocess.TimeoutExpired:
                timed_out=True
                break
            finally:
                if proc.poll() is None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        proc.wait()
        available=time.perf_counter()-start
        if attained:
            break
        if proc.returncode:
            errors.append(dict(restart=restart,returncode=proc.returncode,log=str(d/'stdout.log')))
            break
        if available<=args.seconds:
            events.append((available,d))
            if stop_check is not None:
                attained=bool(stop_check(np.loadtxt(d/"FIGURE_OCCUPATIONS.txt",dtype=int,ndmin=1)))
        restart+=1
    search_wall=time.perf_counter()-start
    # Independent assessment occurs after all search processes have stopped.
    assessed=time.perf_counter();candidates=[]
    # Materialize immutable observations only after search, without charging
    # independent validation to an optimizer's wall-clock budget.
    for index,(available,raw) in enumerate(snapshots):
        snapshot=json.loads(raw)
        d=directory/f'checkpoint-{index:05d}';d.mkdir()
        occ=np.asarray(snapshot['occupations'],dtype=int)
        np.savetxt(d/'FIGURE_OCCUPATIONS.txt',occ,fmt='%d')
        with (d/'FIGURE_CORRELATIONS.csv').open('w') as stream:
            stream.write('component,order,diameter,target,correlation,error\n')
            for i,(f,v) in enumerate(zip(model['features'],snapshot['correlations'])):
                stream.write(f"{i},{f['order']},{f['diameter']},{f['target']},{v},{v-f['target']}\n")
        (d/'stdout.log').write_text(f"FINAL fitness = {snapshot['score']}\n")
        from ase import Atoms
        from ase.io import write
        atoms=Atoms(symbols=[case['species'][i] for i in occ],cell=model['cell'],
                    scaled_positions=model['positions'],pbc=True)
        write(d/'SCRAPS.vasp',atoms,format='vasp',sort=True)
        events.append((available,d))
    for available,d in events:
        try:
            occ=np.loadtxt(d/'FIGURE_OCCUPATIONS.txt',dtype=int,ndmin=1)
            validate_counts(model,case,occ)
            vector=correlations(model,occ)
            table=np.genfromtxt(d/'FIGURE_CORRELATIONS.csv',delimiter=',',names=True)
            reported=np.atleast_1d(table['correlation'])
            targets=np.atleast_1d(table['target'])
            if reported.shape!=vector.shape or not np.all(np.isfinite(reported)) or not np.all(np.isfinite(targets)):
                raise ValueError('Malformed or nonfinite native correlation output')
            discrepancy=float(np.max(np.abs(vector-reported)))
            if discrepancy>2e-10:raise ValueError(f'Native correlation mismatch {discrepancy}')
            if np.max(np.abs(table['target']-[f['target'] for f in model['features']]))>2e-10:
                raise ValueError('Native targets disagree with reference fixture')
            found=re.findall(r'FINAL fitness[^\n]*=\s*([-+\deE.]+)',(d/'stdout.log').read_text())
            if not found:raise ValueError('Missing final objective score')
            score=float(found[-1])
            if not math.isfinite(score):raise ValueError('Nonfinite score')
            candidates.append(dict(available_seconds=available,native_score=score,
                                   metrics=metrics(model,vector,profile['wd'],profile['wn']),
                                   native_correlation_discrepancy=discrepancy,
                                   occupations_sha256=hashlib.sha256((d/'FIGURE_OCCUPATIONS.txt').read_bytes()).hexdigest(),
                                   structure=str(d/'SCRAPS.vasp')))
        except (ValueError,OSError,IndexError) as e:
            errors.append(dict(restart=d.name,error=str(e)))
    passage={metric:[first_passage(candidates,metric,t,args.seconds) for t in thresholds]
             for metric,thresholds in [('max_error',[.05,.025,.01,.001]),('weighted_l1',[.01,.005,.002,.001])]}
    checkpoints=sorted(set([x for x in [1,2,5,10,15,30,60,120,300] if x<=args.seconds]+[args.seconds]))
    curves=[]
    for deadline in checkpoints:
        eligible=[c for c in candidates if c['available_seconds']<=deadline]
        selected=min(eligible,key=lambda c:(c['native_score'],c['metrics']['weighted_l1'],c['metrics']['max_error'])) if eligible else None
        curves.append(dict(seconds=deadline,best_by_metric={m:min(c['metrics'][m] for c in eligible) if eligible else None for m in passage},atat_selected_metrics=selected['metrics'] if selected else None))
    record=dict(case=case['id'],profile=profile['name'],seed=seed,ranks=args.ranks,budget_seconds=args.seconds,
                preparation_seconds=preparation,search_wall_seconds=search_wall,assessment_seconds=time.perf_counter()-assessed,
                allocated_core_seconds=args.ranks*args.seconds,unfinished_restart_at_deadline=timed_out,
                checkpoint_seconds=checkpoint_seconds,
                errors=errors,candidates=candidates,passage=passage,quality_curve=curves)
    if stop_check is not None:
        record['stop_reason']='bound_attained' if attained else 'time_limit'
    (directory/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def summary(records):
    rows=[]
    for case,profile in sorted({(r['case'],r['profile']) for r in records}):
        selected=[r for r in records if (r['case'],r['profile'])==(case,profile)]
        for metric in ['max_error','weighted_l1']:
            for i in range(4):
                entries=[r['passage'][metric][i] for r in selected]
                times=sorted(e['seconds'] for e in entries if e['success'])
                n=len(entries);k=len(times);rank=math.ceil(n/2)
                z=1.95996398454;p=k/n;den=1+z*z/n
                center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
                rows.append(dict(case=case,profile=profile,metric=metric,threshold=entries[0]['threshold'],
                                 trials=n,successes=k,right_censored=n-k,success_fraction=p,
                                 wilson95=[center-half,center+half],
                                 median_observed_passage_seconds=times[rank-1] if k>=rank else None))
    return rows


def historical_comparison(records, ranks):
    baseline=json.loads(gzip.decompress((FIXTURES/'mcsqs-baseline.json.gz').read_bytes()))
    rows=[]
    for case,profile in sorted({(r['case'],r['profile']) for r in records}):
        for deadline in [5,15,60]:
            ours=[c['atat_selected_metrics'] for r in records if (r['case'],r['profile'])==(case,profile)
                  for c in r['quality_curve'] if c['seconds']==deadline]
            refs=[s.get('metrics') for r in baseline['trials'] if (r['case'],r['profile'])==(case,profile)
                  for s in r['observations'] if s['deadline']==deadline]
            if not ours or not refs:continue
            def stats(values):
                valid=[v for v in values if v is not None]
                return dict(trials=len(values),valid=len(valid),
                            median_weighted_l1=statistics.median(v['weighted_l1'] for v in valid) if valid else None,
                            success_max_error_025=sum(v['max_error']<=.025 for v in valid))
            rows.append(dict(case=case,profile=profile,seconds=deadline,current_scraps=stats(ours),
                             archived_mcsqs=stats(refs),same_allocated_cores=ranks==baseline['cores']))
    return dict(provenance={k:v for k,v in baseline.items() if k!='trials'},
                warning='Historical fixed-budget comparison, not a same-host speedup measurement. Compare only matching resource budgets; frozen mcsqs sample is incomplete. Multi-sublattice mcsqs timing baselines are not available.',rows=rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New directory; existing output is never overwritten')
    parser.add_argument('--cases',nargs='+',help='Default: all five frozen cases')
    parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--checkpoint-seconds',type=float,default=0.,help='Native snapshot interval; 0 disables')
    parser.add_argument('--ranks',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261016)
    parser.add_argument('--profile',choices=['geometric_decay','atat_default'],default='geometric_decay')
    parser.add_argument('--binary',type=Path,default=REPO/'SCRAPs/scraps')
    parser.add_argument('--mpirun',default=_find_mpirun())
    args=parser.parse_args(argv)
    data=manifest();args.binary=args.binary.resolve();args.output=args.output.resolve()
    if args.trials<1 or args.ranks<1 or not math.isfinite(args.seconds) or args.seconds<=0:
        parser.error('trials/ranks/seconds must be positive')
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds<0:
        parser.error('checkpoint-seconds must be finite and nonnegative')
    if not args.binary.is_file() or not args.mpirun:parser.error('Build SCRAPs and provide a matching MPI launcher')
    args.mpirun=shutil.which(args.mpirun) or args.mpirun
    cases=[c for c in data['cases'] if not args.cases or c['id'] in args.cases]
    if args.cases and set(args.cases)-{c['id'] for c in cases}:parser.error('Unknown case ID')
    args.output.mkdir(parents=True,exist_ok=False)
    profile=dict(name=args.profile,wd=1 if args.profile=='geometric_decay' else 0,wn=.5 if args.profile=='geometric_decay' else 1)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True)
    dirty=subprocess.run(['git','status','--porcelain'],cwd=REPO,capture_output=True,text=True)
    provenance=dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
                    cpu_count=os.cpu_count(),python=platform.python_version(),binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    git_commit=commit.stdout.strip(),git_dirty=bool(dirty.stdout.strip()),numpy_version=np.__version__,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    evaluator_sha256=hashlib.sha256((REPO/'tools/atat_reference.py').read_bytes()).hexdigest(),
                    fixture_manifest_sha256=hashlib.sha256((FIXTURES/'manifest.json').read_bytes()).hexdigest(),
                    command=vars(args),timing='Native initialization, repeated input loading, MPI startup and search included. Frozen fixture unpacking/model loading timed separately; original figure generation excluded. Assessment excluded. Passage is first observed completed output or enabled atomic checkpoint, not exact internal attainment.')
    provenance['command']={k:str(v) if isinstance(v,Path) else v for k,v in provenance['command'].items()}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    records=[]
    for case in cases:
        for i in range(args.trials):
            record=run_trial(case,profile,args.seed+i*1000000,args,args.output/case['id']/str(i))
            records.append(record)
            (args.output/'summary.json').write_text(json.dumps(summary(records),indent=2)+'\n')
            print(json.dumps(dict(case=case['id'],trial=i,completed_candidates=len(record['candidates']),errors=record['errors'])),flush=True)
    (args.output/'historical-comparison.json').write_text(json.dumps(historical_comparison(records,args.ranks),indent=2)+'\n')
    return 1 if any(r['errors'] for r in records) else 0


if __name__=='__main__':
    raise SystemExit(main())
