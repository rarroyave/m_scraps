#!/usr/bin/env python3
"""Fixed 40-trial submission benchmark; no adaptive campaign expansion."""
import argparse
import collections
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

TOOLS=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TOOLS))
from campaign import save
METHODS=('scraps_cuckoo','scraps_sa','mcsqs','icet_atat')
PROFILE=dict(name='uniform_figures',wd=0,wn=1,wr=1)
SEED=20261019


def job_matrix(cases):
    jobs=[(c,method,seed) for c in cases for method in METHODS
          for seed in ([SEED,SEED+1000000] if c['sites']==128 and c.get('target_mode')=='reference' else [SEED])]
    random.Random(SEED).shuffle(jobs)
    assert len(jobs)==40
    return jobs


def prepare(root,m):
    from ase.build import bulk
    from higher_order import export_figures
    from scraps_ase import write_scraps_poscar
    from sro_targets import make_targets
    inputs=root/'inputs';inputs.mkdir(exist_ok=True)
    cases=[]
    for n in [4,6]:
        count=n**3
        for kind in ['bcc','rocksalt']:
            c=dict(id=f'{kind}_{2*count}',kind=kind,sites=2*count,cell_dim=[n]*3,target_mode='random')
            if kind=='bcc':
                atoms=bulk('Nb','bcc',a=3.3,cubic=True);kw={}
                c.update(species=['Nb','Mo','Ta'],counts=[count,count//2,count//2],cutoffs=[3.4]*3,constraints=[])
            else:
                atoms=bulk('NaCl','rocksalt',a=4.5);kw=dict(sublattices=[(0,[0],[0,1,2]),(1,[1],[3,4])])
                c.update(species=['Ti','Zr','Hf','C','N'],counts=[count//2,count//4,count//4,count//4,3*count//4],cutoffs=[4.0]*3,constraints=['SUBLATTICE 0 BASIS 0 SPECIES 0 1 2','SUBLATTICE 1 BASIS 1 SPECIES 3 4'])
            path=inputs/(c['id']+'.figures')
            if not path.exists():
                export_figures(path,atoms,c['species'],c['counts'],c['cell_dim'],c['cutoffs'],**kw)
                write_scraps_poscar(atoms,inputs/(c['id']+'.poscar'))
                path.with_suffix('.figures.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
            counts=collections.Counter(f['order'] for f in m.model_for(c)['features'])
            assert set(counts)=={2,3,4};c['figure_counts']={str(k):v for k,v in counts.items()}
            cases.append(c);cases.extend(make_targets(root,c))
    for c in cases:
        save(root/'status.json',dict(state='validating_inputs',case=c['id'],pid=os.getpid()))
        m.prepare(c)
    save(root/'cases.json',cases)
    return cases


def summarize(record):
    candidates=record['candidates']
    for c in candidates:
        v=c['metrics'];counts=record['figure_counts']
        v['rms_error']=math.sqrt(sum(counts[o]*row['rms']**2 for o,row in v['by_order'].items())/sum(counts.values()))
    row={k:record[k] for k in ['case','method','seed','budget_seconds']}
    best=min(candidates,key=lambda c:c['metrics']['max_error']) if candidates else None
    row.update(outcome='validated_output' if best else 'no_output_within_budget',
               max_error=best['metrics']['max_error'] if best else None,
               rms_at_best_max=best['metrics']['rms_error'] if best else None,
               mean_absolute_at_best_max=best['metrics']['weighted_l1'] if best else None,
               by_order=best['metrics']['by_order'] if best else None)
    import benchmark_atat_reference as bench
    row['passage']={metric:[bench.first_passage(candidates,metric,t,record['budget_seconds']) for t in [.05,.025,.01]] for metric in ['max_error','rms_error','weighted_l1']}
    row['curves']=[dict(seconds=d,**{metric:min((c['metrics'][metric] for c in candidates if c['available_seconds']<=d),default=None) for metric in ['max_error','rms_error','weighted_l1']}) for d in [1,2,5,10,15,20,25,30]]
    return row


def export_fixtures(root,cases,records,m):
    import numpy as np
    from atat_reference import correlations
    dest=root/'regression-fixtures';dest.mkdir(exist_ok=True)
    for case in cases:
        model=m.model_for(case);parsed=m.parsed_for(case);examples=[]
        audit=root/'fixture-audit'/case['id'];audit.mkdir(parents=True,exist_ok=True)
        for name in ['rndstr.in','clusters.out']:
            if not (audit/name).exists():(audit/name).symlink_to(m.input_dir(case)/name)
        for r in [r for r in records if r['case']==case['id']]:
            example=dict(method=r['method'],seed=r['seed'],budget_seconds=30)
            if not r['candidates']:
                example.update(outcome='no_output_within_budget',passage=summarize(r)['passage'])
            else:
                best=min(r['candidates'],key=lambda c:c['metrics']['max_error']);path=Path(best['structure'])
                occ=m.occupations(path,model,case) if r['method']=='mcsqs' else np.loadtxt(path.parent/'FIGURE_OCCUPATIONS.txt',dtype=int)
                m.write_structure(audit/'probe.out',model,occ);oracle=m.atat_vector(audit,'probe.out')
                assert np.max(np.abs(correlations(parsed,occ)-oracle))<2e-10
                example.update(outcome='validated_output',occupations=occ.tolist(),oracle_correlations=oracle.tolist(),metrics=best['metrics'])
            examples.append(example)
        (dest/(case['id']+'.json.gz')).write_bytes(gzip.compress(json.dumps(dict(case=case,examples=examples)).encode(),mtime=0))
        shutil.copy2(root/'inputs'/(case['id']+'.figures.gz'),dest/(case['id']+'.figures.gz'))
    save(dest/'manifest.json',dict(cases=[c['id'] for c in cases],methods=METHODS,trials=40,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.name!='manifest.json'}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--atat-bin',type=Path,required=True,help='Directory containing mcsqs and corrdump 3.52')
    parser.add_argument('--binary',type=Path,default=TOOLS.parent/'SCRAPs/scraps')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from scraps_ase import _find_mpirun
    mpirun=_find_mpirun()
    if not mpirun or not args.binary.is_file():parser.error('Build SCRAPs and provide a matching MPI launcher')
    for name in ['mcsqs','corrdump']:
        if not (args.atat_bin/name).is_file():parser.error(f'Missing {name}')
    os.environ.update(SCRAPS_BENCHMARK_ROOT=str(root),SCRAPS_BENCHMARK_BINARY=str(args.binary.resolve()),MPIRUN=mpirun,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    import icet
    if icet.__version__!='4.0':parser.error('This frozen protocol requires icet==4.0')
    import matched_mcsqs as m
    import icet_extension as e
    m.BIN.mkdir(parents=True,exist_ok=True)
    for name in ['mcsqs','corrdump']:
        target=m.BIN/name;source=(args.atat_bin/name).resolve()
        if target.exists():assert m.digest(target)==m.digest(source),'ATAT executable changed'
        else:shutil.copy2(source,target)
    cases=prepare(root,m) if not (root/'cases.json').exists() else json.loads((root/'cases.json').read_text())
    protocol=dict(trials=40,seconds=30,ranks=4,methods=METHODS,objective=PROFILE,checkpoint_seconds=.25,
                  target='random or exact pair/triplet/quadruplet correlations of a seeded ordered reference; reference occupations never initialize searches',
                  sampling='1..10,15,20,25,29.98 seconds plus completed native outputs; actual read-completion timestamps; no post-budget candidates',
                  seeds='20261019 everywhere; 21261019 repeats only 128-site nonzero-target cases',
                  interpretation='Small reproducible screening benchmark; limited repetitions do not establish broad statistical superiority')
    save(root/'protocol.json',protocol)
    paths=[Path(__file__),args.binary.resolve(),*Path(__file__).parent.glob('*.py'),TOOLS/'benchmark_atat_reference.py',TOOLS/'atat_reference.py',TOOLS/'higher_order.py',*m.BIN.iterdir(),*sorted((root/'inputs').glob('*'))]
    fingerprints={str(p):m.digest(p) for p in paths if p.is_file()}
    if (root/'manifest.json').exists():assert json.loads((root/'manifest.json').read_text())==fingerprints,'Frozen source or input changed; use a new output directory'
    else:save(root/'manifest.json',fingerprints)
    if not (root/'environment.json').exists():
        versions=subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True).stdout
        save(root/'environment.json',dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),dependencies=versions,mpi=subprocess.run([mpirun,'--version'],capture_output=True,text=True,check=True).stdout))
    if args.prepare_only:return
    records=[];rows=[]
    for index,(case,method,seed) in enumerate(job_matrix(cases)):
        save(root/'status.json',dict(state='running',job=index+1,total=40,case=case['id'],method=method,pid=os.getpid()))
        out=root/'results'/case['id']/method/str(seed)
        record=e.run_record(case,PROFILE,seed,out,budget=30) if method=='icet_atat' else m.record_run(case,PROFILE,seed,out,method,budget=30)
        no_output={'No completed native output within budget','No valid mcsqs output within budget','No completed icet output within budget'}
        assert all(x.get('error') in no_output for x in record['errors']),record['errors']
        assert all(c['available_seconds']<=30 for c in record['candidates'])
        record['figure_counts']=case['figure_counts'];row=summarize(record)
        save(out/'result.json',record);records.append(record);rows.append(row);save(root/'summary.json',rows)
        print(json.dumps(dict(completed=index+1,case=case['id'],method=method,seed=seed,max_error=row['max_error'])),flush=True)
    save(root/'status.json',dict(state='exporting_regression_fixtures',completed=40,pid=os.getpid()))
    export_fixtures(root,cases,records,m)
    save(root/'status.json',dict(state='complete',completed=40,total=40))


if __name__=='__main__':
    main()
