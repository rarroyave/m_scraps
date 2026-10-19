"""Matched multi-sublattice ATAT audit and four-worker mcsqs benchmarks."""
import argparse,hashlib,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(os.environ['SCRAPS_BENCHMARK_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import benchmark_atat_reference as bench
from atat_reference import figure_model,correlations,metrics
from campaign import save
MATCH=ROOT/'matched-atat'
BIN=MATCH/'bin'
PROFILES=[dict(name='geometric_decay',wd=1,wn=.5,wr=1),dict(name='atat_default',wd=0,wn=1,wr=1)]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def model_for(case):return json.loads((ROOT/'inputs'/(case['id']+'.figures.json')).read_text())
def parsed_for(case):return figure_model(ROOT/'inputs'/(case['id']+'.figures'))
def input_dir(case):return MATCH/'inputs'/case['id']

def labels_for(model):
    result=[]
    for gid,g in enumerate(model['sublattice_groups']):
        result.append({s:model['species'][s]+(f'_S{gid}' if model['uses_auxiliary_labels'] else '') for s in g['species']})
    return result

def write_structure(path,model,occ):
    labels=labels_for(model)
    rows=['1 0 0','0 1 0','0 0 1']+[' '.join(f'{x:.17g}' for x in r) for r in model['cell']]
    xyz=np.asarray(model['fractional_sites'])@np.asarray(model['cell'])
    rows+=[' '.join(f'{x:.17g}' for x in r)+' '+labels[g][int(s)] for r,g,s in zip(xyz,model['site_groups'],occ)]
    path.write_text('\n'.join(rows)+'\n')

def atat_vector(directory,structure):
    proc=subprocess.run([str(BIN/'corrdump'),'-l=rndstr.in','-ro','-c','-noe','-nop',f'-s={structure}','-sig=17'],cwd=directory,capture_output=True,text=True,check=True)
    return np.fromstring(proc.stdout,sep=' ')

def occupations(path,model,case):
    rows=[r.split() for r in path.read_text().splitlines() if r.strip()]
    axes=np.asarray(rows[:3],float);cell=np.asarray(rows[3:6],float)@axes
    if not np.allclose(cell,model['cell'],atol=1e-7,rtol=0):raise ValueError('Supercell geometry changed')
    frac=(np.asarray([r[:3] for r in rows[6:]],float)@axes)@np.linalg.inv(cell)
    expected=np.asarray(model['fractional_sites']);delta=frac[:,None,:]-expected[None,:,:];delta-=np.rint(delta)
    distance=np.linalg.norm(delta@cell,axis=2);match=np.argmin(distance,axis=1)
    if len(match)!=len(expected) or len(set(match))!=len(expected) or distance.min(axis=1).max()>1e-6:raise ValueError('Site matching failed')
    labels=labels_for(model);occ=np.empty(len(expected),dtype=int)
    for row,j in zip(rows[6:],match):
        reverse={v:k for k,v in labels[model['site_groups'][j]].items()}
        if row[3] not in reverse:raise ValueError('Species crossed sublattices')
        occ[j]=reverse[row[3]]
    bench.validate_counts(parsed_for(case),case,occ)
    return occ

def prepare(case):
    from ase import Atoms
    from ase.data import atomic_numbers
    from icet import ClusterSpace
    d=input_dir(case);d.mkdir(parents=True,exist_ok=True)
    done=d/'validation.json'
    if done.exists():return json.loads(done.read_text())
    start=time.perf_counter();model=model_for(case);ncell=int(np.prod(case['cell_dim']))
    primitive_cell=np.asarray(model['cell'])/np.asarray(case['cell_dim'])[:,None]
    coords=np.asarray(model['fractional_sites'])[::ncell]*np.asarray(case['cell_dim'])
    groups=[model['site_groups'][i*ncell] for i in range(len(coords))]
    allowed=model['basis_allowed_species'];labels=labels_for(model)
    maps=[dict(zip(g['species'],g['symmetry_labels'])) for g in model['sublattice_groups']]
    primitive=Atoms([maps[g][a[0]] for g,a in zip(groups,allowed)],scaled_positions=coords,cell=primitive_cell,pbc=True)
    cs=ClusterSpace(primitive,case['cutoffs'],[[maps[g][s] for s in a] for g,a in zip(groups,allowed)])
    rows=[' '.join(f'{x:.17g}' for x in r) for r in primitive_cell]+['1 0 0','0 1 0','0 0 1']
    for xyz,gid in zip(coords,groups):
        g=model['sublattice_groups'][gid];counts=dict(zip(g['species'],g['counts']));total=sum(g['counts'])
        ordered=sorted(g['species'],key=lambda s:atomic_numbers[model['species'][s]])
        rows.append(' '.join(f'{x:.17g}' for x in xyz)+' '+','.join(f'{labels[gid][s]}={counts[s]/total:.17g}' for s in ordered))
    (d/'rndstr.in').write_text('\n'.join(rows)+'\n')
    (d/'sqscell.out').write_text('1\n\n'+'\n'.join(' '.join(map(str,row)) for row in np.diag(case['cell_dim']))+'\n')
    cluster=[];targets=[];rows=[r for r in cs.as_list if r['order']>=2 and (case.get('target_mode')!='pair_sro' or r['order']==2)]
    assert len(rows)==len(model['features'])
    for row,f in zip(rows,model['features']):
        assert row['index']==f['index'] and row['order']==f['order']
        cart=np.asarray(cs.get_coordinates_of_representative_cluster(row['orbit_index']));frac=cart@np.linalg.inv(primitive_cell)
        diameter=float(np.linalg.norm(cart[:,None]-cart[None,:],axis=2).max())
        assert abs(diameter-f['diameter'])<1e-9
        cluster.extend([str(row['multiplicity']),f'{diameter:.17g}',str(row['order'])]);target=1.
        for point,k in zip(frac,row['multicomponent_vector']):
            delta=point-coords;delta-=np.rint(delta);dist=np.linalg.norm(delta@primitive_cell,axis=1);b=int(np.argmin(dist));assert dist[b]<1e-7
            gid=groups[b];g=model['sublattice_groups'][gid];ordered=sorted(g['species'],key=lambda s:atomic_numbers[model['species'][s]])
            m=len(ordered);assert m>=2
            cluster.append(' '.join(f'{x:.17g}' for x in point)+f' {m-2} {k}')
            function=-(np.cos if k%2==0 else np.sin)(2*np.pi*((k+2)//2)*np.arange(m)/m)
            counts=dict(zip(g['species'],g['counts']));target*=float(function@np.array([counts[s] for s in ordered])/sum(counts.values()))
        cluster.append('');targets.append(target)
    (d/'clusters.out').write_text('\n'.join(cluster)+'\n')
    if case.get('target_mode')=='reference':
        reference=np.loadtxt(ROOT/'inputs'/(case['id']+'.reference-occupations.txt'),dtype=int)
        write_structure(d/'reference.out',model,reference)
        targets=atat_vector(d,'reference.out').tolist()
    if case.get('target_mode') == 'explicit':
        targets = [f['target'] for f in model['features']]
    target_error=float(np.max(np.abs(np.asarray(targets)-[f['target'] for f in model['features']])));assert target_error<2e-10
    (d/'target-correlations.in').write_text('\n'.join(f'{f["target"]:.17g}' for f in model['features'])+'\n')
    parsed=parsed_for(case);rng=np.random.default_rng(20261018);errors=[]
    for probe in range(3):
        occ=np.empty(case['sites'],dtype=int)
        for gid,g in enumerate(model['sublattice_groups']):
            sites=np.flatnonzero(np.asarray(model['site_groups'])==gid)
            values=np.full(len(sites),g['species'][0]) if g['fixed'] else np.repeat(g['species'],g['counts'])
            rng.shuffle(values);occ[sites]=values
        bench.validate_counts(parsed,case,occ);write_structure(d/'probe.out',model,occ)
        assert np.array_equal(occupations(d/'probe.out',model,case),occ)
        ref=atat_vector(d,'probe.out');actual=correlations(parsed,occ)
        assert ref.shape==actual.shape,(ref.shape,actual.shape)
        error=float(np.max(np.abs(ref-actual)));assert error<2e-10,error;errors.append(error)
    # Directly audit both metric-selected native pilot outputs against ATAT.
    pilot_path=ROOT/'smoke'/case['id']/'result.json'
    pilot=json.loads(pilot_path.read_text()) if pilot_path.exists() else dict(candidates=[])
    selected={min(pilot['candidates'],key=lambda x:x['metrics'][metric])['structure'] for metric in ['max_error','weighted_l1']} if pilot['candidates'] else set()
    native_errors=[]
    for path in selected:
        occ=np.loadtxt(Path(path).parent/'FIGURE_OCCUPATIONS.txt',dtype=int);write_structure(d/'probe.out',model,occ)
        native_errors.append(float(np.max(np.abs(atat_vector(d,'probe.out')-correlations(parsed,occ)))))
    assert not native_errors or max(native_errors)<2e-10
    report=dict(case=case['id'],components=len(targets),orders=sorted(set(f['order'] for f in model['features'])),target_error=target_error,random_probe_errors=errors,native_pilot_atat_errors=native_errors,preparation_seconds=time.perf_counter()-start,input_hashes={n:digest(d/n) for n in ['rndstr.in','clusters.out','sqscell.out','target-correlations.in']},shared_species_labels=model['uses_auxiliary_labels'])
    save(done,report);return report

def run_mcsqs(case,profile,seed,out,budget=60,workers=4):
    out.mkdir(parents=True);source=input_dir(case);model=model_for(case);parsed=parsed_for(case)
    processes=[];logs=[];snapshots=[];errors=[]
    for rank in range(workers):
        d=out/f'worker-{rank}';d.mkdir()
        for name in ['rndstr.in','clusters.out','sqscell.out','target-correlations.in']:(d/name).symlink_to(source/name)
    start=time.perf_counter()
    try:
        for rank in range(workers):
            d=out/f'worker-{rank}';log=(d/'stdout.log').open('w');logs.append(log)
            cmd=[str(BIN/'mcsqs'),f'-n={case["sites"]}','-rc',f'-sd={seed+rank*100000}',f'-wd={profile["wd"]}',f'-wn={profile["wn"]}','-wr=1','-tol=0.001','-sig=17','-tcf=target-correlations.in']
            processes.append(subprocess.Popen(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        for deadline in [x for x in list(range(1,11))+list(range(15,int(budget),5)) if x<budget]+[budget-.02]:
            while time.perf_counter()-start<deadline:time.sleep(min(.05,max(0,deadline-(time.perf_counter()-start))))
            for rank,p in enumerate(processes):
                sourcefile=out/f'worker-{rank}'/'bestsqs.out'
                if sourcefile.exists():
                    dest=out/f'snapshot-{deadline}-{rank}.out';dest.write_bytes(sourcefile.read_bytes())
                    observed=time.perf_counter()-start
                    if observed<=budget:snapshots.append(dict(file=dest.name,available_seconds=observed,deadline=deadline,rank=rank))
    finally:
        codes=[p.poll() for p in processes]
        for p in processes:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for p in processes:
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()
    search_wall=time.perf_counter()-start;assessed=time.perf_counter();cache={};candidates=[];invalid=[]
    for rank,code in enumerate(codes):
        if code not in (None,0):errors.append(dict(worker=rank,returncode=code))
    for s in snapshots:
        path=out/s['file'];sha=digest(path)
        try:
            if sha not in cache:
                occ=occupations(path,model,case);vector=correlations(parsed,occ)
                cache[sha]=dict(metrics=metrics(parsed,vector,profile['wd'],profile['wn']),occupations=occ.tolist(),vector=vector.tolist())
            candidates.append(dict(available_seconds=s['available_seconds'],structure=str(path),metrics=cache[sha]['metrics'],sha256=sha))
        except (ValueError,IndexError,AssertionError) as exc:invalid.append(dict(**s,error=repr(exc)))
    # Check native mcsqs targets; correlations come from independent evaluation.
    target_errors=[]
    for rank in range(workers):
        file=out/f'worker-{rank}'/'bestcorr.out'
        if file.exists():
            rows=[list(map(float,line.split())) for line in file.read_text().splitlines() if line.strip() and not line.startswith('Objective')]
            table=np.array(rows)
            if table.shape!=(len(parsed['features']),5):errors.append(dict(worker=rank,error='bestcorr shape mismatch'));continue
            err=float(np.max(np.abs(table[:,3]-[f['target'] for f in parsed['features']])));target_errors.append(err)
            if err>2e-10:errors.append(dict(worker=rank,error=f'mcsqs target mismatch {err}'))
    audit=out/'audit';audit.mkdir()
    for name in ['rndstr.in','clusters.out']:(audit/name).symlink_to(source/name)
    discrepancy=[]
    if candidates:
        for sha in {min(candidates,key=lambda x:x['metrics'][metric])['sha256'] for metric in ['max_error','weighted_l1']}:
            write_structure(audit/'probe.out',model,cache[sha]['occupations']);ref=atat_vector(audit,'probe.out')
            delta=float(np.max(np.abs(ref-cache[sha]['vector'])));discrepancy.append(delta)
            if delta>2e-10:errors.append(dict(error=f'ATAT correlation mismatch {delta}'))
    else:errors.append(dict(error='No valid mcsqs output within budget'))
    passage={metric:[bench.first_passage(candidates,metric,t,budget) for t in thresholds] for metric,thresholds in [('max_error',[.05,.025,.01,.001]),('weighted_l1',[.01,.005,.002,.001])]}
    record=dict(case=case['id'],method='mcsqs',profile=profile['name'],seed=seed,ranks=workers,budget_seconds=budget,search_wall_seconds=search_wall,assessment_seconds=time.perf_counter()-assessed,errors=errors,invalid_snapshots=invalid,candidates=candidates,passage=passage,native_target_errors=target_errors,atat_discrepancies=discrepancy,snapshot_policy='One-second observations through 10 s, then every 5 s; first observed upper bounds, not exact internal attainment')
    save(out/'result.json',record);return record

def record_run(case,profile,seed,out,method,budget=60):
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    if out.exists():out.rename(out.with_name(out.name+'-interrupted-'+str(time.time_ns())))
    if method=='mcsqs':return run_mcsqs(case,profile,seed,out,budget=budget)
    bench.FIXTURES=ROOT/'inputs'
    args=SimpleNamespace(seconds=budget,ranks=4,binary=Path(os.environ['SCRAPS_BENCHMARK_BINARY']),mpirun=os.environ['MPIRUN'],checkpoint_seconds=.25)
    original_input=bench.input_text
    def kernel_input(c,p,s):
        text=original_input(c,p,s)
        if method=='scraps_sa':
            return text.replace('TOTALIter 2\n','TOTALIter 40\n')+'MC_KERNEL sa\nSA_T_INITIAL 2.0\nSA_T_FINAL 0.001\n'
        return text+'MC_KERNEL cuckoo\n'
    bench.input_text=kernel_input
    try:r=bench.run_trial(case,profile,seed,args,out)
    finally:bench.input_text=original_input
    r['method']=method
    if not r['candidates']:r['errors'].append(dict(error='No completed native output within budget'))
    model=model_for(case);parsed=parsed_for(case);audit=out/'atat-audit';audit.mkdir()
    for name in ['rndstr.in','clusters.out']:(audit/name).symlink_to(input_dir(case)/name)
    discrepancies=[]
    for path in ({min(r['candidates'],key=lambda x:x['metrics'][metric])['structure'] for metric in ['max_error','weighted_l1']} if r['candidates'] else set()):
        occ=np.loadtxt(Path(path).parent/'FIGURE_OCCUPATIONS.txt',dtype=int);write_structure(audit/'probe.out',model,occ)
        delta=float(np.max(np.abs(atat_vector(audit,'probe.out')-correlations(parsed,occ))));discrepancies.append(delta)
        if delta>2e-10:r['errors'].append(dict(error=f'ATAT correlation mismatch {delta}'))
    r['atat_discrepancies']=discrepancies
    save(out/'result.json',r);return r
