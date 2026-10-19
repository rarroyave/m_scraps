"""icet 4.0 annealer with an explicitly matched ATAT objective calculator."""
import argparse,hashlib,json,math,os,random,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import matched_mcsqs as m
from campaign import save
from atat_reference import correlations,metrics
import benchmark_atat_reference as bench
ROOT=Path(os.environ['SCRAPS_BENCHMARK_ROOT'])
m.ROOT=ROOT;m.MATCH=ROOT/'matched-atat';m.BIN=m.MATCH/'bin'
EXT=ROOT/'icet-extension'
METHOD='icet_atat'

class AtatScore:
    def __init__(self,features,profile):
        self.target=np.array([f['target'] for f in features]);self.d=np.array([f['diameter'] for f in features]);self.o=np.array([f['order'] for f in features]);self.d0=self.d.min()
        self.wn=profile['wn'];self.wr=profile.get('wr',1);self.tol=profile.get('tol',.001)
        self.w=np.exp(-profile['wd']*self.d/self.d0+(self.o-2)*np.log(self.wn))
        self.orders=[self.o==n for n in range(2,int(self.o.max())+1)]
    def __call__(self,vector):
        err=np.abs(vector-self.target);ranges=[]
        for mask in self.orders:
            radius=self.d[mask].max()+self.d0 if mask.any() else self.d0
            bad=mask&(err>self.tol)
            if bad.any():radius=min(radius,self.d[bad].min())
            if ranges:radius=min(radius,ranges[-1])
            ranges.append(radius)
        mask=self.d>=min(ranges)-.001;numerator=float(self.w[mask]@err[mask])
        if numerator<.001:return -float(np.finfo(np.float32).max)
        return numerator/float(self.w[mask].sum())-sum(self.wr*self.wn**i*r/self.d0 for i,r in enumerate(ranges))

def setup(case,seed):
    from ase import Atoms
    from ase.data import atomic_numbers
    from icet import ClusterSpace
    model=m.model_for(case);ncell=int(np.prod(case['cell_dim']));gids=model['site_groups'];groups=model['sublattice_groups']
    primitive_cell=np.asarray(model['cell'])/np.asarray(case['cell_dim'])[:,None]
    coords=np.asarray(model['fractional_sites'])[::ncell]*np.asarray(case['cell_dim'])
    bg=[gids[i*ncell] for i in range(len(coords))];allowed=model['basis_allowed_species']
    maps=[dict(zip(g['species'],g['symmetry_labels'])) for g in groups]
    primitive=Atoms([maps[g][a[0]] for g,a in zip(bg,allowed)],scaled_positions=coords,cell=primitive_cell,pbc=True)
    cs=ClusterSpace(primitive,case['cutoffs'],[[maps[g][s] for s in a] for g,a in zip(bg,allowed)])
    indices=np.array([f['index'] for f in model['features']],dtype=int)
    rows={r['index']:r for r in cs.as_list}
    assert all(rows[f['index']]['order']==f['order'] for f in model['features'])
    rng=np.random.default_rng(seed)
    def new_structure():
        occ=np.empty(case['sites'],dtype=int)
        for gid,g in enumerate(groups):
            sites=np.flatnonzero(np.asarray(gids)==gid)
            occ[sites]=np.full(len(sites),g['species'][0]) if g['fixed'] else rng.permutation(np.repeat(g['species'],g['counts']))
        return Atoms([maps[g][int(s)] for g,s in zip(gids,occ)],scaled_positions=model['fractional_sites'],cell=model['cell'],pbc=True)
    inverse=[{atomic_numbers[v]:k for k,v in mapping.items()} for mapping in maps]
    def occupations(atoms):return np.array([inverse[g][int(z)] for g,z in zip(gids,atoms.numbers)])
    # Independently check the icet -> physical species and figure-index mapping.
    sample=new_structure();occ=occupations(sample);parsed=m.parsed_for(case)
    bench.validate_counts(parsed,case,occ)
    delta=float(np.max(np.abs(cs.get_cluster_vector(sample)[indices]-correlations(parsed,occ))))
    assert delta<2e-10,delta
    return cs,indices,model,new_structure,occupations,delta

def worker(case,profile,seed,directory):
    import icet
    from mchammer.calculators import TargetVectorCalculator
    from mchammer.ensembles import TargetClusterVectorAnnealing
    assert icet.__version__=='4.0',icet.__version__
    cs,indices,model,new_structure,occupations,delta=setup(case,seed)
    score=AtatScore(model['features'],profile)
    class Calculator(TargetVectorCalculator):
        def calculate_total(self,occupation):
            self._structure.set_atomic_numbers(occupation)
            return score(self.cluster_space.get_cluster_vector(self._structure)[indices])
    best=float('inf');last=0;restarts=0;trials=0
    def emit(ensemble,force=False):
        nonlocal best,last
        now=time.monotonic()
        if not force and now-last<.25:return
        last=now
        if ensemble.best_score>best:return
        best=ensemble.best_score;atoms=ensemble.best_structure;vector=cs.get_cluster_vector(atoms)[indices];occ=occupations(atoms)
        save(directory/'best.json',dict(occupations=occ.tolist(),vector=vector.tolist(),atat_score=score(vector),restart=restarts,trial_steps=trials+ensemble.total_trials,icet_mapping_error=delta))
    class RecordingAnnealer(TargetClusterVectorAnnealing):
        # Pinned to icet 4.0: the hook only records, preserving the native
        # proposal, Metropolis acceptance, cooling and best-state policy.
        def _do_trial_step(self):
            super()._do_trial_step();emit(self)
    while True:
        atoms=new_structure();calc=Calculator(atoms,cs,np.zeros(len(cs)),optimality_weight=None)
        annealer=RecordingAnnealer([atoms],[calc],T_start=5.0,T_stop=.001,random_seed=seed+restarts)
        emit(annealer,force=True);annealer.generate_structure(number_of_trial_steps=3000);emit(annealer,force=True)
        trials+=annealer.total_trials;restarts+=1

def trial(case,profile,seed,out,budget=60,workers=4,stop_check=None):
    out.mkdir(parents=True);logs=[];processes=[];snapshots=[];errors=[];attained=False
    for rank in range(workers):(out/f'worker-{rank}').mkdir()
    start=time.perf_counter()
    try:
        for rank in range(workers):
            d=out/f'worker-{rank}';log=(d/'stdout.log').open('w');logs.append(log)
            cmd=[sys.executable,str(Path(__file__)),'worker','--case',json.dumps(case),'--profile',json.dumps(profile),'--seed',str(seed+rank*100000),'--directory',str(d)]
            processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')))
        deadlines=[x for x in list(range(1,11))+list(range(15,int(budget),5)) if x<budget]+[budget-.02]
        for deadline in deadlines:
            while time.perf_counter()-start<deadline:time.sleep(min(.05,max(0,deadline-(time.perf_counter()-start))))
            for rank in range(workers):
                src=out/f'worker-{rank}'/'best.json'
                if src.exists():
                    dst=out/f'snapshot-{deadline}-{rank}.json';dst.write_bytes(src.read_bytes())
                    observed=time.perf_counter()-start
                    if observed<=budget:snapshots.append(dict(file=dst.name,available_seconds=observed))
                    if observed<=budget and stop_check is not None and stop_check(np.asarray(json.loads(dst.read_text())["occupations"],dtype=int)):
                        attained=True
                        break
            if attained:
                break
    finally:
        codes=[p.poll() for p in processes]
        for p in processes:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for p in processes:
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()
    search_wall=time.perf_counter()-start;assessed=time.perf_counter();model=m.model_for(case);parsed=m.parsed_for(case);score=AtatScore(model['features'],profile);candidates=[];cache={}
    for rank,code in enumerate(codes):
        if code not in (None,0):errors.append(dict(worker=rank,returncode=code))
    for s in snapshots:
        data=json.loads((out/s['file']).read_text());occ=np.asarray(data['occupations'],int);key=hashlib.sha256(occ.tobytes()).hexdigest()
        if key not in cache:
            bench.validate_counts(parsed,case,occ);vector=correlations(parsed,occ)
            discrepancy=float(np.max(np.abs(vector-data['vector'])))
            if discrepancy>2e-10:raise RuntimeError(f'icet vector mismatch {discrepancy}')
            if not np.isclose(score(vector),data['atat_score'],rtol=1e-12,atol=1e-10):raise RuntimeError('icet ATAT-score mismatch')
            d=out/'structures'/key;d.mkdir(parents=True);np.savetxt(d/'FIGURE_OCCUPATIONS.txt',occ,fmt='%d');m.write_structure(d/'structure.out',model,occ)
            cache[key]=dict(metrics=metrics(parsed,vector,profile['wd'],profile['wn']),structure=str(d/'structure.out'),vector=vector.tolist(),occupations=occ.tolist(),discrepancy=discrepancy)
        row=cache[key];candidates.append(dict(available_seconds=s['available_seconds'],structure=row['structure'],metrics=dict(row['metrics']),sha256=key))
    audit=out/'audit';audit.mkdir()
    for name in ['rndstr.in','clusters.out']:(audit/name).symlink_to(m.input_dir(case)/name)
    errors_atat=[]
    for key in ({min(candidates,key=lambda x:x['metrics'][metric])['sha256'] for metric in ['max_error','weighted_l1']} if candidates else set()):
        row=cache[key];m.write_structure(audit/'probe.out',model,row['occupations']);delta=float(np.max(np.abs(m.atat_vector(audit,'probe.out')-row['vector'])));errors_atat.append(delta)
        if delta>2e-10:errors.append(dict(error=f'ATAT correlation mismatch {delta}'))
    if not candidates:errors.append(dict(error='No completed icet output within budget'))
    passage={metric:[bench.first_passage(candidates,metric,t,budget) for t in thresholds] for metric,thresholds in [('max_error',[.05,.025,.01,.001]),('weighted_l1',[.01,.005,.002,.001])]}
    record=dict(case=case['id'],method=METHOD,profile=profile['name'],seed=seed,ranks=workers,budget_seconds=budget,errors=errors,candidates=candidates,passage=passage,search_wall_seconds=search_wall,assessment_seconds=time.perf_counter()-assessed,atat_discrepancies=errors_atat,implementation='icet 4.0 TargetClusterVectorAnnealing with ATAT-compatible TargetVectorCalculator subclass; configurable independent processes; 3000-step schedules 5 to .001; no stock-icet-score claim')
    if stop_check is not None:
        record['stop_reason']='bound_attained' if attained else 'time_limit'
    save(out/'result.json',record);return record

def run_record(case,profile,seed,out,budget=60):
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    if out.exists():out.rename(out.with_name(out.name+'-interrupted-'+str(time.time_ns())))
    return trial(case,profile,seed,out,budget)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['worker'])
    parser.add_argument('--case');parser.add_argument('--profile')
    parser.add_argument('--seed',type=int);parser.add_argument('--directory',type=Path)
    a=parser.parse_args()
    worker(json.loads(a.case),json.loads(a.profile),a.seed,a.directory)
