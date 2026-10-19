"""Feasible nonzero pair, triplet and quadruplet targets from fixed-composition reference cells."""
import copy,gzip,json,math
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from atat_reference import figure_model,correlations

def write_figures(path,parsed):
    p=parsed
    rows=[f'SCRAPS_FIGURES {p["version"]}',f'{p["sites"]} {p["species"]} {len(p["basis"])} {len(p["features"])}']
    for block in [p['cell'],p['positions'],p['basis']]:rows.extend(' '.join(f'{x:.17g}' for x in r) for r in block)
    if p['version']>=2:rows.extend(str(len(a))+' '+' '.join(map(str,a)) for a in p['allowed'])
    if p['version']==3:rows.extend(str(g)+' '+' '.join(map(str,c)) for g,c in zip(p['groups'],p['local_counts']))
    for f in p['features']:
        rows.append(f'{f["order"]} {f["diameter"]:.17g} {f["target"]:.17g} {len(f["terms"])}')
        for coef,factors in f['terms']:rows.append(f'{coef:.17g} '+' '.join(f'{s} {b}' for s,b in factors))
    path.write_text('\n'.join(rows)+'\n')

def make_targets(root,base):
    directory=root/'inputs';source=directory/(base['id']+'.figures')
    metadata=json.loads(source.with_suffix('.figures.json').read_text())
    original=figure_model(source);original['version']=metadata['version']
    groups=metadata['sublattice_groups'];gids=np.asarray(metadata['site_groups'])
    atoms=Atoms(['H']*base['sites'],scaled_positions=original['positions'],cell=original['cell'],pbc=True)
    ii,jj,dd=neighbor_list('ijd',atoms,base['cutoffs'][0]+1e-6)
    result=[]
    for style,sign in [('ordering',-1)]:
        case=copy.deepcopy(base);case.update(id=base['id']+'_sro_'+style,base_id=base['id'],target_mode='reference',target_style=style)
        dest=directory/(case['id']+'.figures');done=directory/(case['id']+'.sro-target.json')
        if done.exists():result.append(case);continue
        rng=np.random.default_rng(20261019+(style=='clustering'));occ=np.empty(base['sites'],int)
        for gid,g in enumerate(groups):
            sites=np.flatnonzero(gids==gid)
            if g['fixed']:occ[sites]=g['species'][0];continue
            occ[sites]=rng.permutation(np.repeat(g['species'],g['counts']))
            mask=(gids[ii]==gid)&(gids[jj]==gid)&(ii!=jj)
            nearest=float(dd[mask].min());edges={tuple(sorted((int(i),int(j)))) for i,j,d in zip(ii[mask],jj[mask],dd[mask]) if abs(d-nearest)<1e-6}
            adjacent={int(i):set() for i in sites}
            for edge in edges:
                for i in edge:adjacent[i].add(edge)
            for step in range(8*len(sites)):
                a,b=rng.choice(sites,2,replace=False)
                if occ[a]==occ[b]:continue
                affected=adjacent[int(a)]|adjacent[int(b)]
                before=sum(occ[i]==occ[j] for i,j in affected)
                occ[a],occ[b]=occ[b],occ[a]
                delta=sign*(sum(occ[i]==occ[j] for i,j in affected)-before)
                temperature=.5*(.05/.5)**(step/max(1,8*len(sites)-1))
                if delta<0 and rng.random()>=math.exp(delta/temperature):occ[a],occ[b]=occ[b],occ[a]
            # Soften the reference without changing either local composition.
            for _ in range(max(1,len(sites)//10)):
                a,b=rng.choice(sites,2,replace=False);occ[a],occ[b]=occ[b],occ[a]
        parsed=copy.deepcopy(original);indices=list(range(len(parsed['features'])))
        vector=correlations(parsed,occ);parsed['features']=[parsed['features'][i] for i in indices]
        target_meta=copy.deepcopy(metadata);target_meta['features']=[target_meta['features'][i] for i in indices];target_meta['cutoffs']=case['cutoffs'];target_meta['target_convention']='Exact pair/triplet/quadruplet correlations of a fixed-composition reference configuration';target_meta['target_style']=style
        for f,g,i in zip(parsed['features'],target_meta['features'],indices):f['target']=g['target']=float(vector[i])
        departure=max(abs(vector[i]-original['features'][i]['target']) for i in indices)
        if departure<.01:raise RuntimeError('Target is insufficiently different from random correlations')
        write_figures(dest,parsed)
        dest.with_suffix('.figures.json').write_text(json.dumps(target_meta)+'\n')
        dest.with_suffix('.figures.gz').write_bytes(gzip.compress(dest.read_bytes(),mtime=0))
        (directory/(case['id']+'.poscar')).write_bytes((directory/(base['id']+'.poscar')).read_bytes())
        np.savetxt(directory/(case['id']+'.reference-occupations.txt'),occ,fmt='%d')
        reloaded=figure_model(dest);assert np.max(np.abs(correlations(reloaded,occ)-[f['target'] for f in reloaded['features']]))<2e-12
        alpha=[]
        for ga,gb,distance in sorted({(int(gids[i]),int(gids[j]),round(float(d),7)) for i,j,d in zip(ii,jj,dd) if not groups[gids[i]]['fixed'] and not groups[gids[j]]['fixed']}):
            mask=(gids[ii]==ga)&(gids[jj]==gb)&(np.abs(dd-distance)<1e-6)
            a=ii[mask];b=jj[mask];target_group=groups[gb];concentration=dict(zip(target_group['species'],np.asarray(target_group['counts'])/sum(target_group['counts'])))
            for sa in groups[ga]['species']:
                central=occ[a]==sa
                for sb in target_group['species']:
                    probability=float(np.mean(occ[b[central]]==sb))
                    alpha.append(dict(from_group=ga,to_group=gb,distance=distance,central=metadata['species'][sa],neighbor=metadata['species'][sb],alpha=1-probability/concentration[sb]))
        done.write_text(json.dumps(dict(case=case,construction='Seeded within-sublattice nearest-neighbor like-pair bias plus composition-preserving swaps; reference is never a search starting configuration',target_convention='All included pair/triplet/quadruplet correlations from one explicitly realizable fixed-composition reference',max_pair_departure_from_random=float(departure),warren_cowley_targets=alpha),indent=2)+'\n')
        result.append(case)
    return result
