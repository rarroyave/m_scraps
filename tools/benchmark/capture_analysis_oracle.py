#!/usr/bin/env python3
"""Opt-in ATAT 3.52 oracle capture. Requires icet 4.0 only to define figures.

This generator never calls structure_analysis or copies its computed metrics.
The normal regression tests only read the resulting frozen data.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from ase.spacegroup import crystal

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'submission'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--atat-bin', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['SCRAPS_BENCHMARK_ROOT'] = str(root)
    import matched_mcsqs as adapter
    from higher_order import export_figures
    adapter.BIN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.atat_bin / 'corrdump', adapter.BIN / 'corrdump')
    (root / 'inputs').mkdir(exist_ok=True)
    destination = root / 'fixtures'
    destination.mkdir(exist_ok=True)
    prototypes = [
        ('diboride', ['Ti','B'], [(0,0,0),(1/3,2/3,.5)], 191,
         [3.1,3.1,3.3,90,90,120], [['Ti','Zr'],['B']], 4.6),
        ('max211', ['Ti','Al','C'], [(1/3,2/3,.08),(1/3,2/3,.75),(0,0,0)], 194,
         [3.05,3.05,13.6,90,90,120], [['Ti','Zr'],['Al','Si'],['C']], 3.5),
        ('heusler', ['Co','Mn','Al'], [(.25,.25,.25),(0,0,0),(.5,.5,.5)], 225,
         [5.95,5.95,5.95,90,90,90], [['Co','Ni'],['Mn','Fe'],['Al']], 4.3),
    ]
    for name, symbols, basis, group, cellpar, alphabets, cutoff in prototypes:
        atoms = crystal(symbols, basis, spacegroup=group, cellpar=cellpar)
        species = [s for alphabet in alphabets for s in alphabet]
        counts = [0] * len(species)
        sublattices, spectators = [], []
        kinds = atoms.arrays['spacegroup_kinds']
        for gid, alphabet in enumerate(alphabets):
            indices = np.flatnonzero(kinds == gid).tolist()
            ids = [species.index(s) for s in alphabet]
            local = [8 * len(indices) // len(ids)] * len(ids)
            for s, n in zip(ids, local): counts[s] = n
            if len(ids) == 1:
                spectators.extend((i, ids[0]) for i in indices)
            else:
                sublattices.append((gid, indices, ids, local))
        path = root / 'inputs' / f'{name}.figures'
        model = export_figures(path, atoms, species, counts, [2]*3, [cutoff]*3,
                               sublattices=sublattices, spectator_basis_species=spectators)
        case = dict(id=name, species=species, counts=counts, sites=sum(counts),
                    cell_dim=[2]*3, cutoffs=[cutoff]*3, constraints=[])
        validation = adapter.prepare(case)
        directory = adapter.input_dir(case)
        configurations = []
        for seed in [42, 137]:
            rng = np.random.default_rng(seed)
            occupations = np.empty(sum(counts), dtype=int)
            for gid, info in enumerate(model['sublattice_groups']):
                indices = np.flatnonzero(np.array(model['site_groups']) == gid)
                values = np.repeat(info['species'], info['counts'])
                rng.shuffle(values)
                occupations[indices] = values
            structure = f'configuration-{seed}.out'
            adapter.write_structure(directory / structure, model, occupations)
            command = [str(adapter.BIN/'corrdump'), '-l=rndstr.in', '-ro', '-c', '-noe', '-nop', f'-s={structure}', '-sig=17']
            process = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=True)
            vector = np.fromstring(process.stdout, sep=' ')
            assert len(vector) == len(model['features'])
            # Pair-shell spin products from ATAT orbit correlations, combined
            # using orbit multiplicities. Each disordered orbit is binary 50/50.
            pairs = defaultdict(list)
            for feature_index, (feature, value) in enumerate(zip(model['features'], vector)):
                if feature['order'] != 2: continue
                factors = feature['terms'][0]['factors']
                gids = tuple(sorted(model['site_groups'][site] for site, _ in factors))
                key = (*gids, round(feature['diameter'], 7))
                pairs[key].append((feature['multiplicity'], float(value), feature_index))
            sro = []
            for (g0, g1, distance), terms in sorted(pairs.items()):
                spin = sum(w*v for w,v,_ in terms) / sum(w for w,_,_ in terms)
                for source, target in [(g0,g1)] + ([(g1,g0)] if g0 != g1 else []):
                    a = model['sublattice_groups'][source]
                    b = model['sublattice_groups'][target]
                    for si in a['species']:
                        for sj in b['species']:
                            # Values in the exported basis use ATAT's -cos(pi*s).
                            bi = next(row[si] for row in model['basis'] if abs(row[si]) > .5)
                            bj = next(row[sj] for row in model['basis'] if abs(row[sj]) > .5)
                            sro.append(dict(source=str(source), destination=str(target),
                                            center_species=species[si], neighbor_species=species[sj],
                                            distance=distance, alpha=-bi*bj*spin, spin_factor=-bi*bj,
                                            feature_indices=[i for _,_,i in terms],
                                            weights=[w/sum(t[0] for t in terms) for w,_,_ in terms]))
            targets = np.array([f['target'] for f in model['features']])
            residual = vector-targets
            metrics = dict(max_error=float(max(abs(residual))), rms_error=float(np.sqrt(np.mean(residual**2))),
                           mean_absolute_error=float(np.mean(abs(residual))), weighted_l1=float(np.mean(abs(residual))))
            configurations.append(dict(seed=seed, occupations=occupations.tolist(), correlations=vector.tolist(),
                                       metrics=metrics, sro=sro))
            (destination / f'{name}-{seed}.corrdump.txt').write_text(process.stdout)
            shutil.copy2(directory / structure, destination / f'{name}-{seed}.str.out')
        save(destination / f'{name}.json', dict(case=case, configurations=configurations,
            validation=validation, sro_conversion='binary equiatomic: alpha_ab = -gamma(a)*gamma(b)*<gamma_i gamma_j>; ATAT pair orbits combined by multiplicity',
            sro_scope='disordered-disordered pairs within figure cutoff; fixed-site pairs tested analytically'))
        (destination / f'{name}.figures.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        for filename in ['rndstr.in', 'clusters.out']:
            shutil.copy2(directory / filename, destination / f'{name}.{filename}')
        print(name, len(vector), sorted(Counter(f['order'] for f in model['features']).items()), flush=True)
    save(destination / 'manifest.json', dict(schema_version=1, atat_version='3.52',
        cases=[p[0] for p in prototypes], configurations_per_case=2,
        command=['corrdump','-l=rndstr.in','-ro','-c','-noe','-nop','-s=configuration-SEED.out','-sig=17'],
        corrdump_sha256=hashlib.sha256((adapter.BIN/'corrdump').read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        atat_source_sha256='8536fb078375b30096df780e02f4e0c97972f5ebaad9a84a94f22e00f50c7516',
        metric_weights=dict(wd=0, wn=1),
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.name != 'manifest.json'}))


if __name__ == '__main__':
    main()
