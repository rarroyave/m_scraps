import hashlib, html, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read
sys.path.insert(0, str(Path.home()/'src/tools/scraps-perpair/tools'))
from structure_analysis import analyze_structure, map_sites
from sro_stopping import SROBoundStop
from enumeration_symmetry import site_permutations
root = Path(__file__).resolve().parent
ensemble = json.loads((root/'random32/result.json').read_text())
results = json.loads((root/'random32-energies/summary.json').read_text())
manifest = json.loads((root/'random32-energies/manifest.json').read_text())
assert manifest['ensemble_sha256'] == hashlib.sha256((root/'random32/result.json').read_bytes()).hexdigest()
previous_manifest = json.loads((Path.home()/'runs/sqs-to-property-demo/max-ensemble-20261019/exact-energies/manifest.json').read_text())
for key in ('model_sha256', 'fmax', 'steps', 'device', 'dtype', 'relax_cell', 'optimizer'):
    assert manifest[key] == previous_manifest[key]
rows = results['structures']; assert len(rows) == 32
entries = {e['id']: e for e in ensemble['selected']}
seen = set()
permutations = None
for row in rows:
    path = root/'random32'/row['id']/'structure.extxyz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['input_sha256'] == entries[row['id']]['structure_sha256']
    atoms = read(path)
    if permutations is None:
        checker = SROBoundStop(atoms, cutoff=3.5, shells=2, sublattices=ensemble['sublattices'], targets=ensemble['targets'])
        permutations = site_permutations(atoms, checker.labels, checker.prepared, 1e-5)
    symbols = np.array(atoms.get_chemical_symbols())
    canonical = min(tuple(symbols[p]) for p in permutations)
    assert canonical not in seen
    seen.add(canonical)
    analysis = analyze_structure(atoms, cutoff=3.5, shells=2, sublattices=ensemble['sublattices'], sro_targets=ensemble['targets'])
    assert analysis['sro_comparison']['metrics']['max_error'] == 0
    for group, composition in analysis['composition'].items():
        assert composition['counts'] == ensemble['sublattices']['counts'][group]
    relaxed = root/'random32-energies'/row['id']/'relaxed.extxyz'
    assert hashlib.sha256(relaxed.read_bytes()).hexdigest() == row['relaxed_sha256']
    final = read(relaxed)
    assert final.get_chemical_symbols() == atoms.get_chemical_symbols()
    np.testing.assert_allclose(final.cell, atoms.cell, atol=1e-10)
    try:
        mapping, _ = map_sites(final, atoms, tolerance=.75)
        mapping_valid = bool(np.array_equal(mapping, np.arange(len(atoms))))
    except ValueError:
        mapping_valid = False
    assert row['site_mapping_valid'] == mapping_valid
    assert row['force_converged'] == (row['final_max_force_eV_A'] <= manifest['fmax'])
valid = [r for r in rows if r['force_converged'] and r['site_mapping_valid']]
series = {'Unrelaxed': np.array([r['unrelaxed_eV_per_atom'] for r in valid]),
          'Fixed-cell relaxed': np.array([r['relaxed_eV_per_atom'] for r in valid])}
stats = {}
fig, axes = plt.subplots(1,2,figsize=(10,4), constrained_layout=True)
for ax, (name, values) in zip(axes, series.items()):
    centered = 1000*(values-values.mean())
    stats[name] = dict(n=len(values), mean_eV_per_atom=float(values.mean()),
        population_std_meV_per_atom=float(1000*values.std()), range_meV_per_atom=float(1000*np.ptp(values)),
        min_eV_per_atom=float(values.min()), median_eV_per_atom=float(np.median(values)), max_eV_per_atom=float(values.max()),
        q10_eV_per_atom=float(np.quantile(values,.1)),q90_eV_per_atom=float(np.quantile(values,.9)))
    ax.hist(centered,bins=8,color='#267c98',alpha=.8,edgecolor='white')
    ax.plot(centered,np.zeros(len(values))-.12,'|',color='#272727',clip_on=False)
    ax.axvline(0, color='#bb552e',linestyle='--',linewidth=1)
    ax.set(title=name,xlabel='Energy relative to sample mean (meV/atom)',ylabel='Structure count')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle(f'Random MAX-phase ensemble: {len(valid)} accepted 32-atom structures')
fig.savefig(root/'energy-distribution.png',dpi=180)
summary = dict(selection='random without replacement from 128 collected symmetry orbits',
    n_selected=32,n_accepted=len(valid), excluded=[r['id'] for r in rows if r not in valid],
    targeted_sro_error=0,stats=stats,model_sha256=manifest['model_sha256'])
(root/'distribution-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
tr = ''.join(f"<tr><td>{name}</td><td>{s['n']}</td><td>{s['mean_eV_per_atom']:.6f}</td><td>{s['population_std_meV_per_atom']:.2f}</td><td>{s['range_meV_per_atom']:.2f}</td></tr>" for name,s in stats.items())
text = f'''<!doctype html><html><meta charset="utf-8"><title>Random MAX-phase energy distribution</title>
<style>body{{font:17px/1.5 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#24313c}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ccd4db}}img{{width:100%}}code{{overflow-wrap:anywhere}}.note{{padding:18px;background:#eff5f6}}</style>
<h1>Random MAX-phase energy distribution</h1><p>32 distinct configurations of (Ti₀.₅Zr₀.₅)₂(Al₀.₅Si₀.₅)C; 32 atoms per configuration. Random selection seed 17 from 128 symmetry-distinct candidates generated with SA seed 42. First-shell Ti–Zr, Al–Si and Ti–Si targeted SRO errors are exactly zero.</p>
<p>Fixed 2 × 2 × 1 parent cell, lengths 6.10, 6.10, 13.60 Å and basal angle 120°. MACE via MaterialsFramework; CPU float64, FIRE atomic relaxation, force tolerance 0.03 eV/Å, at most 300 steps. Accepted {len(valid)}/32 with force convergence and retained parent-site mapping. Both distributions below use the same accepted structures.</p>
<table><tr><th>Stage</th><th>N</th><th>Mean (eV/atom)</th><th>Population SD (meV/atom)</th><th>Range (meV/atom)</th></tr>{tr}</table>
<img src="energy-distribution.png" alt="Histograms of unrelaxed and relaxed energies centered on their respective means">
<p class="note">Pilot distribution conditional on a bounded SA-generated candidate pool. This is not uniform sampling of all labeled occupations, a Boltzmann distribution, or a thermodynamic uncertainty estimate. The 32 selected structures are distinct; the sampling fraction is 25% of the 128-member pool. Histogram bars depend on bin choice; individual energies are marked along the axis. MACE predictions use a prescribed common lattice, not a validated alloy equilibrium geometry. Initial occupation-based SRO equality does not establish equal distance-based SRO after relaxation.</p>
<p>Excluded IDs: {html.escape(str(summary['excluded']))}. Model SHA-256: <code>{manifest['model_sha256']}</code>.</p>
<p><a href="random32-energies/energies.csv">Raw energies CSV</a> · <a href="distribution-summary.json">Summary JSON</a> · <a href="random32/result.json">Ensemble provenance</a></p></html>'''
(root/'report.html').write_text(text)
print(json.dumps(summary,indent=2))
