#!/usr/bin/env python3
"""Build a standalone HTML report from the frozen submission benchmark."""
import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tools/fixtures/submission_benchmark'
LABELS = {'scraps_cuckoo': 'Cuckoo', 'scraps_sa': 'Simulated annealing',
          'mcsqs': 'ATAT mcsqs', 'icet_atat': 'icet · matched objective'}


def render():
    paths = {name: FIXTURES / (name + '.json')
             for name in ('baseline', 'protocol', 'environment', 'manifest')}
    raw = {name: path.read_text() for name, path in paths.items()}
    data = json.loads(raw['baseline'])
    protocol = json.loads(raw['protocol'])
    environment = json.loads(raw['environment'])
    assert len(data) == protocol['trials'] == 40
    assert all(r['outcome'] == 'validated_output' for r in data)
    assert len({(r['case'], r['method'], r['seed']) for r in data}) == len(data)
    groups = {(r['case'], r['seed']) for r in data}
    assert all({r['method'] for r in data if (r['case'], r['seed']) == group}
               == set(LABELS) for group in groups)
    for row in data:
        for passages in row['passage'].values():
            for passage in passages:
                assert passage['success'] == (passage['seconds'] is not None)
                if passage['success']:
                    assert 0 <= passage['seconds'] <= row['budget_seconds']
                else:
                    assert passage['right_censored']
    summary = []
    for method, label in LABELS.items():
        rows = [r for r in data if r['method'] == method]
        counts = [sum(next(p for p in r['passage']['max_error']
                           if p['threshold'] == t)['success'] for r in rows)
                  for t in (.05, .025, .01)]
        summary.append('<tr><th scope="row">' + label + '</th>' + ''.join(
            f'<td>{n}/{len(rows)}</td>' for n in counts) + '</tr>')
    svg = (ROOT / 'docs/submission_benchmark.svg').read_text()
    svg = svg[svg.index('<svg'):]
    provenance = '\n'.join(f'{path.name}  {hashlib.sha256(path.read_bytes()).hexdigest()}'
                           for path in [*paths.values(), ROOT / 'docs/submission_benchmark.svg'])
    package = json.dumps({'rows': data, 'labels': LABELS}, ensure_ascii=False).replace('<', '\\u003c')
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>m_scraps — Reproducible benchmark report</title>
<style>
:root{color-scheme:light;--ink:#172b3a;--muted:#506575;--line:#d9e2e8;--blue:#0072b2;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#eef3f6;color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}main{max-width:1200px;margin:auto;padding:42px 32px 60px}header{border-top:6px solid var(--blue);padding:28px 0}h1{font-size:clamp(30px,4vw,48px);line-height:1.14;letter-spacing:-1.5px;margin:12px 0}h2{font-size:25px;margin:0 0 16px}h3{font-size:19px}p{max-width:94ch}.eyebrow{font-size:12px;letter-spacing:2px;font-weight:750;color:var(--blue);text-transform:uppercase}.lead{font-size:19px;color:var(--muted)}a{color:#00639c}section{background:var(--paper);padding:28px;margin:22px 0;border:1px solid var(--line);border-radius:12px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{padding:18px;background:#e2edf4;border-radius:9px}.stat strong{font-size:30px;display:block}.stat span{color:var(--muted);font-size:14px}.note{padding:16px 20px;background:#fff4da;border-left:4px solid #b57700;border-radius:4px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:11px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#edf3f7;font-weight:650;white-space:normal}td:first-child,td:nth-child(2){white-space:normal;min-width:125px}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#f1f7fb}caption{text-align:left;color:var(--muted);padding:0 0 12px}figure{margin:0}figure svg{width:100%;height:auto}figcaption,.small{font-size:13px;color:var(--muted)}.filters{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}label{font-size:13px;font-weight:650}select{display:block;padding:9px;margin-top:5px;border:1px solid #a9bdca;border-radius:5px;background:white;color:var(--ink);font:inherit}button{cursor:pointer;background:#006da8;color:white;padding:10px 16px;border:0;border-radius:6px;font:inherit}.ok{color:#086844;font-weight:650}.censored{color:#805400}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}dt{font-weight:700;margin-top:14px}dd{margin:3px 0 12px}code{background:#edf3f7;padding:2px 5px;border-radius:3px}pre{overflow:auto;background:#edf3f7;padding:14px;font-size:12px}footer{color:var(--muted);font-size:13px}details summary{cursor:pointer;font-weight:650}noscript{display:block;background:#fff4da;padding:18px}
@media(max-width:700px){main{padding:20px 14px}section{padding:18px}.stats{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}h1{letter-spacing:-.8px}}
@media print{body{background:white}main{padding:0;max-width:none}section{break-inside:avoid;border:0;padding:12px 0}.filters,button{display:none}table{font-size:9px}th,td{padding:5px}.scroll{overflow:visible}header{padding:10px 0}a{color:inherit}}
</style></head><body><main>
<header><div class="eyebrow">m_scraps / archived benchmark</div>
<h1>m_scraps<br>Benchmark results</h1>
<p class="lead">Single- and multi-sublattice structure generation assessed against the same decorated pair, triplet, and quadruplet correlations.</p>
<p><a href="https://github.com/rarroyave/m_scraps">Repository</a> · <a href="https://doi.org/10.5281/zenodo.23475896">Frozen software release v0.5.1</a> · <a href="#trials">Explore individual trials</a></p>
<div class="stats"><div class="stat"><strong>40</strong><span>archived trials · 10 per method</span></div><div class="stat"><strong>8</strong><span>structure / size / target combinations</span></div><div class="stat"><strong>30 s</strong><span>search budget · four cores per trial</span></div><div class="stat"><strong>40/40</strong><span>trials with ATAT-validated output</span></div></div>
</header>
<aside class="note"><strong>Scope:</strong> a small reproducible screening campaign, not a universal solver ranking. This report presents archived measurements; opening or regenerating it runs no optimization. The icet arm uses a custom matched ATAT objective, not the stock icet objective.</aside>
<section><h2>What the archived trials show</h2>
<p>At maximum absolute correlation error ≤0.01, cuckoo search reached the threshold in 5 of 10 trials; each other method reached it in 2 of 10. These are descriptive counts for the tested problems and seeds, not estimates of general success probabilities.</p>
<div class="scroll"><table><caption>Threshold attainment within 30 seconds · maximum absolute correlation error</caption><thead><tr><th>Method</th><th>≤0.05</th><th>≤0.025</th><th>≤0.01</th></tr></thead><tbody>__SUMMARY__</tbody></table></div>
<p class="small">No average attainment time is reported: trials that miss a threshold are censored, and averaging only successful trials would bias a method comparison.</p>
<figure>__SVG__<figcaption>Each marker represents a trial. The error axis is logarithmic. Crosses at 30 seconds mean “threshold not reached,” not a successful attainment at 30 seconds. Repeated seeds are shown individually. Figure reproduced from the archived results.</figcaption></figure>
</section>
<section><h2>Campaign design</h2><div class="grid"><div><dl>
<dt>Single disordered sublattice</dt><dd>BCC Nb–Mo–Ta, 128 and 432 sites.</dd>
<dt>Two disordered sublattices</dt><dd>Rocksalt (Ti,Zr,Hf)(C,N), 128 and 432 sites, with separate metal and anion composition constraints.</dd>
<dt>Targets</dt><dd>Independent-site random correlations and exact correlations of an ordering-biased reference configuration. The reference proves nonzero-target attainability but never initializes a search. Random targets may be unattainable exactly in a finite cell. This is not a sweep over all SRO strengths.</dd>
</dl></div><div><dl>
<dt>Sampling</dt><dd>Seed 20261019 on every combination. Seed 21261019 repeats only the two 128-site nonzero-target cases: 10 problem/seed combinations × four methods.</dd>
<dt>Equal resource allocation</dt><dd>30 seconds including startup and initialization. Native methods use four MPI ranks; mcsqs and icet use four independent workers.</dd>
<dt>Search objective</dt><dd>Matched ATAT matching-range objective, WD=0, WN=1, WR=1, tolerance=0.001. Optimization scores are distinct from actual correlation errors.</dd>
</dl></div></div>
<p><strong>Recorded host:</strong> __HOST__. ATAT 3.52 and icet 4.0 were used for the external comparisons. Timing comparisons apply to this environment and allocation.</p>
</section>
<section id="trials"><h2>Explore the trial metrics</h2><p>Select a case, error metric, and threshold. RMS and mean absolute errors below refer to the structure selected for lowest maximum absolute error. Their threshold times are assessed independently over all observed candidates.</p>
<div class="filters"><label>Case<select id="case"><option value="all">All cases</option></select></label><label>Method<select id="method"><option value="all">All methods</option></select></label><label>Threshold metric<select id="metric"><option value="max_error">Maximum absolute error</option><option value="rms_error">RMS error</option><option value="weighted_l1">Mean absolute error</option></select></label><label>Threshold<select id="threshold"><option>0.05</option><option>0.025</option><option selected>0.01</option></select></label></div>
<p id="count" class="small" aria-live="polite"></p>
<div class="scroll"><table><caption>Individual measured trials; “not reached” is right-censored at the trial budget.</caption><thead><tr><th>Case / target</th><th>Method</th><th>Seed</th><th>Lowest max error</th><th>RMS at best max</th><th>MAE at best max</th><th>Time to threshold</th><th>Output</th></tr></thead><tbody id="rows"></tbody></table></div>
<details><summary>Per-order maximum errors at the best-maximum-error structure</summary><div class="scroll"><table><thead><tr><th>Case / target</th><th>Method</th><th>Seed</th><th>Pairs</th><th>Triplets</th><th>Quadruplets</th></tr></thead><tbody id="orders"></tbody></table></div></details>
<noscript>The overview figure and protocol remain available without JavaScript. Enable JavaScript for the filterable trial tables.</noscript>
<p><button id="download">Download all 40 trials as JSON</button></p>
</section>
<section><h2>Definitions and interpretation</h2><dl>
<dt>Actual correlation errors</dt><dd>For achieved-minus-target residual Δᵢ over K assessed components: maximum absolute error = maxᵢ abs(Δᵢ); RMS = √(Σᵢ Δᵢ²/K); mean absolute error = Σᵢ abs(Δᵢ)/K. All methods use the same component definitions for each case. These tolerances do not imply a calibrated error in energy or another material property.</dd>
<dt>Lowest observed error</dt><dd>The minimum maximum absolute error among valid structures observed within the budget. RMS, mean absolute, and per-order errors in the tables describe that selected structure; they need not equal the independently lowest values of those metrics.</dd>
<dt>First-observed time</dt><dd>Readers sample at 1–10, 15, 20, 25, and 29.98 seconds, plus completed native outputs. Native checkpoints are emitted at proposal boundaries at intervals of at least 0.25 seconds. Read-completion timestamps determine availability; observations after 30 seconds are excluded. Threshold times are upper bounds on internal attainment times.</dd>
<dt>Censoring and limitations</dt><dd>Failure to reach a threshold means right-censoring at 30 seconds, not a measured time-to-solution. Few seeds, two structure families, and finite cell sizes limit generalization. Different machines, budgets, or objective functions cannot support direct speedup claims from these results. Examples for diborides, MAX phases, and Heusler phases are not extra trials in this campaign.</dd>
</dl></section>
<section><h2>Reproduce and audit</h2><p>Frozen results and fixtures require neither ATAT nor icet. The first command reads archived data; the second runs a separate short native-method demonstration, not this full campaign.</p>
<pre>m_scraps benchmark --report-only --output archived-results
m_scraps benchmark --seconds 5 --ranks 4 --output fresh-results</pre>
<p>The <a href="https://github.com/rarroyave/m_scraps#optional-full-four-method-rerun">full four-method rerun</a> requires the documented external engines. Routine regression tests check saved numerical references, not wall-clock performance.</p>
<p>Regenerate this standalone report from the repository:</p><pre>python tools/benchmark/submission/html_report.py --output docs/benchmark_report.html</pre>
<details><summary>Source provenance and SHA-256 checksums</summary><p>Source directory: <code>tools/fixtures/submission_benchmark</code>. This report embeds the archived trial data and figure; it has no external JavaScript, fonts, or chart dependencies.</p><pre>__HASHES__</pre><h3>Recorded protocol</h3><pre>__PROTOCOL__</pre><h3>Recorded environment</h3><pre>__ENV__</pre></details>
</section>
<footer>m_scraps · Please cite both the <a href="https://doi.org/10.1038/s43588-020-00006-7">original SCRAPS paper</a> and the <a href="https://doi.org/10.5281/zenodo.23475896">m_scraps software archive</a>. See the repository for attribution, license, and funding acknowledgments.</footer>
</main><script type="application/json" id="data">__DATA__</script>
<script>
'use strict';
const {rows,labels}=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id);
const caseLabel=c=>c.replace('_sro_ordering',' · ordering target').replace(/^(bcc|rocksalt)_(\d+)/,(_,s,n)=>s.toUpperCase()+' '+n)+(c.includes('sro')?'':' · random target');
for(const c of [...new Set(rows.map(r=>r.case))].sort()) $('case').add(new Option(caseLabel(c),c));
for(const [value,label] of Object.entries(labels)) $('method').add(new Option(label,value));
const num=v=>v==null?'—':v.toPrecision(5);
function cell(tr,text,cls){const td=document.createElement('td');td.textContent=text;if(cls)td.className=cls;tr.appendChild(td);}
function update(){
 const filtered=rows.filter(r=>($('case').value==='all'||r.case===$('case').value)&&($('method').value==='all'||r.method===$('method').value)).sort((a,b)=>a.case.localeCompare(b.case)||a.seed-b.seed||Object.keys(labels).indexOf(a.method)-Object.keys(labels).indexOf(b.method));
 $('rows').replaceChildren();$('orders').replaceChildren();let successes=0;
 for(const r of filtered){const p=r.passage[$('metric').value].find(p=>p.threshold===Number($('threshold').value));successes+=Number(p.success);
 const tr=document.createElement('tr');[caseLabel(r.case),labels[r.method],r.seed,num(r.max_error),num(r.rms_at_best_max),num(r.mean_absolute_at_best_max)].forEach(v=>cell(tr,v));cell(tr,p.success?p.seconds.toFixed(3)+' s':'Not reached (≥'+r.budget_seconds+' s)',p.success?'ok':'censored');cell(tr,r.outcome==='validated_output'?'Validated':r.outcome);$('rows').appendChild(tr);
 const order=document.createElement('tr');[caseLabel(r.case),labels[r.method],r.seed,...['2','3','4'].map(o=>num(r.by_order?.[o]?.max_error))].forEach(v=>cell(order,v));$('orders').appendChild(order);}
 $('count').textContent=filtered.length+' trials shown · '+successes+' reached the selected threshold · '+(filtered.length-successes)+' right-censored.';
}
for(const id of ['case','method','metric','threshold']) $(id).addEventListener('change',update);
$('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='m_scraps-benchmark-40-trials.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});update();
</script></body></html>'''
    host = (f"{environment['cpu']} Mac Studio; {environment['physical_cores']} physical cores, "
            f"{environment['memory_bytes'] // 2**30} GiB RAM; {environment['platform']}")
    replacements = {'SUMMARY': ''.join(summary), 'SVG': svg, 'HOST': html.escape(host),
                    'HASHES': html.escape(provenance), 'PROTOCOL': html.escape(raw['protocol']),
                    'ENV': html.escape(raw['environment']), 'DATA': package}
    for key, value in replacements.items():
        template = template.replace('__' + key + '__', value)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/benchmark_report.html')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render())
    print(args.output)


if __name__ == '__main__':
    main()
