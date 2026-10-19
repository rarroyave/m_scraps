#!/usr/bin/env python3
"""Render the archived eight-method small-cell comparison without optional engines."""
import argparse
import csv
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = dict(labeled_exhaustive='Labeled enumeration', symmetry_exhaustive='Symmetry enumeration',
             symmetry_bound_stop='Bound-stopping enumeration', python_pair_sro_sa='Python pair-SRO SA',
             scraps_cuckoo='Native cuckoo', scraps_sa='Native SA', mcsqs='ATAT mcsqs', icet_atat='icet (matched score)')


def render(reference, added, output):
    original = json.loads((reference/'results.json').read_text())
    rows = json.loads((added/'summary.json').read_text())
    cases = json.loads((reference/'cases.json').read_text())['cases']
    output.parent.mkdir(parents=True, exist_ok=True)
    def fmt(v):
        return '—' if v is None else f'{v:.6g}' if isinstance(v,float) else str(v)
    def table(headers, data):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(fmt(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'
    outcomes=[]
    for method in ['scraps_cuckoo','scraps_sa','mcsqs','icet_atat']:
        group=[r for r in rows if r['method']==method]
        times=[r['time_to_0.01'] for r in group if r.get('time_to_0.01') is not None]
        outcomes.append([NAMES[method], sum(r.get('gap_to_optimum',float('inf'))<2e-10 for r in group),
                         len(times), statistics.median(times) if times else None])
    outcome_table=table(['Method','Cases at certified optimum / 12','Cases reaching 0.01 / 12','Median observed time to 0.01 (s)'],outcomes)
    comparison=[]
    for case in cases:
        line=[case['id'], case['optimum']]
        for method in NAMES:
            selected=[r for r in original+rows if r['case']==case['id'] and r['method']==method]
            errors=[r['max_error'] for r in selected if r.get('max_error') is not None]
            line.append(min(errors) if errors else 'capped' if selected and all(r['status']=='configuration_cap' for r in selected) else 'no output')
        comparison.append(line)
    headers=['Case','Certified minimum',*NAMES.values()]
    columns=['case','method','seed','status','max_error','rms_error','mean_absolute_error','gap_to_optimum',
             'time_to_0.01','time_to_0.05','elapsed_seconds']
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        writer.writerows({k:r.get(k) for k in columns} for r in rows)
    details=table(columns,[[r.get(k) for k in columns] for r in sorted(rows,key=lambda r:(r['case'],r['method']))])
    output.write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>m_scraps: small cells, all methods</title>
<style>body{font:15px system-ui;margin:28px;color:#172333}p{max-width:1050px;line-height:1.5}table{border-collapse:collapse;font-size:12px}th,td{padding:9px;border-bottom:1px solid #ccd4dd;text-align:right}th{background:#eaf0f6}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f6f8fa}.scroll{overflow:auto}h1{font-size:28px}</style>
<h1>Small-cell comparison: all eight methods</h1>
<p>12 fixed-cell problems: BCC Cu–Zn and two-sublattice rocksalt (Ti,Zr)(C,N), 8/16/32 atoms, targets 0 and −0.1. Each case has one first-shell pair-SRO component. Maximum error, RMS error and MAE are consequently identical. These are not higher-order or cell-shape benchmarks.</p>
<p>The original 72 records contain three enumeration variants and Python pair-SRO SA (three seeds). The extension adds 48 trials: native cuckoo, native SA, ATAT mcsqs and icet, each with seed 42, one worker and five seconds. Cells, compositions and targets match; native initialization differs across engines. The table reports the lowest observed error for each method/case.</p>
<p>The binary pair correlation equals the selected SRO alpha for these equiatomic groups; mappings and best outputs were checked with ATAT corrdump. The added methods use the established ATAT figure score, with the same error ranking for these single-component discrete problems. icet uses a custom matched calculator, not its stock objective. Certified minima are from exhaustive enumeration or a witness attaining a rigorous bound.</p>
<h2>Added-method outcomes</h2><p>Median threshold times use only successful cases. All methods have the same seven eligible cases; the other five have certified error floors above 0.01. Observation cadence and startup overhead differ, so these are not isolated kernel-speed measurements.</p><div class="scroll">'''+outcome_table+'''</div>
<h2>Lowest errors</h2><div class="scroll">'''+table(headers,comparison)+'''</div>
<h2>Added-method timing and accuracy</h2>
<p>Times are seconds from process launch, including executable/import startup; input preparation and post-search verification are excluded. ATAT/icet observations occur at approximately 1-second intervals, plus 4.98 seconds. Native observations include checkpoints and completed outputs. Threshold times are first-observed upper bounds, with tolerance 1e-12; a dash means not observed, not zero. Search wall time includes stopping processes. The earlier Python SA timings have a different startup/preparation boundary and cannot support a direct speedup claim.</p>
<p>One seed is a limited screening result, not evidence of broad statistical superiority. A capped enumeration is not a failed optimizer. Frozen protocol, environment, inputs, measurements and witnesses are under tools/fixtures/small_cells_external/; the companion CSV contains all added-method rows. Rerunning external engines is optional; regression tests need neither ATAT nor icet.</p>
<div class="scroll">'''+details+'</div></html>\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'tools/fixtures/small_cells')
    parser.add_argument('--added',type=Path,default=ROOT/'tools/fixtures/small_cells_external')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/small_cell_all_methods.html')
    args=parser.parse_args();render(args.reference,args.added,args.output)
