#!/usr/bin/env python3
"""Self-contained native benchmark with frozen ATAT/icet reference results."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import benchmark_atat_reference as bench
from scraps_ase import _find_mpirun

FIXTURES=Path(__file__).resolve().parent/'fixtures/submission_benchmark'


def load_corpus():
    manifest=json.loads((FIXTURES/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((FIXTURES/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Frozen benchmark fixture changed: {name}')
    cases={name:json.loads(gzip.decompress((FIXTURES/(name+'.json.gz')).read_bytes()))['case'] for name in manifest['cases']}
    return manifest,cases,json.loads((FIXTURES/'baseline.json').read_text())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=['bcc_128','rocksalt_128'])
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--ranks',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261019)
    parser.add_argument('--binary',type=Path,default=bench.REPO/'SCRAPs/scraps')
    parser.add_argument('--report-only',action='store_true',help='Inspect frozen results only; no native build or MPI needed')
    args=parser.parse_args(argv)
    manifest,cases,baseline=load_corpus()
    if set(args.cases)-cases.keys():parser.error('Unknown case; available: '+', '.join(cases))
    if not 0<args.seconds<=30 or args.ranks<1:parser.error('seconds must be in (0,30], ranks positive')
    args.binary=args.binary.resolve();args.mpirun=_find_mpirun();args.checkpoint_seconds=.25
    if not args.report_only and (not args.binary.exists() or not args.mpirun):
        parser.error('Run bash build.sh and provide a matching MPI launcher, or use --report-only')
    args.output.mkdir(parents=True,exist_ok=False)
    references=[dict(row,measurement='archived_reference') for row in baseline if row['case'] in args.cases]
    report=dict(protocol='Frozen submission figures; WD=0, WN=1, WR=1; actual decorated correlation residuals',
                comparison_limit='Archived references are from a different run/host; no measured speedup claim. Resource and wall-time budgets must be checked explicitly.',
                archived_reference=references,fresh_native=[])
    provenance=dict(platform=platform.platform(),python=sys.version,seconds=args.seconds,ranks=args.ranks,seed=args.seed,
                    fixture_manifest_sha256=hashlib.sha256((FIXTURES/'manifest.json').read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    measurement='archived_only' if args.report_only else 'fresh_native_and_archived_reference')
    if not args.report_only:
        provenance['native_binary_sha256']=hashlib.sha256(args.binary.read_bytes()).hexdigest()
        bench.FIXTURES=FIXTURES
        original=bench.input_text
        for ident in args.cases:
            for method in ['scraps_cuckoo','scraps_sa']:
                def input_text(case,profile,seed):
                    text=original(case,profile,seed)
                    return text.replace('TOTALIter 2\n','TOTALIter 40\n')+'MC_KERNEL sa\nSA_T_INITIAL 2.0\nSA_T_FINAL 0.001\n' if method=='scraps_sa' else text+'MC_KERNEL cuckoo\n'
                bench.input_text=input_text
                try:record=bench.run_trial(cases[ident],dict(name='uniform_figures',wd=0,wn=1),args.seed,args,args.output/ident/method)
                finally:bench.input_text=original
                counts = cases[ident]['figure_counts']
                for candidate in record['candidates']:
                    values = candidate['metrics']
                    values['rms_error'] = math.sqrt(sum(counts[o]*row['rms']**2
                        for o,row in values['by_order'].items()) / sum(counts.values()))
                    values['mean_absolute_error'] = values['weighted_l1']
                record['passage'] = {metric: [bench.first_passage(record['candidates'], metric, t, args.seconds)
                    for t in [.05, .025, .01]] for metric in ['max_error', 'rms_error', 'weighted_l1']}
                record.update(method=method,measurement='fresh_native')
                report['fresh_native'].append(record)
                print(json.dumps(dict(case=ident,method=method,max_error=min((c['metrics']['max_error'] for c in record['candidates']),default=None),errors=record['errors'])),flush=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return 1 if any(r['errors'] for r in report['fresh_native']) else 0


if __name__=='__main__':raise SystemExit(main())
