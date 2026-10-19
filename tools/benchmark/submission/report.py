#!/usr/bin/env python3
"""Render measured submission results without averaging away censored trials."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data=json.loads(args.summary.read_text())
    cases=sorted({r['case'] for r in data},key=lambda s:('sro' in s,'rocksalt' in s,int(s.split('_')[1])))
    methods=['scraps_cuckoo','scraps_sa','mcsqs','icet_atat']
    labels=['Cuckoo','SA','mcsqs','icet (matched objective)']
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
    for j,(method,label,color) in enumerate(zip(methods,labels,colors)):
        for i,case in enumerate(cases):
            rows=[r for r in data if r['case']==case and r['method']==method]
            for k,row in enumerate(rows):
                x=i+(j-1.5)*.17+(k-(len(rows)-1)/2)*.045
                err=row['max_error']
                if err is not None:axes[0].scatter(x,err,color=color,s=42,label=label if i==0 and k==0 else None)
                else:axes[0].text(x,1,'no output',rotation=90,ha='center',va='top',fontsize=7,transform=axes[0].get_xaxis_transform())
                passage=next(p for p in row['passage']['max_error'] if p['threshold']==.01)
                axes[1].scatter(x,passage['seconds'] if passage['success'] else 30,
                                marker='o' if passage['success'] else 'x',color=color,s=42)
    axes[0].axhline(.01,color='black',ls='--',lw=1)
    axes[0].set_yscale('log');axes[0].set_ylabel('Lowest observed maximum correlation error')
    axes[0].legend(ncols=4,loc='upper left',bbox_to_anchor=(0,1.15),frameon=False)
    axes[1].set_ylabel('First observed time to error ≤ 0.01 (s)')
    axes[1].set_ylim(0,33);axes[1].axhline(30,color='gray',ls=':',lw=1)
    axes[1].text(.99,.98,'× at 30 s: threshold not reached (right-censored)',ha='right',va='top',transform=axes[1].transAxes,fontsize=9)
    axes[1].set_xticks(range(len(cases)),[s.replace('_sro_ordering','\nnonzero target').replace('_',' ') if 'sro' in s else s.replace('_',' ')+'\nrandom target' for s in cases],fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('40 fixed trials · pairs, triplets and quadruplets · 30 s / four cores per trial',fontsize=13)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in data:
        row={k:r[k] for k in ['case','method','seed','max_error','rms_at_best_max','mean_absolute_at_best_max','outcome']}
        for order,values in (r['by_order'] or {}).items():
            row['max_error_order_'+order]=values['max_error']
        for p in r['passage']['max_error']:
            row['time_to_'+str(p['threshold'])]=p['seconds']
            row['success_'+str(p['threshold'])]=p['success']
        rows.append(row)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with args.output.with_suffix('.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    fig.savefig(args.output,dpi=180)
    fig.savefig(args.output.with_suffix('.svg'))
    svg = args.output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
