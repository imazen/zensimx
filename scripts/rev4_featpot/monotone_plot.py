#!/usr/bin/env python3
"""Plot existing native results and paired source intervals; never fit/select models."""
import argparse
import json
import os
from pathlib import Path

from activation_curriculum import sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('summary','bootstrap','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--control-summary',type=Path)
    parser.add_argument('--title',required=True)
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text());bootstrap=json.loads(args.bootstrap.read_text())
    if sha(args.summary)!=bootstrap['summary_sha256']:raise ValueError('plot source summary drift')
    control_summary_path=args.control_summary or args.summary
    if bootstrap['control_summary_sha256']!=sha(control_summary_path):
        raise ValueError('plot control summary drift')
    control=json.loads(control_summary_path.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(args.output/'matplotlib-config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    names=list(summary['arms']);baseline=bootstrap['baseline']
    failed={cell.split('/')[0] for cell in summary.get('feature_gate_failures',[])}
    populations=[('human_kadid_development','KADID'),('human_tid2013_development','TID2013'),
                 ('human_konfig_development','KonFiG F'),('human_cid22_a25_development','CID22 human')]
    fig,axes=plt.subplots(1,4,sharey=True,figsize=(15,max(4.5,.33*len(names)+2)))
    for ax,(population,label) in zip(axes,populations):
        source=bootstrap['populations'][population]
        ax.axvline(0,color='#64748b',linewidth=.8,zorder=1)
        for i,name in enumerate(names):
            if 'comparisons' in source:
                point=source['comparisons'][name]['point'];low=source['comparisons'][name]['low'];high=source['comparisons'][name]['high']
                ax.plot([low,high],[i,i],color='#64748b',linewidth=1.5,zorder=2)
            else:
                point=summary['arms'][name][population]['pooled_srocc']['mean']-control['arms'][baseline][population]['pooled_srocc']['mean']
            ax.scatter([point],[i],color='#2563eb' if point>=0 else '#c2410c',s=24,zorder=3)
        count=source['references'];ax.set_title(label+f"\n{count} development source"+('s' if count!=1 else ''))
        ax.set_xlabel('Signed SROCC difference')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(names)),[name.replace('_',' ')+(' [gate failed]' if name in failed else '') for name in names]);axes[0].invert_yaxis()
    fig.suptitle(args.title+' — versus '+baseline.replace('_',' '),fontsize=13)
    fig.text(.02,.025,'Dots: mean of three fixed-final fits. Bars: 95% paired source-bootstrap intervals (2,000 draws).\n'
        'Conditional on exposed development sources; no multiplicity correction. KonFiG has one source, so no interval. Axis scales differ.'
        + (' Failed-gate candidates are diagnostic only.' if failed else ''),fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.94))
    for suffix in ('svg','png'):fig.savefig(args.output/('comparison.'+suffix),dpi=160)
    plt.close(fig)
    write(args.output/'PLOT.json',dict(generator_sha256=sha(__file__),summary_sha256=sha(args.summary),bootstrap_sha256=sha(args.bootstrap),
        control_summary_sha256=sha(control_summary_path),
        title=args.title,files={p.name:sha(p) for p in args.output.glob('comparison.*')}))
    print(json.dumps(dict(arms=len(names),plot_receipt_sha256=sha(args.output/'PLOT.json'))))


if __name__=='__main__':main()
