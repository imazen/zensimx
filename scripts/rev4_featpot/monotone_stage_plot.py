#!/usr/bin/env python3
"""Plot exact-replay TRAIN panels; no fitting, scoring or checkpoint selection."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean

from activation_curriculum import sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('stages','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    summary_path=args.stages/'SUMMARY.json';summary=json.loads(summary_path.read_text())
    names=['control','easy_first','hard_first','family_blocks','teacher_warm']
    epochs=[0,29,59,89,119]
    populations=[('human_kadid','KADID'),('human_tid2013','TID2013'),('human_konfig','KonFiG F'),
                 ('human_cid22_a25','CID22 human'),('cid22','CID22 SSIM2 teacher')]
    if summary['complete']!=5 or summary['required']!=5 or set(summary['arms'])!=set(names):
        raise ValueError('all five registered schedules must finish')
    if summary['exact_final_replay_cells']!=15 or summary['checkpoint_coverage']!=1800:
        raise ValueError('incomplete exact replay or checkpoint retention')
    curves={};family_changes={}
    for name in names:
        path=args.stages/name/'STAGES.json'
        if sha(path)!=summary['arms'][name]:raise ValueError('stage panel drift')
        source=json.loads(path.read_text())
        if source['scope']!='TRAIN only':raise ValueError('not TRAIN-only panels')
        rows=source['panels'];curves[name]={};family_changes[name]={}
        for population,_ in populations:
            points={}
            for epoch in epochs:
                selected=[r for r in rows if r['scope']=='population' and r['population']==population and r['epoch']==epoch]
                if len(selected)!=3 or sorted(r['seed_index'] for r in selected)!=[0,1,2]:
                    raise ValueError('missing/duplicate population seed')
                values=[r['srocc_signed'] for r in sorted(selected,key=lambda r:r['seed_index'])]
                points[str(epoch)]=dict(mean=mean(values),minimum=min(values),maximum=max(values),values=values)
            curves[name][population]=points
            families=sorted({r['label'] for r in rows if r['scope']=='family' and r['population']==population})
            changes=[]
            for family in families:
                values={}
                for epoch in (89,119):
                    selected=[r for r in rows if r['scope']=='family' and r['population']==population and r['label']==family and r['epoch']==epoch]
                    if len(selected)!=3 or sorted(r['seed_index'] for r in selected)!=[0,1,2]:
                        raise ValueError('missing/duplicate family seed')
                    values[epoch]=mean(r['srocc_signed'] for r in selected)
                changes.append(dict(family=family,before=values[89],after=values[119],delta=values[119]-values[89]))
            family_changes[name][population]=sorted(changes,key=lambda x:x['delta'])
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(args.output/'matplotlib-config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    fig,axes=plt.subplots(1,5,figsize=(17,4.8))
    for ax,(population,label) in zip(axes,populations):
        ax.axvspan(89,119,color='#cbd5e1',alpha=.25)
        for name in names:
            points=curves[name][population]
            ax.plot(epochs,[points[str(e)]['mean'] for e in epochs],marker='o',markersize=3,label=name.replace('_',' '))
        ax.set_title(label);ax.set_xlabel('Epoch index');ax.set_xticks(epochs)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('TRAIN signed SROCC')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,.08))
    fig.suptitle('Distortion-order trajectories on TRAIN populations',fontsize=14)
    fig.text(.02,.025,'Mean of three fixed-seed fits. Shading: identical random final-quarter events. '
        'All 15 replays reproduce original model bytes and logged trajectories.\n'
        'These curves describe training interference and recovery; they do not establish a local minimum or select a checkpoint. Axis scales differ.',fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.94))
    for suffix in ('svg','png'):fig.savefig(args.output/('trajectories.'+suffix),dpi=160)
    plt.close(fig)
    write(args.output/'PLOT.json',dict(generator_sha256=sha(__file__),stage_summary_sha256=sha(summary_path),
        stage_panel_sha256=summary['arms'],curves=curves,family_changes_89_to_119=family_changes,
        scope='TRAIN only; all registered schedules and fixed epochs; no selection',
        files={p.name:sha(p) for p in args.output.glob('trajectories.*')}))
    print(json.dumps(dict(arms=5,exact_replays=15,plot_receipt_sha256=sha(args.output/'PLOT.json'))))


if __name__=='__main__':main()
