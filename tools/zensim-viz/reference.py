#!/usr/bin/env python3
"""Join native registry signals to read-only paper prose and bibliography.

Only the four named design/model/spatial documents and bibliography are read.
Claim registries and result tables are deliberately not consulted. Equations
are separately curated against revision-7 source; the Typst draft has no
fenced LaTeX equations. No metric arithmetic is performed by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.paper), 'rev-parse', 'main'], text=True).strip()
    prefix = 'papers/zensim-2026/'
    paths = ['sections/03-design.typ', 'sections/09-spatial.typ', 'sections/a2-models.typ', 'main.typ', *['refs/'+n for n in ['B1-metrics.bib','B2-datasets.bib','B3-methods.bib','B4-gaps.bib']]]
    docs = {p: subprocess.check_output(['git','-C',str(args.paper),'show',f'{revision}:{prefix}{p}'],text=True) for p in paths}
    bibliography = '\n'.join(v for p,v in docs.items() if p.endswith('.bib'))
    def cite(key):
        match = re.search(r'@\w+\{'+re.escape(key)+r',\s*(.*?)(?=\n\})', bibliography, re.S)
        if not match:
            raise ValueError('missing bibliography key '+key)
        fields = dict(re.findall(r'^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$',match[1],re.M))
        title = fields['title'].replace('{','').replace('}','').replace(r'\&','&')
        authors = fields['author'].split(' and ')
        # Bib entries use both "Surname, Given" and "Given Surname".
        surname = lambda a: a.split(',')[0] if ',' in a else a.split()[-1]
        label = surname(authors[0])+(' et al.' if len(authors)>2 else ' & '+surname(authors[1]) if len(authors)==2 else '')+' ('+fields['year']+')'
        url = fields.get('url') or ('https://doi.org/'+fields['doi'] if fields.get('doi') else 'https://arxiv.org/abs/'+fields['eprint'] if fields.get('eprint') else None)
        if not url: raise ValueError('citation has no preserved URL: '+key)
        return dict(key=key,author_year=label,title=title,url=url)
    paper_url = f'https://github.com/imazen/zenpapers/blob/{revision}/{prefix}sections/03-design.typ'
    code_base = 'https://github.com/imazen/zensimx/blob/main/zensim/src/'
    source = docs['sections/03-design.typ'].splitlines()
    line_basic = next(i+1 for i,s in enumerate(source) if s.startswith('zensim keeps that vocabulary'))
    line_weighted = next(i+1 for i,s in enumerate(source) if s.startswith('Every feature has a stable ID'))
    line_v2 = next(i+1 for i,s in enumerate(source) if '[v2]' in s)
    base_cites = ['sneyers2023ssimulacra2','alakuijala2016butteraugli','wang2003msssim']
    formulas = {
      'ssim': r'\begin{aligned}m&=\min((\mu_r-\mu_d)^2,1)\\v&=\max(E[r^2+d^2]-\mu_r^2-\mu_d^2,0)\\q&=\max(E[(r-d)^2]-(\mu_r-\mu_d)^2,0)\\d_1&=m+(1-m)\frac{q}{v+0.0009}\end{aligned}',
      'edge_art': r'e=\frac{1+|d-\mu_d|}{1+|r-\mu_r|}-1,\quad a=\max(e,0)',
      'edge_det': r'e=\frac{1+|d-\mu_d|}{1+|r-\mu_r|}-1,\quad \ell=\max(-e,0)',
      'mse': r'\operatorname{MSE}=\operatorname{mean}[(r-d)^2]',
      'var_loss': r'\begin{aligned}V_j&=\operatorname{mean}[(j-\mu_j)^2]\\L_V&=\begin{cases}\max(1-V_d/V_r,0)&V_r>10^{-10}\\0&\text{otherwise}\end{cases}\end{aligned}',
      'tex_loss': r'\begin{aligned}A_j&=\operatorname{mean}|j-\mu_j|\\L_A&=\begin{cases}\max(1-A_d/A_r,0)&A_r>10^{-10}\\0&\text{otherwise}\end{cases}\end{aligned}',
      'contrast_inc': r'\begin{aligned}V_j&=\operatorname{mean}[(j-\mu_j)^2]\\g&=\max(V_d/V_r-1,0)\\G_V&=\begin{cases}g/(g+1)&V_r>10^{-10}\\0&\text{otherwise}\end{cases}\end{aligned}'
    }

    # Algebraic notation transcribed from dense_terms32, gradient_terms64 and
    # finish_channel_scale_at_revision. The native fold remains the numeric
    # oracle; these equations do not prescribe floating-point reassociation.
    v2_formulas = {
      'art': r'\begin{aligned}u&=|r-\mu_r|,\quad v=|d-\mu_d|\\t&=\frac{(u-v)^2}{u^2+v^2+10^{-4}}\\a&=t\,\mathbf1_{v>u},\quad f=\operatorname{clip}_{0,1}E[a]\end{aligned}',
      'det': r'\begin{aligned}u&=|r-\mu_r|,\quad v=|d-\mu_d|\\t&=\frac{(u-v)^2}{u^2+v^2+10^{-4}}\\\ell&=t\,\mathbf1_{v<u},\quad f=\operatorname{clip}_{0,1}E[\ell]\end{aligned}',
      'mse': r'f=\operatorname{clip}_{0,1}E\!\left[\frac{(r-d)^2}{(r-d)^2+0.01}\right]',
      'hf_gain': r'\begin{aligned}h_j&=j-\mu_j\\f&=\operatorname{clip}_{0,1}E\!\left[\frac{\max(h_d^2-h_r^2,0)}{h_d^2+h_r^2+10^{-4}}\right]\end{aligned}',
      'hf_loss': r'\begin{aligned}h_j&=j-\mu_j\\f&=\operatorname{clip}_{0,1}E\!\left[\frac{\max(h_r^2-h_d^2,0)}{h_d^2+h_r^2+10^{-4}}\right]\end{aligned}',
      'hf_mag_loss': r'\begin{aligned}h_j&=j-\mu_j\\f&=\operatorname{clip}_{0,1}E\!\left[\frac{\max(|h_r|-|h_d|,0)}{|h_r|+|h_d|+10^{-4}}\right]\end{aligned}',
      'ssim_dev2': r'f=\operatorname{clip}_{0,2}\sqrt{E[(d_2-E[d_2])^2]}',
      'ssim_dev4': r'f=\operatorname{clip}_{0,2}\sqrt[4]{E[(d_2-E[d_2])^4]}',
      'gms': r'\begin{aligned}g_j&=\sqrt{(j_{x+1}-j_{x-1})^2+(j_{y+1}-j_{y-1})^2}\\f&=E\!\left[\frac{(g_r-g_d)^2}{g_r^2+g_d^2+10^{-4}}\right]\end{aligned}',
      'ringing': r'\begin{aligned}S(z,c)&=\frac{\max(z,0)}{\max(z,0)+c}\\f&=E[S(|r-d|,0.05)\\&\quad\cdot S(a,0.01)(1-S(g_r,0.02))]\end{aligned}',
      'banding': r'f=E\!\left[\frac{\max(g_d-g_r,0)}{g_d+g_r+0.01}\frac{0.02}{g_r+0.02}\right]',
      'edge_width_change': r'\begin{aligned}D_j&=\frac{E[g_j]_{s+1}}{E[g_j]_s+10^{-4}}\\f_s&=\frac{(D_r-D_d)^2}{D_r^2+D_d^2+10^{-3}}\\f_{S-1}&=f_{S-2}\end{aligned}',
      'blockiness': r'\begin{aligned}B_j^x&=|j_{x,y}-j_{x-1,y}|\\B_j^y&=|j_{x,y}-j_{x,y-1}|\\f&=E\!\left[\sum_{t\in\{x,y\}}\mathbf1_{t>0,\,t\equiv0\ (8)}\frac{\max(B_d^t-B_r^t,0)}{B_d^t+B_r^t+10^{-3}}\right]\end{aligned}'
    }
    for name,k in [('pjnd_transducer',4),('pjnd_transducer_low_k',1),('pjnd_transducer_high_k',16)]:
        v2_formulas[name] = r'f=\operatorname{clip}_{0,1}E\!\left[\frac{|r-d|}{|r-d|+0.1(1+'+str(k)+r'a)}\right]'
    for mode in ['masked','iw']:
        for detector in ['ssim','art','det','mse']:
            weight = r'1-\frac{a}{a+0.01}' if mode=='masked' else r'\frac{a}{a+0.01}+10^{-3}'
            v2_formulas[mode+'_'+detector] = r'\begin{aligned}w&='+weight+r'\\z&=\mathrm{v2\_'+detector+r'}\\f&=\operatorname{clip}_{0,2}\begin{cases}\frac{\sum wz}{\sum w}&\sum w\ge10^{-12}\\0&\text{otherwise}\end{cases}\end{aligned}'
    for detector in ['ssim','art','det']:
        v2_formulas[detector+'_soft_peak'] = r'\begin{aligned}z&=\mathrm{v2\_'+detector+r'}\\w&=\frac{\max(z,0)}{\max(z,0)+0.05}\\f&=\operatorname{clip}_{0,2}\begin{cases}\frac{\sum wz}{\sum w}&\sum w\ge10^{-12}\\0&\text{otherwise}\end{cases}\end{aligned}'
    detects = {
      'ssim':'Measures local structural and mean differences in XYB. Mean, L2 and L4 pooling emphasize different parts of the same detector distribution.',
      'edge_art':'Responds to high-frequency magnitude added in the distorted image, including ringing and blocking. It uses the positive half of the asymmetric edge ratio.',
      'edge_det':'Responds to local detail lost in the distorted image, including smoothing. It uses the negative half of the asymmetric edge ratio.',
      'mse':'Measures squared XYB sample error. It preserves the channel and scale of the stored feature rather than treating the input as display RGB.',
      'var_loss':'Measures loss of high-frequency energy using moments over the whole scale/channel. This is a global ratio, not a local error map.',
      'tex_loss':'Measures loss of mean absolute high-frequency magnitude over the scale/channel. This is a global ratio and has no pre-pooling spatial detector.',
      'contrast_inc':'Measures increased high-frequency energy. Revision 7 uses the registered saturating-excess form; the paper describes the family without a standalone equation.'
    }
    entries = {}
    weights=json.loads(args.catalog.read_text())
    signals=sorted({(i['family'],i['signal']) for i in weights['inputs']})
    for family,signal in signals:
        keys=[];line=line_v2;covered=False;formula=None;terms=[]
        if family=='basic':
            covered=True;line=line_basic;keys=base_cites;formula=formulas[signal];text=detects[signal]
            background='The paper traces these detectors to SSIMULACRA2 and the XYB colour space introduced by Butteraugli. zensim uses four XYB scales and box windows; this does not claim an official SSIMULACRA2 map.'
            terms=['mu_r','mu_d','variance_sum','error_moment','d1','artifact','detail','mse'] if signal in ['ssim','edge_art','edge_det','mse'] else []
        elif signal.startswith(('masked_','iw_')):
            covered=True;line=line_weighted;keys=['wang2011iwssim'];text='Re-pools a local error signal with weights derived from reference activity. Masked pools down-weight activity; IW pools up-weight it.';background="The paper explicitly identifies IW as an activity proxy, not IW-SSIM's information-content weighting. It claims no numerical parity with IW-SSIM."
        elif signal in ['ssim','ssim_dev2','ssim_dev4','gms'] or signal.startswith('pjnd_transducer'):
            covered=True;keys=['xue2014gmsd'];text='The paper describes the v2 block as bounded similarities, GMSD-style deviation pooling and JND transducers. The exact stored signal and pooling are given above.';background='This citation supports the deviation-pooling inspiration of the block; it does not establish numerical equivalence to GMSD, or a detection threshold for this model.'
            if signal=='ssim':
                formula=r'\begin{aligned}a&=2\mu_r\mu_d+10^{-4}\\b&=\mu_r^2+\mu_d^2+10^{-4}\\v&=E[r^2+d^2]-\mu_r^2-\mu_d^2+0.0009\\m&=(\mu_r-\mu_d)^2\\q&=\max(E[(r-d)^2]-m,0)\\d_2&=\max\left(\frac{vm+aq}{bv},0\right)\end{aligned}'
        else:
            text='no paper text yet for this signal. Its registry identity, native map role and stored model evidence remain available.';background='The paper lists the v2 block, but does not yet describe this individual detector. See its implementation; no formula or inspiration is inferred from the name.'

        if family=='v2' and signal in v2_formulas:
            formula=v2_formulas[signal]
        notation = ('r,d are positive XYB samples; μ are the retained local box means. E and sums cover the scale/channel. a is retained reference activity; g is central-difference gradient magnitude. d₂ and v2_ssim name the bounded SSIM detector, and v2_art/det/mse name the local detectors before pooling. clip is the native finalizer bound. Equations describe real arithmetic; native f32 operation order and revision-7 reductions define the exported numbers.' if family=='v2' else 'r,d are positive XYB samples; μ are retained box means. Means and norms pool over the scale/channel. Native f32 mul_add and deterministic reductions define the exported values. The basic SSIM map uses the Clamp luminance form.')
        entries[f'{family}/{signal}']=dict(formula=formula,notation=notation,detects=text,background=background,inspiration=[cite(k) for k in keys],section=paper_url+f'#L{line}',paper_anchor='sec:des',terms=terms,paper_text=covered,paper_excerpt=source[line-1] if covered else None,code=code_base+('ssim_form.rs#L1585' if family=='basic' and signal=='ssim' else 'hf_gain_form.rs#L324' if family=='basic' and signal in ['var_loss','tex_loss','contrast_inc'] else 'feature_v2_viz.rs#L150' if family=='basic' else 'feature_v2.rs#L1912'))
    entries['fragility']=dict(formula=r'x_i=\operatorname{f32}(f_i)\,\operatorname{f32}(b_i)',detects='The bake declares a product of its damage slot and a reference-only fragility slot. This is a model input, not a new local detector.',background='no paper text yet for these product inputs; the declaration parser and native scorer define them.',inspiration=[],section=f'https://github.com/imazen/zenpapers/blob/{revision}/{prefix}sections/a2-models.typ#L1',paper_anchor='sec:models',terms=[],paper_text=False,code='https://github.com/imazen/zensimx/blob/main/tools/zensim-viz/export/model.rs#L20')
    matched={f"{i['family']}/{i['signal']}" for i in weights['inputs']}
    assert matched==set(entries)-{'fragility'}
    assert any(i['factor'] is not None for i in weights['inputs'])
    out=dict(code_sources={str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve().parents[2]/name for name in ['zensim/src/ssim_form.rs','zensim/src/hf_gain_form.rs','zensim/src/feature_v2_oracle.rs','zensim/src/feature_v2_kernels.rs','zensim/src/feature_v2.rs','tools/zensim-viz/export/model.rs']]},schema='zensim-viz-reference-v1',paper_commit=revision,sources={prefix+p:hashlib.sha256(v.encode()).hexdigest() for p,v in docs.items()},equation_origin='Revision-7 code transcription; no fenced LaTeX blocks in the four paper sources.',counts=dict(entries=len(entries),family_signals=len(signals),rendered_equations=sum(bool(e['formula']) for e in entries.values()),paper_text=sum(e['paper_text'] for e in entries.values()),no_paper_text=sum(not e['paper_text'] for e in entries.values()),sidebar_inputs=len(weights['inputs'])),entries=entries)
    with args.out.open('x') as stream: json.dump(out,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(out['counts']))

if __name__=='__main__':main()
