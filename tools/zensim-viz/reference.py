#!/usr/bin/env python3
"""Public registry equations with bibliographic links; no paper prose or private links."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path

formulas = {
      'ssim': r'\begin{aligned}m&=\min((\mu_r-\mu_d)^2,1)\\v&=\max(E[r^2+d^2]-\mu_r^2-\mu_d^2,0)\\q&=\max(E[(r-d)^2]-(\mu_r-\mu_d)^2,0)\\d_1&=m+(1-m)\frac{q}{v+0.0009}\end{aligned}',
      'edge_art': r'e=\frac{1+|d-\mu_d|}{1+|r-\mu_r|}-1,\quad a=\max(e,0)',
      'edge_det': r'e=\frac{1+|d-\mu_d|}{1+|r-\mu_r|}-1,\quad \ell=\max(-e,0)',
      'mse': r'\operatorname{MSE}=\operatorname{mean}[(r-d)^2]',
      'var_loss': r'\begin{aligned}V_j&=\operatorname{mean}[(j-\mu_j)^2]\\L_V&=\begin{cases}\max(1-V_d/V_r,0)&V_r>10^{-10}\\0&\text{otherwise}\end{cases}\end{aligned}',
      'tex_loss': r'\begin{aligned}A_j&=\operatorname{mean}|j-\mu_j|\\L_A&=\begin{cases}\max(1-A_d/A_r,0)&A_r>10^{-10}\\0&\text{otherwise}\end{cases}\end{aligned}',
      'contrast_inc': r'\begin{aligned}V_j&=\operatorname{mean}[(j-\mu_j)^2]\\g&=\max(V_d/V_r-1,0)\\G_V&=\begin{cases}g/(g+1)&V_r>10^{-10}\\0&\text{otherwise}\end{cases}\end{aligned}'
    }
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
v2_formulas['ssim']='\\begin{aligned}a&=2\\mu_r\\mu_d+10^{-4}\\\\b&=\\mu_r^2+\\mu_d^2+10^{-4}\\\\v&=E[r^2+d^2]-\\mu_r^2-\\mu_d^2+0.0009\\\\m&=(\\mu_r-\\mu_d)^2\\\\q&=\\max(E[(r-d)^2]-m,0)\\\\d_2&=\\max\\left(\\frac{vm+aq}{bv},0\\right)\\end{aligned}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['paper','catalog','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();revision=subprocess.check_output(['git','-C',str(a.paper),'rev-parse','main'],text=True).strip()
    paths=['papers/zensim-2026/refs/'+name for name in ['B1-metrics.bib','B2-datasets.bib','B3-methods.bib','B4-gaps.bib']]
    bibliography='\n'.join(subprocess.check_output(['git','-C',str(a.paper),'show',revision+':'+path],text=True) for path in paths)
    def cite(key):
        match=re.search(r'@\w+\{'+re.escape(key)+r',\s*(.*?)(?=\n\})',bibliography,re.S)
        if not match:raise ValueError('Missing bibliography entry '+key)
        fields=dict(re.findall(r'^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$',match[1],re.M))
        authors=fields['author'].split(' and ');surname=lambda v:v.split(',')[0] if ',' in v else v.split()[-1]
        label=surname(authors[0])+(' et al.' if len(authors)>2 else ' & '+surname(authors[1]) if len(authors)==2 else '')+' ('+fields['year']+')'
        url=fields.get('url') or ('https://doi.org/'+fields['doi'] if fields.get('doi') else 'https://arxiv.org/abs/'+fields['eprint'] if fields.get('eprint') else None)
        if not url:raise ValueError('Citation has no preserved URL')
        return dict(author_year=label,title=fields['title'].replace('{','').replace('}','').replace(r'\&','&'),url=url)
    entries={};inputs=json.loads(a.catalog.read_text())['inputs']
    for family,signal in sorted({(i['family'],i['signal']) for i in inputs}):
        keys=['sneyers2023ssimulacra2','alakuijala2016butteraugli','wang2003msssim'] if family=='basic' else ['wang2011iwssim'] if signal.startswith(('masked_','iw_')) else ['xue2014gmsd'] if signal in ['ssim','ssim_dev2','ssim_dev4','gms'] or signal.startswith('pjnd_transducer') else []
        formula=formulas[signal] if family=='basic' else v2_formulas[signal]
        code='ssim_form.rs' if family=='basic' and signal=='ssim' else 'hf_gain_form.rs' if family=='basic' and signal in ['var_loss','tex_loss','contrast_inc'] else 'feature_v2_viz.rs' if family=='basic' else 'feature_v2.rs'
        entries[family+'/'+signal]=dict(formula=formula,inspiration=[cite(k) for k in keys],code='https://github.com/imazen/zensimx/blob/main/zensim/src/'+code)
    entries['fragility']=dict(formula=r'x_i=\operatorname{f32}(f_i)\,\operatorname{f32}(b_i)',inspiration=[],code='https://github.com/imazen/zensimx/blob/main/tools/zensim-viz/export/model.rs')
    with a.out.open('x') as f:json.dump(dict(schema='zensim-public-reference-v1',entries=entries),f,ensure_ascii=False,separators=(',',':'))
    print(json.dumps(dict(entries=len(entries),formulae=sum(bool(e['formula']) for e in entries.values()),bibliographic_links=sum(len(e['inspiration']) for e in entries.values()))))

if __name__=='__main__':main()
