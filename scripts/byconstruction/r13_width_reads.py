"""R13 invokes the existing R12 assessments with one shared lock per job."""
import sys

import r13_width as owner
import r13_width_analysis as analysis


def main():
    r = owner.O
    common = [sys.executable,'-B',owner.X/'scripts/byconstruction/r12_linear_reads.py',
              '--models',r/'MODELS.json','--out',r/'reads']
    for stage in ('eval','e36','chroma','hdr'):
        owner.job('read-'+stage,common+['--only',stage])
    for name in analysis.models():
        for stage in ('steer','dial','nearid'):
            owner.job(f'read-{stage}-{name}',common+['--only',stage,'--include-model',name])


if __name__=='__main__':main()
