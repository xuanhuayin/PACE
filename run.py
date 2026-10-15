"""Portable launcher for native capture, graph compilation, and PACE generation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=('tess','llada','llada2','dream'),required=True)
    ap.add_argument('--benchmark',choices=('ifeval','ifbench'),default='ifeval')
    ap.add_argument('--stage',choices=('native','compile','generate'),required=True)
    ap.add_argument('--checkpoint',help='public model ID or local checkpoint; use this for LLaDA-1.5')
    ap.add_argument('--work-dir',default='outputs')
    ap.add_argument('extra',nargs=argparse.REMAINDER,help='arguments after -- are passed to the selected script')
    a=ap.parse_args()
    base=Path(a.work_dir).resolve()/a.model/a.benchmark
    native=base/'native'; graphs=base/'graphs'; output=base/'pace'
    for d in (base,native,graphs,output):d.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(NATIVE_DIR=str(native),GRAPHS_DIR=str(graphs),GRAPH_OUT=str(graphs),GRAPHS_OUT=str(graphs),OUT_DIR=str(output),MAP_DIR='',NLTK_DATA=str(ROOT/'assets/nltk'),IFBL_CERT_WORKERS='0',IFB4_CERT_POOL='0')
    # Keep the full public row order so graph and response ordinals stay aligned.
    env.pop('ROWS_FILE',None)
    env['ROWS_FILE']=str(ROOT/('assets/ifeval/data/input_data.jsonl' if a.benchmark=='ifeval' else 'assets/ifbench/data/IFBench_test.jsonl'))
    env['PREDICATES']='predicates_hard' if a.benchmark=='ifeval' else 'predicates_ifb_hard'
    if a.checkpoint:
        env[dict(tess='TESS_MODEL',llada='LLADA_SNAP',llada2='LLADA2_SNAP',dream='DREAM_SNAP')[a.model]]=a.checkpoint
    if a.stage=='native':
        script=('ifeval_native_capture.py' if a.benchmark=='ifeval' else 'ifbench_native_capture.py') if a.model=='tess' else a.model+'_native_capture.py'
        flags=['--out',str(native)]
    elif a.stage=='compile':
        script=dict(tess='compile_ifeval_v2.py',llada='compile_llada_all.py',llada2='compile_llada2_all.py',dream='compile_dream_all.py')[a.model]
        flags=[]
    elif a.benchmark=='ifeval':
        script='program_run4.py' if a.model=='tess' else a.model+'_program_run4.py'
        flags=['--out',str(output)]
    else:
        script='llada_ifbench_program_run4.py' if a.model=='llada' else 'ifbench_program_run4.py'
        env['IFB4_MODEL']=a.model
        flags=['--out',str(output)]
    extra=a.extra[1:] if a.extra[:1]==['--'] else a.extra
    return subprocess.call([sys.executable,str(ROOT/'cah'/script),*flags,*extra],cwd=ROOT,env=env)

if __name__=='__main__':
    raise SystemExit(main())
