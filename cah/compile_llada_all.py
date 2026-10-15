import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
OUT = Path(os.environ.get('GRAPH_OUT', str(ROOT / 'cah/llada_graphs_v2')))
SNAP = os.environ.get('TOK_SNAP') or os.environ.get('LLADA_SNAP', 'GSAI-ML/LLaDA-8B-Instruct')
EOS_OVERRIDE = os.environ.get('EOS_ID')
_DEC = None

def init():
    global _DEC
    from transformers import AutoTokenizer
    from llada_decoder import LLaDADecoderContract
    extra = () if 'LLaDA' not in SNAP else None
    kw = dict(eos=int(EOS_OVERRIDE) if EOS_OVERRIDE else None)
    if extra is not None:
        kw['extra_special'] = extra
    _DEC = LLaDADecoderContract(AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True), **kw)

def compile_row(args):
    ordinal, row = args
    dst = OUT / f'{ordinal:04d}_{row['key']}'
    if (dst / 'result.json').exists():
        return (ordinal, json.loads((dst / 'result.json').read_text())['status'])
    if os.environ.get('SKIP_STARTED') and dst.exists():
        return (ordinal, 'skip_started')
    dst.mkdir(parents=True, exist_ok=True)
    import importlib
    _name = os.environ.get('PREDICATES', 'v2')
    _name = {'v2': 'predicates_v2', 'ifb': 'predicates_ifb'}.get(_name, _name)
    PV = importlib.import_module(_name)
    from llada_decoder import compile_llada_event, save_graph, accepts
    from tess_branched_quotient_v1 import minimize
    t0 = time.perf_counter()
    levels = list(PV.components_ladder(row)) if hasattr(PV, 'components_ladder') else [(None, *PV.components_for(row))]
    ev = None
    last_err = None
    for lvl, comps, uns in levels:
        rec = dict(ordinal=ordinal, key=row['key'], supported=[[i, iid] for i, iid, _ in comps], ladder_level=lvl, unsupported=[[i, iid, m] for i, iid, m in uns], full_coverage=not uns, n_instructions=len(row['instruction_id_list']))
        if not comps:
            rec.update(status='no_supported', seconds=time.perf_counter() - t0)
            (dst / 'result.json').write_text(json.dumps(rec))
            return (ordinal, rec['status'])
        try:
            dfa = PV.product([PV.nonempty()] + [d for _, _, d in comps], max_states=200000)
            ev = compile_llada_event(_DEC, dfa, max_states=int(os.environ.get('MAX_STATES', 150000)))
            break
        except Exception as e:
            last_err = e
            ev = None
            continue
    try:
        if ev is None:
            raise last_err
        q, sc, gc, rounds = minimize(ev)
        save_graph(dst / 'graph.npz', q)
        rec.update(status='ok', text_states=len(dfa.states), product_states=ev.product_states, groups=len(ev.groups), quotient_states=len(q.accept), quotient_groups=len(q.groups))
    except Exception as e:
        rec.update(status='error', error=repr(e)[:300])
    rec['seconds'] = time.perf_counter() - t0
    (dst / 'result.json').write_text(json.dumps(rec))
    return (ordinal, rec['status'])
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.environ.get('ROWS_FILE', ROOT / 'assets/ifeval/data/input_data.jsonl'))]
    OUT.mkdir(exist_ok=True)
    with Pool(a.workers, initializer=init) as pool:
        for ordinal, status in pool.imap_unordered(compile_row, list(enumerate(rows))[::-1] if os.environ.get('REVERSE') else list(enumerate(rows))):
            print(time.strftime('%X'), ordinal, status, flush=True)
