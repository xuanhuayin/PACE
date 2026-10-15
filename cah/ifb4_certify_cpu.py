import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'cah'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(C))
os.environ.setdefault('NLTK_DATA', str(Path(__file__).resolve().parents[1] / 'assets/nltk'))
GRAPHS = Path(os.environ.get('GRAPHS_DIR', str(C / 'ifbench_graphs_hard')))
NATIVE = C / 'ifbench_native_v1'
MAP_DIR = C / 'ifbench_map_hard'
OUT = Path(os.environ.get('OUT_DIR', str(C / 'ifbench_program_v4')))
PRE = Path(str(OUT) + '_pre')
LOGITS = Path(str(OUT) + '_logits')
FILL, V, EOS = (-10000.0, 32001, 2)
_W = {}

class _TokShim:
    """Minimal engine for ProgramIFB on CPU (tokenizer + decode; generation is never called here)."""

    def __init__(self, tok):
        self.tok = tok

    def generate_suffix(self, prompt, answer_prefix, seed, total=2048):
        raise RuntimeError('no generation in the CPU stage')

    def decode(self, ids):
        ids = list(ids)
        cut = ids.index(EOS) if EOS in ids else len(ids)
        return (self.tok.decode(ids[:cut], skip_special_tokens=True, clean_up_tokenization_spaces=False), cut)

def init():
    from transformers import AutoTokenizer
    from ifb_infill import Oracle
    from program_ifb4 import ProgramIFB
    snap = os.environ.get('TESS_MODEL', 'hamishivi/tess2')
    tok = AutoTokenizer.from_pretrained(snap, use_fast=True, local_files_only=False)
    T = _TokShim(tok)
    _W.update(T=T, P=ProgramIFB(T, Oracle()))

def certify_logits(P, T, row, text, arr, event, plan, max_resolves=2):
    from cah_tess import certify
    answer = [int(t) for t in arr['answer']]
    n_ids = int(arr['n_ids'])
    N = int(arr['N'])
    pad = T.tok.pad_token_id
    x = np.full((N, V), FILL)
    np.put_along_axis(x, arr['ids'][:N].astype(np.int64), arr['logp'][:N].astype(np.float64), axis=1)
    x[:n_ids - 1, EOS] = -10000.0
    x[:n_ids - 1, pad] = -10000.0
    t1 = time.perf_counter()
    reasons, resolves, rejected_text = ([], 0, None)
    while True:
        out = certify(x, event, fixed_ids=answer, fixed_mask=[False] * N, truncate=None)
        new = list(out) + answer[N:]
        cert_text = T.decode(new)[0]
        reasons, bad = P.guard(text, cert_text, plan)
        if not reasons:
            break
        rejected_text = cert_text
        if resolves >= max_resolves:
            break
        hit = 0
        for j in range(min(N, len(new))):
            if new[j] == answer[j]:
                continue
            piece = T.tok.decode([new[j]]).strip().lower()
            if len(piece) >= 2 and any((piece in w.lower() or w.lower() in piece for w in bad)):
                x[j, new[j]] = -10000.0
                hit += 1
        if not hit:
            for j in range(min(N, len(new))):
                if new[j] != answer[j]:
                    x[j, new[j]] = -10000.0
                    hit += 1
        resolves += 1
    if reasons:
        new, cert_text = (answer, text)
    return dict(text=cert_text, edits=sum((a != b for a, b in zip(new, answer))), cert_rejected=bool(reasons), reasons=reasons, resolves=resolves, rejected_text=rejected_text, map_seconds=time.perf_counter() - t1)

def job(args):
    ordinal, key = args
    from ifb_rows import rows
    from program_ifb4 import ifb_plan
    from ifbench_program_run4 import load_graph
    P, T = (_W['P'], _W['T'])
    dst = OUT / f'{ordinal:04d}_{key}.json'
    if dst.exists():
        return (ordinal, 'cached')
    t0 = time.perf_counter()
    row = rows()[ordinal]
    pre = json.loads((PRE / f'{ordinal:04d}_{key}.json').read_text())
    res = pre['res']
    text = res['text']
    native = json.loads((NATIVE / f'{ordinal:04d}_{key}.json').read_text())['response']
    rec = dict(key=key, ordinal=ordinal, program='ifb4', graphs=str(GRAPHS), pre_text=text, src=res['src'], applied=res['applied'], refused=res['refused'], candidates=res['candidates'], log=res['log'], weak=res['weak'], regens=res['regens'], gen_seconds=res['gen_seconds'], gen_calls=res['gen_calls'], pre_eval=res['ev'], graph_idx_assumed=pre.get('graph_idx_assumed'))
    try:
        grec, event = load_graph(GRAPHS / f'{ordinal:04d}_{key}')
        rec['graph_status'] = (grec or {}).get('status', 'missing')
        rec['ladder_level'] = (grec or {}).get('ladder_level')
        P._row = row
        plan = ifb_plan(row)
        mf = MAP_DIR / f'{ordinal:04d}_{key}.json'
        m = json.loads(mf.read_text()) if text == native and mf.exists() else None
        if m is not None and m.get('status') == 'ok' and m.get('map'):
            reasons, _ = P.guard(native, m['map'], plan)
            if not reasons:
                rec.update(response=m['map'], certified=True, cert_rejected=False, cert_reasons=[], cert_source='map_hard', edits=m.get('edits'))
            else:
                rec['map_rejected_reasons'] = reasons
        if 'response' not in rec:
            lf = LOGITS / f'{ordinal:04d}_{key}.npz'
            if event is None:
                rec.update(response=text, certified=False, cert_rejected=False, cert_source='none')
            elif not lf.exists():
                rec.update(response=text, certified=False, cert_rejected=False, cert_source='no_logits')
            else:
                with np.load(lf) as arr:
                    a = {k: arr[k] for k in arr.files}
                try:
                    cert = certify_logits(P, T, row, text, a, event, plan)
                    rec.update(response=cert['text'], certified=not cert['cert_rejected'], cert_rejected=cert['cert_rejected'], cert_reasons=cert['reasons'], cert_resolves=cert['resolves'], rejected_text=cert['rejected_text'], edits=cert['edits'], map_seconds=cert['map_seconds'], cert_source='program_cert')
                except Exception as e:
                    rec.update(response=text, certified=False, cert_rejected=False, cert_source='error', cert_error=repr(e)[:300])
        ev, unmet, tricked = P.final(row, rec['response'])
        rec.update(final_eval=ev, unmet=unmet, tricked=tricked, status='ok')
    except Exception as e:
        rec.update(status='error', error=repr(e)[:400], trace=traceback.format_exc()[-1500:])
        rec.setdefault('response', text)
    rec['cert_stage_seconds'] = time.perf_counter() - t0
    dst.write_text(json.dumps(rec))
    fe = rec.get('final_eval') or []
    return (ordinal, f'{rec['status']} src={rec['src']} cert={rec.get('cert_source')} rej={(rec.get('cert_reasons') or rec.get('map_rejected_reasons') or [])[:3]} official={[e['official'] for e in fe]} genuine={[e['genuine'] for e in fe]} {rec['cert_stage_seconds']:.0f}s {rec.get('error', '')}')

def refresh(f):
    """Recompute the final per-instruction evaluation of a written record with the current guards (records certified
    before a guard fix stay consistent with the scorer)."""
    from ifb_rows import rows
    r = json.loads(Path(f).read_text())
    if r.get('status') != 'ok':
        return (f, False)
    ev, unmet, tricked = _W['P'].final(rows()[r['ordinal']], r['response'])
    if ev == r.get('final_eval'):
        return (f, False)
    r.update(final_eval=ev, unmet=unmet, tricked=tricked, final_refreshed=time.strftime('%m-%d %H:%M'))
    Path(f).write_text(json.dumps(r))
    return (f, True)

def ready(ordinal, key, text_is_native):
    if (OUT / f'{ordinal:04d}_{key}.json').exists():
        return False
    if not (PRE / f'{ordinal:04d}_{key}.json').exists() or not (LOGITS / f'{ordinal:04d}_{key}.npz').exists():
        return False
    g = GRAPHS / f'{ordinal:04d}_{key}' / 'result.json'
    retry_done = (C / 'ifb4_retry.done').exists()
    if not g.exists():
        return retry_done
    st = json.loads(g.read_text())['status']
    if st == 'error' and (not retry_done):
        return False
    if text_is_native and (not (MAP_DIR / f'{ordinal:04d}_{key}.json').exists()) and (not retry_done):
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    from ifb_rows import rows
    R = rows()
    natives = {o: json.loads((NATIVE / f'{o:04d}_{r['key']}.json').read_text())['response'] for o, r in enumerate(R)}
    submitted = set()
    with Pool(a.workers, initializer=init) as pool:
        pending = []
        while True:
            for o, r in enumerate(R):
                k = str(r['key'])
                if o in submitted:
                    continue
                pf = PRE / f'{o:04d}_{k}.json'
                if not pf.exists():
                    continue
                tn = json.loads(pf.read_text())['res']['text'] == natives[o]
                if ready(o, k, tn):
                    submitted.add(o)
                    pending.append(pool.apply_async(job, ((o, k),)))
            still = []
            for p in pending:
                if p.ready():
                    o, msg = p.get()
                    print(time.strftime('%X'), o, msg, flush=True)
                else:
                    still.append(p)
            pending = still
            n_done = sum((1 for o, r in enumerate(R) if (OUT / f'{o:04d}_{r['key']}.json').exists()))
            if n_done >= len(R) or (a.once and (not pending)):
                break
            time.sleep(20)
        changed = sum((ch for _, ch in pool.map(refresh, sorted((str(x) for x in OUT.glob('*.json'))))))
    print('certification stage done', n_done, '| final evaluations refreshed:', changed, flush=True)
if __name__ == '__main__':
    main()
