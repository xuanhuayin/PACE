import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
from program_gen4 import Program, plan, needs_program, needs_program3
GRAPHS = Path(os.environ.get('GRAPHS_DIR', str(ROOT / 'cah/ifeval_graphs_v2')))
NATIVE = Path(os.environ.get('NATIVE_DIR', str(ROOT / 'cah/ifeval_native_v1')))
MAP_DIR = os.environ.get('MAP_DIR', str(ROOT / 'cah/ifeval_map_hard'))

def load_graph(gdir):
    """Graph record + event from a graphs dir entry (official or hardened compile; result.json optional)."""
    from compile_tess_ifeval_quotients_v1 import to_event
    if not (gdir / 'graph.npz').exists():
        return (None, None)
    grec = json.loads((gdir / 'result.json').read_text()) if (gdir / 'result.json').exists() else dict(status='ok')
    if grec.get('status', 'ok') != 'ok':
        return (grec, None)
    with np.load(gdir / 'graph.npz', allow_pickle=False) as arr:
        return (grec, to_event({x: arr[x] for x in arr.files}))

def load_map(ordinal, key):
    if not MAP_DIR:
        return None
    f = Path(MAP_DIR) / f'{ordinal:04d}_{key}.json'
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return r if r.get('status') == 'ok' and r.get('map') else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'cah/ifeval_program_v4'))
    ap.add_argument('--ids', default='all')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--re-t', type=int, default=0)
    ap.add_argument('--graphs', default=None, help='overrides GRAPHS_DIR')
    a = ap.parse_args()
    graphs = Path(a.graphs) if a.graphs else GRAPHS
    OUT = Path(a.out)
    OUT.mkdir(exist_ok=True, parents=True)
    from cah_tess import Tess
    from tess_ifeval_baseline import rows
    from tess_ifeval_prompt_contract import format_prompt
    R = [json.loads(l) for l in open(os.environ['ROWS_FILE'])] if os.environ.get('ROWS_FILE') else rows()
    k, n = map(int, a.shard.split('/'))
    T = Tess()
    P = Program(T, re_t=a.re_t)
    print(f'program_gen4 on TESS-2 | graphs={graphs} native={NATIVE} map={MAP_DIR or None} out={OUT}', flush=True)
    done = 0
    for ordinal, row in enumerate(R):
        if ordinal % n != k:
            continue
        if a.ids != 'all' and row['key'] not in {int(x) for x in a.ids.split(',')}:
            continue
        p = plan(row)
        if not needs_program(p):
            continue
        dst = OUT / f'{ordinal:04d}_{row['key']}.json'
        if dst.exists():
            continue
        if a.limit and done >= a.limit:
            break
        t0 = time.perf_counter()
        guard_only = not needs_program3(p)
        rec = dict(key=row['key'], ordinal=ordinal, plan=p, program='gen4', graphs=str(graphs), guard_only=guard_only)
        try:
            nat = NATIVE / f'{ordinal:04d}_{row['key']}.json'
            native_text = json.loads(nat.read_text())['response'] if nat.exists() else None
            res = P.run(row, native_text)
            rec.update(pre_text=res['text'], log=res['log'], gen_seconds=res['gen_seconds'], clamp_spans=res['clamp_spans'], unmet=res['unmet'], weak=res['weak'], regens=res['regens'], g4_final=res.get('g4_final', []), gen_calls=res.get('gen_calls'))
            m = load_map(ordinal, row['key']) if guard_only and native_text is not None and (res['text'] == native_text) else None
            reasons = None
            if m is not None:
                reasons, _ = P.guard(native_text, m['map'], p)
                if not reasons:
                    rec.update(response=m['map'], certified=True, cert_rejected=False, cert_reasons=[], cert_source='map_hard', edits=m.get('edits'))
                else:
                    rec['map_rejected_reasons'] = reasons
            if 'response' not in rec:
                grec, event = load_graph(graphs / f'{ordinal:04d}_{row['key']}')
                if event is not None:
                    prefix = T.tok.encode(format_prompt(row['prompt'], T.tok.bos_token), add_special_tokens=False)
                    cert = P.certify_text(dict(prefix_ids=prefix), res['text'], res['clamp_spans'], event, plan=p)
                    rec.update(response=cert['text'], certified=not cert['cert_rejected'], cert_rejected=cert['cert_rejected'], cert_reasons=cert['reasons'], cert_resolves=cert['resolves'], rejected_text=cert['rejected_text'], edits=cert['edits'], re_seconds=cert['re_seconds'], map_seconds=cert['map_seconds'], cert_source='program_cert')
                else:
                    rec.update(response=res['text'], certified=False, cert_rejected=False, cert_source='none', graph_status=(grec or {}).get('status', 'missing'))
            rec['status'] = 'ok'
        except Exception as e:
            rec.update(status='error', error=repr(e)[:400], trace=traceback.format_exc()[-1500:])
            if 'response' not in rec:
                rec['response'] = rec.get('pre_text', '')
        rec['seconds'] = time.perf_counter() - t0
        dst.write_text(json.dumps(rec))
        done += 1
        g4n = sum((1 for w in rec.get('weak') or [] if str(w).startswith('g4:')))
        print(time.strftime('%X'), ordinal, row['key'], rec['status'], f'{rec['seconds']:.0f}s', '| unmet', rec.get('unmet'), '| g4', g4n, '| cert', rec.get('certified'), rec.get('cert_source'), rec.get('cert_reasons') or '', '|', (rec.get('log') or [])[-2:], '|', rec['response'][:70].replace('\n', ' / '), flush=True)
if __name__ == '__main__':
    main()
