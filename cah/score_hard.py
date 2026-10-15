import glob
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
from ifeval_official_runtime import load_rows, score, summarize
import predicates_hard as H
from loophole_detectors import flag

def load(d, field):
    out = {}
    for f in glob.glob(str(ROOT / 'cah' / d / '*.json')):
        r = json.load(open(f))
        out[str(r['key'])] = r.get(field) or ''
    return out

def main():
    import os
    rows = all_rows = load_rows()
    if os.environ.get('KEYS_FILE'):
        keep = set(open(os.environ['KEYS_FILE']).read().strip().split(','))
        rows = [r for r in rows if str(r['key']) in keep]
    byk = {str(r['key']): r for r in rows}
    cache = {}

    def hard_ok(r, j, text):
        k = (str(r['key']), j)
        if k not in cache:
            try:
                kw = {a: b for a, b in (r['kwargs'][j] or {}).items() if b is not None}
                cache[k] = [d.accepts_text for d in H.predicate(r['instruction_id_list'][j], kw)]
            except Exception:
                cache[k] = None
        fns = cache[k]
        return True if fns is None else all((f(text) for f in fns))
    print(f'{'arm':14s} official strict prompt/instr | hardened prompt/instr | detector-flagged share of official passes')
    for spec in sys.argv[1:]:
        name, rest = spec.split('=')
        resp = {}
        for part in reversed(rest.split('|')):
            d, field = part.split(':')
            resp.update({k: v for k, v in load(d, field).items() if v})
        s = score([dict(key=r['key'], prompt=r['prompt'], response=resp.get(str(r['key']), '')) for r in all_rows])
        recs = [x for x in s['strict']['records'] if str(x['key']) in byk]
        miss = sum((1 for k in byk if k not in resp))
        sm = summarize(recs)
        P = I = n = ni = flagged = passes = 0
        for x in recs:
            r = byk[str(x['key'])]
            t = resp.get(str(x['key']), '')
            oks = []
            for j, ok in enumerate(x['follow_instruction_list']):
                h = ok and hard_ok(r, j, t)
                oks.append(h)
                if ok:
                    passes += 1
                    flagged += bool(flag(r['instruction_id_list'][j], r['kwargs'][j], r['prompt'], t))
            n += 1
            ni += len(oks)
            P += all(oks)
            I += sum(oks)
        print(f'{name:14s} {100 * sm['prompt_accuracy']:5.1f} / {100 * sm['instruction_accuracy']:5.1f} | {100 * P / n:5.1f} / {100 * I / ni:5.1f} | {100 * flagged / max(1, passes):.1f}%  (n={n}, missing={miss})')
if __name__ == '__main__':
    main()
