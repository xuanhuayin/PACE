import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
from llada2_common import EOS_ID, PAD_ID, MASK_ID, VOCAB, SKIP_IDS, GEN_LENGTH, SNAP, MODEL_NAME
from llada2_engine import LLaDA2Engine, DECODING
TOPK = 2048
TAIL = 224

def tail_pattern(ids, e):
    after = ids[e + 1:] if e < len(ids) else []
    return dict(has_eos=e < len(ids), skip_in_answer=int(sum((t in SKIP_IDS for t in ids[:e]))), role_end_in_answer=156900 in ids[:e], after_eos_n=len(after), after_eos_non_pad=int(sum((t != PAD_ID for t in after))))

@torch.no_grad()
def capture(T, prompt_ids, gen_length=GEN_LENGTH):
    x, commits, info = T.run_sampler(prompt_ids, gen_length=gen_length, capture_topk=TOPK, continue_after_eos=TAIL)
    Lp = len(prompt_ids)
    ans = x[Lp:]
    e = ans.index(EOS_ID) if EOS_ID in ans else len(ans)
    missing = [p for p in range(Lp, Lp + gen_length) if p not in commits]
    assert all((p - Lp > e + TAIL - 1 for p in missing)), f'uncommitted positions inside eos+tail {missing[:10]}'
    ph_ids = np.concatenate([[EOS_ID], np.arange(TOPK - 1)]).astype(np.int32)
    ph_lp = np.full(TOPK, -60000.0, dtype=np.float16)
    ph_lp[0] = 0.0
    for p in missing:
        commits[p] = (ph_ids, ph_lp, 0.0, 0.0, -1)
    rows = [commits[p] for p in range(Lp, Lp + gen_length)]
    info['uncommitted'] = len(missing)
    return (ans, rows, info)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'cah/llada2_native_v1'))
    ap.add_argument('--gen', type=int, default=GEN_LENGTH)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--ids', default='all')
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    T = LLaDA2Engine()
    rows = [json.loads(l) for l in open(os.environ.get('ROWS_FILE', ROOT / 'assets/ifeval/data/input_data.jsonl'))]
    want = None if a.ids == 'all' else {x.strip() for x in (open(a.ids[1:]).read() if a.ids.startswith('@') else a.ids).replace('\n', ',').split(',') if x.strip()}
    k, n = map(int, a.shard.split('/'))
    print(f'{MODEL_NAME} native capture -> {OUT} | snap {SNAP} | decoding {DECODING} gen {a.gen}', flush=True)
    for ordinal, row in enumerate(rows):
        if ordinal % n != k:
            continue
        if want is not None and str(row['key']) not in want:
            continue
        dst = OUT / f'{ordinal:04d}_{row['key']}.json'
        if dst.exists():
            continue
        prompt = T.prompt_ids(row['prompt'])
        torch.manual_seed(0)
        t0 = time.perf_counter()
        ids, commit, info = capture(T, prompt, gen_length=a.gen)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        resp, eos = T.decode(ids)
        np.savez_compressed(OUT / f'{ordinal:04d}_{row['key']}.npz', ids=np.stack([c[0] for c in commit]).astype(np.int32), logp=np.stack([c[1] for c in commit]), eos_logp=np.array([c[2] for c in commit], dtype=np.float32), max_logp=np.array([c[3] for c in commit], dtype=np.float32), step=np.array([c[4] for c in commit], dtype=np.int16))
        dst.write_text(json.dumps(dict(key=row['key'], ordinal=ordinal, prefix_len=len(prompt), answer_ids=ids, response=resp, eos=eos, stored_positions=len(ids), topk=TOPK, t_native=dt, uncommitted=info['uncommitted'], nfe=info['nfe'], eos_block=info['eos_block'], tail=tail_pattern(ids, eos), settings=dict(gen_length=a.gen, **DECODING, eos_id=EOS_ID, pad_id=PAD_ID, mask_id=MASK_ID), snap=SNAP, model=MODEL_NAME)))
        print(time.strftime('%X'), ordinal, row['key'], 'eos', eos, f'{dt:.0f}s nfe {info['nfe']}', '|', resp[:80].replace('\n', ' / '), flush=True)
if __name__ == '__main__':
    main()
