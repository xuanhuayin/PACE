import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
from dream_common import EOS_ID, PAD_ID, IM_END, MASK_ID, VOCAB
from dream_engine import DreamEngine, DECODING, CONF_TEMPERATURE
TOPK = 2048

def tail_pattern(ids, e):
    """Diagnostics of what fills the canvas after the answer."""
    after = ids[e + 1:] if e < len(ids) else []
    return dict(has_eos=e < len(ids), im_end_in_answer=IM_END in ids[:e], mask_in_answer=MASK_ID in ids[:e], after_eos_n=len(after), after_eos_non_pad=int(sum((t != PAD_ID for t in after))))

@torch.no_grad()
def capture(T, prompt_ids, gen_length=256):
    x, commits = T.run_sampler(torch.tensor([prompt_ids], device='cuda'), gen_length=gen_length, capture_topk=TOPK)
    Lp = len(prompt_ids)
    missing = [p for p in range(Lp, Lp + gen_length) if p not in commits]
    if missing:
        raw = T.model(x).logits[0]
        lp = F.log_softmax(raw[torch.tensor(missing, device='cuda') - 1, :VOCAB].float(), dim=-1)
        tv, ti = torch.topk(lp, TOPK, dim=-1)
        for r, p in enumerate(missing):
            commits[p] = (ti[r].cpu().numpy().astype(np.int32), tv[r].cpu().numpy().astype(np.float16), float(lp[r, EOS_ID]), float(lp[r].max()), -1)
    rows = [commits[p] for p in range(Lp, Lp + gen_length)]
    return (x[0, Lp:].tolist(), rows, len(missing))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'cah/dream_native_v1'))
    ap.add_argument('--gen', type=int, default=256)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--ids', default='all')
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    T = DreamEngine()
    rows = [json.loads(l) for l in open(os.environ.get('ROWS_FILE', ROOT / 'assets/ifeval/data/input_data.jsonl'))]
    want = None if a.ids == 'all' else {x.strip() for x in a.ids.split(',')}
    k, n = map(int, a.shard.split('/'))
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
        ids, commit, n_missing = capture(T, prompt, gen_length=a.gen)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        resp, eos = T.decode(ids)
        np.savez_compressed(OUT / f'{ordinal:04d}_{row['key']}.npz', ids=np.stack([c[0] for c in commit]).astype(np.int32), logp=np.stack([c[1] for c in commit]), eos_logp=np.array([c[2] for c in commit], dtype=np.float32), max_logp=np.array([c[3] for c in commit], dtype=np.float32), step=np.array([c[4] for c in commit], dtype=np.int16))
        dst.write_text(json.dumps(dict(key=row['key'], ordinal=ordinal, prefix_len=len(prompt), answer_ids=ids, response=resp, eos=eos, stored_positions=len(ids), topk=TOPK, t_native=dt, uncommitted=n_missing, tail=tail_pattern(ids, eos), settings=dict(max_new_tokens=a.gen, steps=a.gen, **DECODING, conf_temperature=CONF_TEMPERATURE, eos_id=EOS_ID, pad_id=PAD_ID))))
        print(time.strftime('%X'), ordinal, row['key'], 'eos', eos, f'{dt:.0f}s', '|', resp[:80].replace('\n', ' / '), flush=True)
if __name__ == '__main__':
    main()
