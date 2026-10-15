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
SNAP = os.environ.get('LLADA_SNAP', 'GSAI-ML/LLaDA-8B-Instruct')
MASK_ID = 126336

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, :remainder[i]] += 1
    return num

@torch.no_grad()
def generate_capture(model, prompt, gen_length=256, steps=256, block_length=16, topk=2048):
    """LLaDA generate (temperature 0, low-confidence remasking, cfg 0) + commit-time logits."""
    B, Lp = prompt.shape
    x = torch.full((B, Lp + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    commit_lp = [None] * gen_length
    num_blocks = gen_length // block_length
    steps_b = steps // num_blocks
    for nb in range(num_blocks):
        lo, hi = (Lp + nb * block_length, Lp + (nb + 1) * block_length)
        block_mask = x[:, lo:hi] == MASK_ID
        ntt = get_num_transfer_tokens(block_mask, steps_b)
        for i in range(steps_b):
            mask_index = x == MASK_ID
            logits = model(x).logits
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, hi:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, -np.inf)
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                _, sel = torch.topk(conf[j], k=int(ntt[j, i]))
                transfer[j, sel] = True
            lp = F.log_softmax(logits[0, :, :126349].float(), dim=-1)
            for pos in torch.nonzero(transfer[0]).flatten().tolist():
                row = lp[pos]
                tv, ti = torch.topk(row, topk)
                commit_lp[pos - Lp] = (ti.cpu().numpy().astype(np.int32), tv.cpu().numpy().astype(np.float16), float(row[126081]), float(row.max()))
            x[transfer] = x0[transfer]
    return (x[0, Lp:].cpu().tolist(), commit_lp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'cah/llada_native_v1'))
    ap.add_argument('--gen', type=int, default=256)
    ap.add_argument('--block', type=int, default=16)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--ids', default='all')
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModel.from_pretrained(SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    rows = [json.loads(l) for l in open(os.environ.get('ROWS_FILE', ROOT / 'assets/ifeval/data/input_data.jsonl'))]
    k, n = map(int, a.shard.split('/'))
    for ordinal, row in enumerate(rows):
        if ordinal % n != k:
            continue
        if a.ids != 'all' and row['key'] not in {int(x) for x in a.ids.split(',')}:
            continue
        dst = OUT / f'{ordinal:04d}_{row['key']}.json'
        if dst.exists():
            continue
        m = [{'role': 'user', 'content': row['prompt']}]
        text = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        prompt = tok(text)['input_ids']
        prompt = torch.tensor(prompt).unsqueeze(0).cuda()
        torch.manual_seed(0)
        t0 = time.perf_counter()
        ids, commit = generate_capture(model, prompt, gen_length=a.gen, steps=a.gen, block_length=a.block)
        dt = time.perf_counter() - t0
        eos = ids.index(126081) if 126081 in ids else len(ids)
        resp = tok.decode(ids[:eos], skip_special_tokens=True)
        N = len(ids)
        top_ids = np.stack([c[0] for c in commit])
        top_lp = np.stack([c[1] for c in commit])
        np.savez_compressed(OUT / f'{ordinal:04d}_{row['key']}.npz', ids=top_ids.astype(np.int32), logp=top_lp, eos_logp=np.array([c[2] for c in commit], dtype=np.float32), max_logp=np.array([c[3] for c in commit], dtype=np.float32))
        dst.write_text(json.dumps(dict(key=row['key'], ordinal=ordinal, prefix_len=int(prompt.shape[1]), answer_ids=ids, response=resp, eos=eos, stored_positions=N, topk=2048, t_native=dt, settings=dict(gen_length=a.gen, steps=a.gen, block_length=a.block, temperature=0), snap=SNAP)))
        print(time.strftime('%X'), ordinal, row['key'], 'eos', eos, f'{dt:.0f}s', '|', resp[:80].replace('\n', ' / '), flush=True)
if __name__ == '__main__':
    main()
