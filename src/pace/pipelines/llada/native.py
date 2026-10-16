import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from pace.paths import PROJECT_ROOT as ROOT
from pace.models.llada.native import SNAP, generate_capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/llada/ifeval/native"))
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ids", default="all")
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    rows = [
        json.loads(l)
        for l in open(os.environ.get("ROWS_FILE", ROOT / "assets/ifeval/data/input_data.jsonl"))
    ]
    k, n = map(int, a.shard.split("/"))
    for ordinal, row in enumerate(rows):
        if ordinal % n != k:
            continue
        if a.ids != "all" and row["key"] not in {int(x) for x in a.ids.split(",")}:
            continue
        dst = OUT / f"{ordinal:04d}_{row['key']}.json"
        if dst.exists():
            continue
        m = [{"role": "user", "content": row["prompt"]}]
        text = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        prompt = tok(text)["input_ids"]
        prompt = torch.tensor(prompt).unsqueeze(0).cuda()
        torch.manual_seed(0)
        t0 = time.perf_counter()
        ids, commit = generate_capture(
            model, prompt, gen_length=a.gen, steps=a.gen, block_length=a.block
        )
        dt = time.perf_counter() - t0
        eos = ids.index(126081) if 126081 in ids else len(ids)
        resp = tok.decode(ids[:eos], skip_special_tokens=True)
        N = len(ids)
        top_ids = np.stack([c[0] for c in commit])
        top_lp = np.stack([c[1] for c in commit])
        np.savez_compressed(
            OUT / f"{ordinal:04d}_{row['key']}.npz",
            ids=top_ids.astype(np.int32),
            logp=top_lp,
            eos_logp=np.array([c[2] for c in commit], dtype=np.float32),
            max_logp=np.array([c[3] for c in commit], dtype=np.float32),
        )
        dst.write_text(
            json.dumps(
                dict(
                    key=row["key"],
                    ordinal=ordinal,
                    prefix_len=int(prompt.shape[1]),
                    answer_ids=ids,
                    response=resp,
                    eos=eos,
                    stored_positions=N,
                    topk=2048,
                    t_native=dt,
                    settings=dict(
                        gen_length=a.gen, steps=a.gen, block_length=a.block, temperature=0
                    ),
                    snap=SNAP,
                )
            )
        )
        print(
            time.strftime("%X"),
            ordinal,
            row["key"],
            "eos",
            eos,
            f"{dt:.0f}s",
            "|",
            resp[:80].replace("\n", " / "),
            flush=True,
        )


if __name__ == "__main__":
    main()
