import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from pace.models.tess.engine import ROOT, read, Tess

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "outputs/tess/ifbench/native"))
ap.add_argument("--topk", type=int, default=2048)
ap.add_argument("--tail", type=int, default=256)
ap.add_argument("--shard", default="0/1")
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
from pace.data.ifbench import rows

R = rows()
k, n = map(int, a.shard.split("/"))
T = Tess()
torch = T.torch
for ordinal, row in enumerate(R):
    if ordinal % n != k:
        continue
    dst = OUT / f"{ordinal:04d}_{row['key']}.json"
    if dst.exists():
        continue
    entry = dict(index=row["key"], prompt=row["prompt"], seed=202610020000 + ordinal)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = T.native(entry)
    torch.cuda.synchronize()
    t_nat = time.perf_counter() - t0
    nat = out["native"]["answer_ids"]
    P = len(out["native"]["prefix_ids"])
    logits = out["final_logits"][0, P:].float()
    del out
    eos = nat.index(2) if 2 in nat else len(nat)
    N = min(len(nat), eos + a.tail)
    lp = torch.log_softmax(logits[:N], -1)
    top = torch.topk(lp, a.topk, dim=-1)
    np.savez_compressed(
        OUT / f"{ordinal:04d}_{row['key']}.npz",
        ids=top.indices.numpy().astype(np.uint16),
        logp=top.values.numpy().astype(np.float16),
        eos_logp=lp[:, 2].numpy().astype(np.float32),
        max_logp=lp.max(-1).values.numpy().astype(np.float32),
    )
    dst.write_text(
        json.dumps(
            dict(
                key=row["key"],
                ordinal=ordinal,
                seed=entry["seed"],
                prefix_len=P,
                answer_ids=nat,
                response=T.decode(nat)[0],
                eos=eos,
                stored_positions=N,
                topk=a.topk,
                t_native=t_nat,
            )
        )
    )
    print(time.strftime("%X"), ordinal, row["key"], "eos", eos, f"{t_nat:.0f}s", flush=True)
