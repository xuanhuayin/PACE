from pace.paths import PROJECT_ROOT
import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = PROJECT_ROOT


import os

OUT = Path(os.environ.get("GRAPHS_OUT", str(ROOT / "outputs/tess/ifeval/graphs")))
NATIVE = Path(os.environ.get("NATIVE_DIR", str(ROOT / "outputs/tess/ifeval/native")))
_DEC = None


def init():
    global _DEC
    from transformers import AutoTokenizer
    from pace.models.tess.decoder import TessDecoderContract

    snap = os.environ.get("TESS_MODEL", "hamishivi/tess2")
    tok = AutoTokenizer.from_pretrained(snap, use_fast=True, local_files_only=False)
    _DEC = TessDecoderContract(tok)


def save_graph(path, q):
    tokens = np.asarray([t for g in q.groups for t in g], dtype=np.int32)
    offsets = np.r_[0, np.cumsum([len(g) for g in q.groups])].astype(np.int32)
    np.savez_compressed(
        path,
        destination=q.destination,
        accept=q.accept,
        start=np.array(q.start, dtype=np.int32),
        token_group=q.token_group,
        group_tokens=tokens,
        group_offsets=offsets,
    )


def compile_row(args):
    ordinal, row, text_limit, graph_limit = args
    dst = OUT / f"{ordinal:04d}_{row['key']}"
    if (dst / "result.json").exists():
        return (ordinal, json.loads((dst / "result.json").read_text())["status"])
    dst.mkdir(parents=True, exist_ok=True)
    import importlib

    _PV = importlib.import_module(os.environ.get("PREDICATES", "pace.constraints.base"))
    components_for, nonempty, product = (_PV.components_for, _PV.nonempty, _PV.product)
    from pace.automata.context import compile_contextual_event
    from pace.automata.quotient import minimize

    t0 = time.perf_counter()
    levels = (
        list(_PV.components_ladder(row))
        if hasattr(_PV, "components_ladder")
        else [(None, *components_for(row))]
    )
    last_err = None
    for lvl, comps, unsupported in levels:
        rec = dict(
            ordinal=ordinal,
            key=row["key"],
            supported=[[i, iid] for i, iid, _ in comps],
            unsupported=[[i, iid, m] for i, iid, m in unsupported],
            ladder_level=lvl,
            full_coverage=not unsupported,
            n_instructions=len(row["instruction_id_list"]),
        )
        if not comps:
            rec.update(status="no_supported", seconds=time.perf_counter() - t0)
            (dst / "result.json").write_text(json.dumps(rec))
            return (ordinal, rec["status"])
        try:
            _dfa_try = product([nonempty()] + [d for _, _, d in comps], max_states=text_limit)
            _ev_try = compile_contextual_event(_DEC, _dfa_try, max_states=graph_limit)
            break
        except Exception as e:
            last_err = e
            continue
    try:
        if "_ev_try" not in locals():
            raise last_err
        dfa = _dfa_try
        rec["text_states"] = len(dfa.states)
        event, machine, states = _ev_try
        rec["graph_states"] = len(event.accept)
        rec["groups"] = len(event.groups)
        q, sc, gc, rounds = minimize(event)
        rec["quotient_states"] = len(q.accept)
        rec["quotient_groups"] = len(q.groups)
        rec["rounds"] = rounds
        save_graph(dst / "graph.npz", q)
        checks = []
        nat = sorted(NATIVE.glob(f"{ordinal:04d}_*.json"))
        rng = random.Random(ordinal)
        seqs = []
        if nat:
            ids = json.loads(nat[0].read_text())["answer_ids"]
            e = ids.index(2) if 2 in ids else len(ids)
            seqs.append(ids[: e + 1])
        for _ in range(6):
            seqs.append([rng.randrange(32001) for _ in range(rng.randint(1, 24))] + [2])
        for ids in seqs:
            text = _DEC.decode(ids)
            want = int(dfa.accepts_text(text))
            got = event.accepting_paths(ids)
            checks.append([want, got])
            if got != want:
                raise AssertionError(f"graph/text disagreement {want} vs {got}")
        rec["checks"] = checks
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = repr(e)[:400]
    rec["seconds"] = time.perf_counter() - t0
    (dst / "result.json").write_text(json.dumps(rec))
    return (ordinal, rec["status"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--text-limit", type=int, default=200000)
    ap.add_argument("--graph-limit", type=int, default=150000)
    ap.add_argument("--ids", default="all")
    a = ap.parse_args()
    from pace.models.tess.sampling import rows

    R = (
        [json.loads(l) for l in open(os.environ["ROWS_FILE"])]
        if os.environ.get("ROWS_FILE")
        else rows()
    )
    jobs = [
        (i, r, a.text_limit, a.graph_limit)
        for i, r in enumerate(R)
        if a.ids == "all"
        or r["key"] in {int(x) for x in a.ids.split(",")}
        or str(r["key"]) in set(a.ids.split(","))
    ]
    OUT.mkdir(exist_ok=True)
    with Pool(a.workers, initializer=init) as pool:
        for ordinal, status in pool.imap_unordered(compile_row, jobs):
            print(time.strftime("%X"), ordinal, status, flush=True)
