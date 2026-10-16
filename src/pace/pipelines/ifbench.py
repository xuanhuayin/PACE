from pace.paths import PROJECT_ROOT
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = PROJECT_ROOT


os.environ.setdefault("NLTK_DATA", str(PROJECT_ROOT / "assets/nltk"))
from pace.construction.ifbench import ProgramIFB, ifb_plan

MODEL = os.environ.get("IFB4_MODEL", "tess")
assert MODEL in ("tess", "dream", "llada2"), MODEL
_PFX = "" if MODEL == "tess" else MODEL + "_"
GRAPHS = Path(os.environ.get("GRAPHS_DIR", str(ROOT / f"outputs/{MODEL}/ifbench/graphs")))
NATIVE = Path(os.environ.get("NATIVE_DIR", str(ROOT / f"outputs/{MODEL}/ifbench/native")))
MAP_DIR = os.environ.get("MAP_DIR", str(ROOT / f"outputs/{MODEL}/ifbench/certification"))
MODEL_NAME = dict(tess="TESS-2", dream="Dream-v0-Instruct-7B", llada2="LLaDA2.0-mini")[MODEL]


def load_graph(gdir):
    """(result record, event) -- direct npz load (no 100k-state validation; large quotients are valid)."""
    if not (gdir / "result.json").exists():
        return (None, None)
    grec = json.loads((gdir / "result.json").read_text())
    if grec.get("status") != "ok" or not (gdir / "graph.npz").exists():
        return (grec, None)
    with np.load(gdir / "graph.npz", allow_pickle=False) as a:
        toks, offs = (a["group_tokens"], a["group_offsets"])
        ev = SimpleNamespace(
            destination=a["destination"],
            accept=a["accept"],
            start=int(a["start"]),
            token_group=a["token_group"],
            groups=tuple(
                (tuple(map(int, toks[offs[g] : offs[g + 1]])) for g in range(len(offs) - 1))
            ),
        )
    return (grec, ev)


def save_logits(T, row, text, path, topk=2048, tail=160, seed=999):
    """Certification inputs for the CPU stage (ifb4_certify_cpu.py): the program text re-encoded on the 2048 canvas and the
    top-K log-probabilities of one t*=0 harmonization pass (exactly Program.certify_text's logits: free answer, seed
    999) on positions [0, len(ids) + tail)."""
    import torch
    from pace.models.tess.prompts import format_prompt

    prefix = T.tok.encode(format_prompt(row["prompt"], T.tok.bos_token), add_special_tokens=False)
    eos, pad = (2, T.tok.pad_token_id)
    ids = T.tok.encode(text, add_special_tokens=False) + [eos]
    A = 2048 - len(prefix)
    if len(ids) > A - 8:
        ids = ids[: A - 8]
    answer = ids + [pad] * (A - len(ids))
    logits = T.harmonize_batch(
        dict(prefix_ids=prefix), answer, [[True] * A], tstar=0, seeds=[seed]
    )[0]
    N = min(A, len(ids) + tail)
    lp = torch.log_softmax(logits[:N].float(), -1)
    top = torch.topk(lp, topk, dim=-1)
    np.savez_compressed(
        path,
        ids=top.indices.numpy().astype(np.uint16),
        logp=top.values.numpy().astype(np.float16),
        answer=np.asarray(answer, dtype=np.int32),
        n_ids=len(ids),
        N=N,
    )


CERT_POOL = os.environ.get("IFB4_CERT_POOL", "0") == "1" and MODEL != "tess"


def save_cert_logits(P, T, prefix, text, path, tail=160):
    """[B-ifbD] exactly the GPU part of Program.certify_text (clamp_spans = [], tstar = P.re_t, seed P.seed + 999):
    the full (N, VOCAB) float32 re-encoding logits + the answer canvas; the MAP / guard / re-solve loop runs on CPU from these
    (ifbD_certify_pool.py) -- identical inputs, so identical results to the in-process path."""
    eos = getattr(T, "eos_id", 2)
    pad = getattr(T, "pad_id", P.tok.pad_token_id)
    ids = P.enc(text) + [eos]
    A = getattr(T, "canvas_total", 2048) - len(prefix)
    if len(ids) > A - 8:
        ids = ids[: A - 8]
    answer = ids + [pad] * (A - len(ids))
    free = [True] * A
    T.torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = T.harmonize_batch(
        dict(prefix_ids=prefix), answer, [free], tstar=P.re_t, seeds=[P.seed + 999]
    )[0]
    T.torch.cuda.synchronize()
    t_re = time.perf_counter() - t0
    N = min(A, len(ids) + tail)
    x = logits[:N].numpy().astype(np.float32)
    tmp = str(path) + ".tmp.npz"
    np.savez(
        tmp, x=x, answer=np.asarray(answer, dtype=np.int64), n_ids=len(ids), N=N, re_seconds=t_re
    )
    os.replace(tmp, path)


def load_map(ordinal, key):
    if not MAP_DIR:
        return None
    f = Path(MAP_DIR) / f"{ordinal:04d}_{key}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return r if r.get("status") == "ok" and r.get("map") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / f"outputs/{MODEL}/ifbench/pace"))
    ap.add_argument("--ids", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--re-t", type=int, default=0)
    ap.add_argument("--stage", default="full", choices=("program", "full"))
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(exist_ok=True, parents=True)
    PRE = Path(str(OUT) + "_pre")
    PRE.mkdir(exist_ok=True, parents=True)
    global CPEND
    CPEND = Path(str(OUT) + "_certpending")
    if CERT_POOL:
        CPEND.mkdir(exist_ok=True, parents=True)
    LOG = Path(str(OUT) + "_logits")
    if MODEL == "tess":
        LOG.mkdir(exist_ok=True, parents=True)
    from pace.construction.infill import Oracle
    from pace.data.ifbench import rows

    R = rows()
    k, n = map(int, a.shard.split("/"))
    ids = None if a.ids == "all" else set(a.ids.split(","))
    if MODEL == "tess":
        from pace.models.tess.engine import Tess
        from pace.models.tess.prompts import format_prompt

        T = Tess()
        O = Oracle()
        P = ProgramIFB(T, O, re_t=a.re_t)
        prefix_of = lambda prompt: T.tok.encode(
            format_prompt(prompt, T.tok.bos_token), add_special_tokens=False
        )
    else:
        if MODEL == "dream":
            from pace.models.dream.engine import DreamEngine as Engine
        else:
            from pace.models.llada2.engine import LLaDA2Engine as Engine
        T = Engine()
        O = Oracle()
        P = ProgramIFB(T, O, re_t=0, gen_total=10**9)
        prefix_of = T.prompt_ids
    print(
        f"program_ifb4 on {MODEL_NAME} | graphs={GRAPHS} native={NATIVE} map={MAP_DIR or None} out={OUT} max_regens={P.max_regens} eos={getattr(T, 'eos_id', 2)} canvas={getattr(T, 'canvas_total', 2048)}",
        flush=True,
    )
    done = 0
    for ordinal, row in enumerate(R):
        if ordinal % n != k:
            continue
        if ids is not None and str(row["key"]) not in ids:
            continue
        dst = OUT / f"{ordinal:04d}_{row['key']}.json"
        pre_f = PRE / f"{ordinal:04d}_{row['key']}.json"
        if dst.exists() or (a.stage == "program" and pre_f.exists()):
            continue
        if CERT_POOL and a.stage == "full" and (CPEND / f"{ordinal:04d}_{row['key']}.npz").exists():
            continue
        if a.limit and done >= a.limit:
            break
        t0 = time.perf_counter()
        p = ifb_plan(row)
        rec = dict(key=row["key"], ordinal=ordinal, program="ifb4", graphs=str(GRAPHS))
        if MODEL != "tess":
            rec["model"] = MODEL_NAME
        try:
            native_text = json.loads((NATIVE / f"{ordinal:04d}_{row['key']}.json").read_text())[
                "response"
            ]
            grec, event = load_graph(GRAPHS / f"{ordinal:04d}_{row['key']}")
            graph_idx = (
                {int(i) for i, iid in (grec or {}).get("supported", []) if int(i) >= 0}
                if event is not None
                else set()
            )
            rec["graph_status"] = (grec or {}).get("status", "missing")
            rec["graph_idx"] = sorted(graph_idx)
            if pre_f.exists():
                res = json.loads(pre_f.read_text())["res"]
                rec["program_cached"] = True
            else:
                gi = graph_idx
                if grec is None:
                    import pace.constraints.ifbench.acceptance as HB

                    gi = {i for i, _, _ in HB.components_for(row, generic=False)[0] if i >= 0}
                rec["graph_idx_assumed"] = sorted(gi)
                res = P.run(row, native_text, gi)
                lg = LOG / f"{ordinal:04d}_{row['key']}.npz"
                if MODEL == "tess" and (not lg.exists()):
                    save_logits(T, row, res["text"], lg)
                pre_f.write_text(
                    json.dumps(
                        dict(key=row["key"], ordinal=ordinal, graph_idx_assumed=sorted(gi), res=res)
                    )
                )
            if a.stage == "program":
                done += 1
                print(
                    time.strftime("%X"),
                    ordinal,
                    row["key"],
                    "program",
                    f"{time.perf_counter() - t0:.0f}s",
                    "| src",
                    res["src"],
                    "| applied",
                    res["applied"],
                    "| refused",
                    res["refused"],
                    "| genuine",
                    [e["genuine"] for e in res["ev"]],
                    "| calls",
                    res["gen_calls"],
                    flush=True,
                )
                continue
            text = res["text"]
            rec.update(
                pre_text=text,
                src=res["src"],
                applied=res["applied"],
                refused=res["refused"],
                candidates=res["candidates"],
                log=res["log"],
                weak=res["weak"],
                regens=res["regens"],
                gen_seconds=res["gen_seconds"],
                gen_calls=res["gen_calls"],
                pre_eval=res["ev"],
            )
            P._row = row
            m = load_map(ordinal, row["key"]) if text == native_text else None
            if m is not None:
                reasons, _ = P.guard(native_text, m["map"], p)
                if not reasons:
                    rec.update(
                        response=m["map"],
                        certified=True,
                        cert_rejected=False,
                        cert_reasons=[],
                        cert_source="map_hard",
                        edits=m.get("edits"),
                    )
                else:
                    rec["map_rejected_reasons"] = reasons
            if "response" not in rec:
                if event is not None:
                    try:
                        prefix = prefix_of(row["prompt"])
                        rec["cert_truncated"] = (
                            len(P.enc(text)) + 1
                            > getattr(T, "canvas_total", 2048) - len(prefix) - 8
                        )
                        if CERT_POOL:
                            save_cert_logits(
                                P, T, prefix, text, CPEND / f"{ordinal:04d}_{row['key']}.npz"
                            )
                            rec["cert_pending"] = True
                            (CPEND / f"{ordinal:04d}_{row['key']}.json").write_text(json.dumps(rec))
                            print(
                                time.strftime("%X"),
                                ordinal,
                                row["key"],
                                "cert_pending",
                                f"{time.perf_counter() - t0:.0f}s",
                                flush=True,
                            )
                            continue
                        cert = P.certify_text(dict(prefix_ids=prefix), text, [], event, plan=p)
                        rec.update(
                            response=cert["text"],
                            certified=not cert["cert_rejected"],
                            cert_rejected=cert["cert_rejected"],
                            cert_reasons=cert["reasons"],
                            cert_resolves=cert["resolves"],
                            rejected_text=cert["rejected_text"],
                            edits=cert["edits"],
                            re_seconds=cert["re_seconds"],
                            map_seconds=cert["map_seconds"],
                            cert_source="program_cert",
                        )
                    except Exception as e:
                        rec.update(
                            response=text,
                            certified=False,
                            cert_rejected=False,
                            cert_source="error",
                            cert_error=repr(e)[:300],
                        )
                else:
                    rec.update(
                        response=text, certified=False, cert_rejected=False, cert_source="none"
                    )
            ev, unmet, tricked = P.final(row, rec["response"])
            rec.update(final_eval=ev, unmet=unmet, tricked=tricked)
            rec["status"] = "ok"
        except Exception as e:
            if a.stage == "program":
                print(
                    time.strftime("%X"),
                    ordinal,
                    row["key"],
                    "program ERROR",
                    repr(e)[:300],
                    traceback.format_exc()[-800:],
                    flush=True,
                )
                continue
            rec.update(status="error", error=repr(e)[:400], trace=traceback.format_exc()[-1500:])
            if "response" not in rec:
                rec["response"] = rec.get("pre_text") or ""
        rec["seconds"] = time.perf_counter() - t0
        dst.write_text(json.dumps(rec))
        done += 1
        fe = rec.get("final_eval") or []
        print(
            time.strftime("%X"),
            ordinal,
            row["key"],
            rec["status"],
            f"{rec['seconds']:.0f}s",
            "| src",
            rec.get("src"),
            "| applied",
            rec.get("applied"),
            "| refused",
            rec.get("refused"),
            "| cert",
            rec.get("certified"),
            rec.get("cert_source"),
            (rec.get("cert_reasons") or "")[:4]
            if isinstance(rec.get("cert_reasons"), list)
            else "",
            "| official",
            [e["official"] for e in fe],
            "genuine",
            [e["genuine"] for e in fe],
            "| calls",
            rec.get("gen_calls"),
            "|",
            rec.get("error") or "",
            flush=True,
        )
    O.close()


if __name__ == "__main__":
    main()
