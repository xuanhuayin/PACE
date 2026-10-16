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

GRAPHS = Path(os.environ.get("GRAPHS_DIR", str(ROOT / "outputs/llada/ifbench/graphs")))
NATIVE = Path(os.environ.get("NATIVE_DIR", str(ROOT / "outputs/llada/ifbench/native")))
MAP_DIR = os.environ.get("MAP_DIR", str(ROOT / "outputs/llada/ifbench/certification"))
MODEL_NAME = os.environ.get("LLADA_MODEL_NAME", "LLaDA-8B-Instruct")


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


_CW = {}


def _cw_init():
    from transformers import AutoTokenizer
    from pace.construction.infill import Oracle
    from pace.models.llada.native import SNAP
    import pace.models.llada.engine as LE

    class _Shim:
        eos_id = LE.EOS
        pad_id = LE.EOS
        canvas_total = LE.LLaDAEngine.canvas_total

        def __init__(self, tok):
            self.tok = tok

        def generate_suffix(
            self, prompt, answer_prefix, seed, *, total=None, gen_length=256, block_length=16
        ):
            raise RuntimeError("no generation in the certification pool")

        decode = LE.LLaDAEngine.decode

    T = _Shim(AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True))
    _CW.update(T=T, P=ProgramIFB(T, Oracle(), re_t=0, gen_total=10**9))


def cert_tail(P, T, text, answer, ids, clamp, x, event, plan, t_re, max_resolves=2):
    """Program.certify_text after the logits (verbatim logic): x = logits[:N] (float64) with the no-shortening rule."""
    from pace.models.tess.engine import certify

    N = len(x)
    t1 = time.perf_counter()
    reasons, resolves, rejected_text = ([], 0, None)
    while True:
        out = certify(x, event, fixed_ids=answer, fixed_mask=clamp[:N], truncate=None)
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
            if new[j] == answer[j] or clamp[j]:
                continue
            piece = T.decode([new[j]])[0].strip().lower()
            if len(piece) >= 2 and any((piece in w.lower() or w.lower() in piece for w in bad)):
                x[j, new[j]] = -10000.0
                hit += 1
        if not hit:
            for j in range(min(N, len(new))):
                if new[j] != answer[j] and (not clamp[j]):
                    x[j, new[j]] = -10000.0
                    hit += 1
        resolves += 1
    if reasons:
        new = answer
        cert_text = text
    return dict(
        ids=new,
        text=cert_text,
        re_seconds=t_re,
        map_seconds=time.perf_counter() - t1,
        edits=sum((a != b for a, b in zip(new, answer))),
        cert_rejected=bool(reasons),
        reasons=reasons,
        resolves=resolves,
        rejected_text=rejected_text,
    )


def cert_logits(P, T, prefix, text, re_t, seed):
    """GPU part of Program.certify_text (verbatim; clamp_spans = [] as in the run): answer canvas + logits."""
    torch = T.torch
    eos = T.eos_id
    pad = T.pad_id
    ids = P.enc(text) + [eos]
    A = T.canvas_total - len(prefix)
    if len(ids) > A - 8:
        ids = ids[: A - 8]
    answer = ids + [pad] * (A - len(ids))
    clamp = [False] * A
    free = [not c for c in clamp]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = T.harmonize_batch(
        dict(prefix_ids=prefix), answer, [free], tstar=re_t, seeds=[seed + 999]
    )[0]
    torch.cuda.synchronize()
    t_re = time.perf_counter() - t0
    N = min(A, len(ids) + 160)
    return (answer, ids, clamp, logits[:N].float().numpy(), t_re)


def cert_job(args):
    rec, row, text, gdir, npy, answer, ids, clamp, t_re, dst = args
    P, T = (_CW["P"], _CW["T"])
    t0 = time.perf_counter()
    try:
        grec, event = load_graph(Path(gdir))
        p = ifb_plan(row)
        P._row = row
        x = np.load(npy).astype(np.float64)
        eos, pad = (T.eos_id, T.pad_id)
        x[: len(ids) - 1, eos] = -10000.0
        x[: len(ids) - 1, pad] = -10000.0
        try:
            cert = cert_tail(P, T, text, answer, ids, clamp, x, event, p, t_re)
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
        ev, unmet, tricked = P.final(row, rec["response"])
        rec.update(final_eval=ev, unmet=unmet, tricked=tricked, status="ok")
    except Exception as e:
        rec.update(status="error", error=repr(e)[:400], trace=traceback.format_exc()[-1500:])
        if "response" not in rec:
            rec["response"] = rec.get("pre_text") or ""
    rec["seconds"] = rec.get("seconds_gpu", 0) + time.perf_counter() - t0
    rec["cert_pool"] = True
    Path(dst).write_text(json.dumps(rec))
    try:
        os.unlink(npy)
    except OSError:
        pass
    fe = rec.get("final_eval") or []
    return f"{rec['ordinal']} {rec['key']} {rec['status']} {rec['seconds']:.0f}s | src {rec.get('src')} | cert {rec.get('certified')} {rec.get('cert_source')} {(rec.get('cert_reasons') or [])[:4]} | official {[e['official'] for e in fe]} genuine {[e['genuine'] for e in fe]} | {rec.get('error') or ''}"


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
    ap.add_argument("--out", default=str(ROOT / "outputs/llada/ifbench/pace"))
    ap.add_argument(
        "--wait-graphs",
        action="store_true",
        help="full stage: skip prompts whose automaton has no result.json yet",
    )
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
    from pace.construction.infill import Oracle
    from pace.data.ifbench import rows

    R = rows()
    k, n = map(int, a.shard.split("/"))
    ids = None if a.ids == "all" else set(a.ids.split(","))
    from pace.models.llada.engine import LLaDAEngine
    import pace.guards.ifbench as GD

    T = LLaDAEngine()
    O = Oracle()
    P = ProgramIFB(T, O, re_t=a.re_t, gen_total=10**9)
    print(
        f"program_ifb4 on {MODEL_NAME} ({os.environ.get('LLADA_SNAP') or 'default LLaDA-8B-Instruct snapshot'}) | graphs={GRAPHS} native={NATIVE} map={MAP_DIR or None} out={OUT} max_regens={P.max_regens} strict_canned={GD.STRICT_CANNED} eos={T.eos_id} canvas={T.canvas_total} stage={a.stage}",
        flush=True,
    )
    NW = int(os.environ.get("IFBL_CERT_WORKERS", 3)) if a.stage == "full" else 0
    pool = None
    pending = []
    TMP = Path(str(OUT) + "_certtmp")
    if NW:
        import multiprocessing as mp

        TMP.mkdir(exist_ok=True)
        pool = mp.get_context("spawn").Pool(NW, initializer=_cw_init)

    def drain(block=False):
        nonlocal pending
        still = []
        for j in pending:
            if block or j.ready():
                print(time.strftime("%X"), "pool", j.get(), flush=True)
            else:
                still.append(j)
        pending = still

    done = 0
    for ordinal, row in enumerate(R):
        if pool is not None:
            drain()
            while len(pending) >= 2 * NW:
                time.sleep(5)
                drain()
        if ordinal % n != k:
            continue
        if ids is not None and str(row["key"]) not in ids:
            continue
        dst = OUT / f"{ordinal:04d}_{row['key']}.json"
        pre_f = PRE / f"{ordinal:04d}_{row['key']}.json"
        if dst.exists() or (a.stage == "program" and pre_f.exists()):
            continue
        if a.limit and done >= a.limit:
            break
        if a.stage == "full" and a.wait_graphs:
            if not (GRAPHS / f"{ordinal:04d}_{row['key']}" / "result.json").exists():
                continue
            if not pre_f.exists():
                continue
            if MAP_DIR and (not (Path(MAP_DIR) / f"{ordinal:04d}_{row['key']}.json").exists()):
                _nat = json.loads((NATIVE / f"{ordinal:04d}_{row['key']}.json").read_text())[
                    "response"
                ]
                if json.loads(pre_f.read_text())["res"]["text"] == _nat:
                    continue
        t0 = time.perf_counter()
        p = ifb_plan(row)
        rec = dict(
            key=row["key"], ordinal=ordinal, program="ifb4", graphs=str(GRAPHS), model=MODEL_NAME
        )
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
                _pre = json.loads(pre_f.read_text())
                res = _pre["res"]
                rec["program_cached"] = True
                rec["graph_idx_assumed"] = _pre.get("graph_idx_assumed")
            else:
                gi = graph_idx
                if grec is None:
                    import pace.constraints.ifbench.acceptance as HB

                    gi = {i for i, _, _ in HB.components_for(row, generic=False)[0] if i >= 0}
                rec["graph_idx_assumed"] = sorted(gi)
                res = P.run(row, native_text, gi)
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
                if event is not None and pool is not None:
                    prefix = T.prompt_ids(row["prompt"])
                    rec["cert_truncated"] = len(P.enc(text)) + 1 > T.canvas_total - len(prefix) - 8
                    answer, cids, clamp, lg, t_re = cert_logits(P, T, prefix, text, P.re_t, P.seed)
                    npy = TMP / f"{ordinal:04d}_{row['key']}.npy"
                    np.save(npy, lg)
                    del lg
                    rec["seconds_gpu"] = time.perf_counter() - t0
                    pending.append(
                        pool.apply_async(
                            cert_job,
                            (
                                (
                                    rec,
                                    row,
                                    text,
                                    str(GRAPHS / f"{ordinal:04d}_{row['key']}"),
                                    str(npy),
                                    answer,
                                    cids,
                                    clamp,
                                    t_re,
                                    str(dst),
                                ),
                            ),
                        )
                    )
                    done += 1
                    continue
                if event is not None:
                    try:
                        prefix = T.prompt_ids(row["prompt"])
                        rec["cert_truncated"] = (
                            len(P.enc(text)) + 1 > T.canvas_total - len(prefix) - 8
                        )
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
    if pool is not None:
        drain(block=True)
        pool.close()
        pool.join()
    O.close()


if __name__ == "__main__":
    main()
