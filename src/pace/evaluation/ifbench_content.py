from pace.paths import PROJECT_ROOT
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = PROJECT_ROOT


import pace.evaluation.ifbench as O
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "ifbench_evaluation_lib", str(ROOT / "assets/ifbench/evaluation_lib.py")
)
E = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(E)
import pace.constraints.ifbench.acceptance as H
import pace.guards.ifbench as GD


def load(d, field):
    out = {}
    for f in glob.glob(str(ROOT / d / "*.json")):
        r = json.load(open(f))
        if r.get(field):
            out[str(r["key"])] = r[field]
    return out


def official_scores(rows, resp):
    inputs = [
        E.InputExample(
            key=r["key"],
            instruction_id_list=r["instruction_id_list"],
            prompt=r["prompt"],
            kwargs=[dict(k) for k in r["kwargs"]],
        )
        for r in rows
    ]
    p2r = {r["prompt"]: resp.get(r["key"], "") for r in rows}
    res = {}
    for name, fn in (
        ("strict", E.test_instruction_following_strict),
        ("loose", E.test_instruction_following_loose),
    ):
        outs = [fn(inp, p2r) for inp in inputs]
        res[name] = dict(
            prompt=sum((o.follow_all_instructions for o in outs)) / len(outs),
            instr=sum((sum(o.follow_instruction_list) for o in outs))
            / sum((len(o.follow_instruction_list) for o in outs)),
            records={i.key: list(o.follow_instruction_list) for i, o in zip(inputs, outs)},
        )
    return res


_CACHE = {}
NE = H.nonempty()


def hard_ok(r, j, text):
    k = (r["key"], j)
    if k not in _CACHE:
        iid = r["instruction_id_list"][j]
        kw = {a: b for a, b in (r["kwargs"][j] or {}).items() if b is not None}
        try:
            _CACHE[k] = (
                None
                if iid in H.PROGRAM_IDS
                else [d.accepts_text for d in H.predicate(iid, kw, r["prompt"])]
            )
        except Exception:
            _CACHE[k] = None
    fns = _CACHE[k]
    return True if fns is None else NE.accepts_text(text) and all((f(text) for f in fns))


def main():
    rows = [dict(r, key=str(r["key"])) for r in O.rows()]
    lines = []
    hdr = f"{'arm':12s} | official strict prompt / instr | official loose prompt / instr | official AND hardened prompt / instr | + guards prompt / instr | guard-flagged share of official instr passes | n / missing"
    print(hdr)
    lines.append(hdr)
    per_type = {}
    for spec in sys.argv[1:]:
        name, rest = spec.split("=", 1)
        resp = {}
        for part in reversed(rest.split("|")):
            d, field = part.split(":")
            resp.update(load(d, field))
        s = official_scores(rows, resp)
        P = I = PG = IG = n = ni = flagged = passes = 0
        pt = collections.defaultdict(lambda: [0, 0, 0])
        for r in rows:
            t = resp.get(r["key"], "")
            oks = s["strict"]["records"][r["key"]]
            tt = (
                GD.text_tags(t, r, lambda i, kw, x, p: O.official(i, kw, x, p))
                if t and any(oks)
                else {}
            )
            hs, gs = ([], [])
            for j, ok in enumerate(oks):
                h = bool(ok and hard_ok(r, j, t))
                hs.append(h)
                rel = GD.relevant(tt, j) if ok else {}
                g = h and (not rel)
                gs.append(g)
                if ok:
                    passes += 1
                    flagged += bool(rel)
                c = pt[r["instruction_id_list"][j]]
                c[0] += bool(ok)
                c[1] += h
                c[2] += g
            n += 1
            ni += len(oks)
            P += all(hs)
            I += sum(hs)
            PG += all(gs)
            IG += sum(gs)
        per_type[name] = pt
        miss = sum((1 for r in rows if not resp.get(r["key"])))
        line = f"{name:12s} | {100 * s['strict']['prompt']:5.1f} / {100 * s['strict']['instr']:5.1f} | {100 * s['loose']['prompt']:5.1f} / {100 * s['loose']['instr']:5.1f} | {100 * P / n:5.1f} / {100 * I / ni:5.1f} | {100 * PG / n:5.1f} / {100 * IG / ni:5.1f} | {100 * flagged / max(1, passes):4.1f}% | {n} / {miss}"
        print(line, flush=True)
        lines.append(line)
    lines += ["", "Per instruction type (strict): official / official AND hardened / + guards", ""]
    names = list(per_type)
    n_type = collections.Counter((i for r in rows for i in r["instruction_id_list"]))
    lines.append(f"{'type':34s} {'n':>3s} | " + " | ".join((f"{x:>14s}" for x in names)))
    for iid, cnt in sorted(n_type.items(), key=lambda x: (-x[1], x[0])):
        lines.append(
            f"{iid:34s} {cnt:3d} | "
            + " | ".join(
                (
                    f"{per_type[x][iid][0]:4d}/{per_type[x][iid][1]:4d}/{per_type[x][iid][2]:4d}"
                    for x in names
                )
            )
        )
    return lines


if __name__ == "__main__":
    out = main()
    import os

    if os.environ.get("SCORES_OUT"):
        Path(os.environ["SCORES_OUT"]).write_text("\n".join(out) + "\n")
