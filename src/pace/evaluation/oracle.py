from pace.paths import PROJECT_ROOT
from pathlib import Path
import json
import os
import sys

os.environ.setdefault("NLTK_DATA", str(PROJECT_ROOT / "assets/nltk"))
_reply = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr


import pace.evaluation.ifbench as O
import pace.construction.transforms as TR


def handle(req):
    op = req.get("op")
    if op == "ping":
        return {"ok": True}
    if op == "list":
        return {"transforms": sorted(TR.TRANSFORMS)}
    if op == "check":
        return {
            "ok": O.official(
                req["iid"], O.clean_kwargs(req.get("kw")), req["text"], req.get("prompt")
            )
        }
    if op == "checks":
        return {
            "ok": {
                i: O.official(i, O.clean_kwargs(kw), req["text"], req.get("prompt"))
                for i, kw in req["pairs"]
            }
        }
    if op == "transform":
        text = TR.TRANSFORMS[req["iid"]](
            req["text"], O.clean_kwargs(req.get("kw")), req.get("prompt")
        )
        return {"text": text, "last": dict(TR.LAST)}
    return {"error": "unknown op %r" % op}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = handle(json.loads(line))
        except Exception as e:
            out = {"error": repr(e)[:500]}
        _reply.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
