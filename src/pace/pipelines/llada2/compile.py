from pace.paths import PROJECT_ROOT
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT

from pace.models.llada2.config import SNAP, EOS_ID, MASK_ID, VOCAB, ADDED_IDS

os.environ["TOK_SNAP"] = SNAP
os.environ["EOS_ID"] = str(EOS_ID)
_PRED = os.environ.get("PREDICATES", "v2")
_DEFAULT_OUT = "outputs/llada2/ifeval/graphs"
os.environ.setdefault("GRAPH_OUT", str(ROOT / _DEFAULT_OUT))
import pace.pipelines.llada.compile as C


def init():
    from pace.models.llada2.config import contract

    C._DEC = contract()
    assert C._DEC.eos == EOS_ID and C._DEC.vocab == VOCAB
    assert C._DEC.bytes[126336] is not None and C._DEC.bytes[MASK_ID] is None
    assert sorted(C._DEC.special) == list(ADDED_IDS), len(C._DEC.special)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rows = [
        json.loads(l)
        for l in open(os.environ.get("ROWS_FILE", ROOT / "assets/ifeval/data/input_data.jsonl"))
    ]
    C.OUT.mkdir(exist_ok=True)
    print(
        "LLaDA2 compile ->",
        C.OUT,
        "predicates",
        _PRED,
        "max_states",
        os.environ.get("MAX_STATES"),
        "rows",
        len(rows),
        "tok",
        SNAP,
        "reverse",
        bool(os.environ.get("REVERSE")),
        flush=True,
    )
    jobs = list(enumerate(rows))
    if os.environ.get("REVERSE"):
        jobs = jobs[::-1]
    with Pool(a.workers, initializer=init) as pool:
        for ordinal, status in pool.imap_unordered(C.compile_row, jobs):
            print(time.strftime("%X"), ordinal, status, flush=True)
    print("COMPILE_DONE", flush=True)
