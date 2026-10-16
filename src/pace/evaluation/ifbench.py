from pace.paths import PROJECT_ROOT
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NLTK_DATA", str(PROJECT_ROOT / "assets/nltk"))
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / "assets/ifbench"))
from ifbench import instructions_registry

DATA = ROOT / "assets/ifbench/data/IFBench_test.jsonl"


def rows():
    return [json.loads(l) for l in open(DATA)]


def clean_kwargs(kw):
    return {k: v for k, v in (kw or {}).items() if v is not None}


def checker(iid, kw, prompt=None):
    inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
    inst.build_description(**clean_kwargs(kw))
    args = inst.get_instruction_args()
    if args and "prompt" in args:
        inst.build_description(prompt=prompt)
    return inst


def official(iid, kw, text, prompt=None):
    inst = checker(iid, kw, prompt)
    try:
        return bool(text and text.strip() and inst.check_following(text))
    except Exception:
        return False
