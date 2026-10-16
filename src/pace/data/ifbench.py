from pace.paths import PROJECT_ROOT
import json
from pathlib import Path

DATA = PROJECT_ROOT / "assets/ifbench/data/IFBench_test.jsonl"


def rows():
    return [json.loads(l) for l in open(DATA)]
