import json
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / 'assets/ifbench/data/IFBench_test.jsonl'

def rows():
    return [json.loads(l) for l in open(DATA)]
