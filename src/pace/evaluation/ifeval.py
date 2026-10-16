from pace.paths import PROJECT_ROOT
from contextlib import contextmanager
import copy
import hashlib
import importlib
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = PROJECT_ROOT
SOURCE = ROOT / "assets/ifeval"
ASSETS = ROOT / "assets/nltk"
SOURCE_SHA = "7de366f158a592823ce02e6ac9e134eb16d34e9cc988bc846061552f8a47cf80"
ASSETS_SHA = "e85d309d6dbf963f97c31af5b471a49592cc8e4b24f7d8a7c151f50d40d70835"
_ACTIVE = False


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official():
    if sha(SOURCE / "sources.json") != SOURCE_SHA:
        raise ValueError("Official source manifest changed")
    manifest = json.loads((SOURCE / "sources.json").read_text())
    for name, metadata in manifest["files"].items():
        if sha(SOURCE / name) != metadata["sha256"]:
            raise ValueError("Official source changed: " + name)
    name = "instruction_following_eval"
    if name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(SOURCE)]
        sys.modules[name] = module
    elif list(sys.modules[name].__path__) != [str(SOURCE)]:
        raise ValueError("Conflicting IFEval package already loaded")
    lib = importlib.import_module(name + ".evaluation_lib")
    for leaf in ("evaluation_lib", "instructions_registry", "instructions", "instructions_util"):
        if Path(sys.modules[name + "." + leaf].__file__).resolve() != SOURCE / (leaf + ".py"):
            raise ValueError("Wrong official module loaded")
    return lib


def check_assets():
    path = ROOT / "assets/scoring_assets.json"
    if sha(path) != ASSETS_SHA:
        raise ValueError("Scoring asset manifest changed")
    manifest = json.loads(path.read_text())
    if manifest["source_manifest_sha256"] != SOURCE_SHA:
        raise ValueError("Scoring assets bind different official sources")
    for name, wanted in manifest["packages"].items():
        if version(name) != wanted:
            raise ValueError("Scoring dependency version differs: " + name)
    for name, digest in manifest["nltk_english"].items():
        if sha(ASSETS / name) != digest:
            raise ValueError("Sentence tokenizer data differs")
    import langdetect

    package = Path(langdetect.__file__).parent
    for name, digest in manifest["language_profiles"].items():
        if sha(package / name) != digest:
            raise ValueError("Language profile differs")


@contextmanager
def scoring_context():
    global _ACTIVE
    if _ACTIVE:
        raise ValueError("Nested/concurrent scoring unsupported")
    check_assets()
    lib = official()
    import nltk
    import langdetect.detector_factory as detector

    factory = detector._factory
    paths = list(nltk.data.path)
    state = random.getstate()
    _ACTIVE = True
    try:
        detector._factory = detector.DetectorFactory()
        detector._factory.load_profile(detector.PROFILES_DIRECTORY)
        detector._factory.seed = 0
        nltk.data.path[:] = [str(ASSETS)]
        nltk.data.clear_cache()
        nltk.tokenize._get_punkt_tokenizer.cache_clear()
        lib.instructions_registry.instructions.instructions_util._get_sentence_tokenizer.cache_clear()
        random.seed(0)
        yield lib
    finally:
        random.setstate(state)
        detector._factory = factory
        nltk.data.path[:] = paths
        nltk.data.clear_cache()
        nltk.tokenize._get_punkt_tokenizer.cache_clear()
        lib.instructions_registry.instructions.instructions_util._get_sentence_tokenizer.cache_clear()
        _ACTIVE = False


def load_rows():
    official()
    rows = [
        json.loads(line) for line in (SOURCE / "data/input_data.jsonl").read_text().splitlines()
    ]
    if (
        len(rows) != 541
        or len({r["key"] for r in rows}) != 541
        or len({r["prompt"] for r in rows}) != 541
    ):
        raise ValueError("Official input grid differs")
    return rows


def align(rows, responses):
    expected = {r["key"]: r for r in rows}
    if len(expected) != len(rows) or len({r["prompt"] for r in rows}) != len(rows):
        raise ValueError("Duplicate reference key/prompt")
    aligned = {}
    for record in responses:
        if (
            type(record.get("key")) is not int
            or record["key"] not in expected
            or record["key"] in aligned
        ):
            raise ValueError("Duplicate/unknown response key")
        if record.get("prompt") != expected[record["key"]]["prompt"] or not isinstance(
            record.get("response"), str
        ):
            raise ValueError("Prompt/response mismatch")
        aligned[record["key"]] = record["response"]
    if set(aligned) != set(expected):
        raise ValueError("Missing responses; no denominator reduction")
    return {r["prompt"]: aligned[r["key"]] for r in rows}


def summarize(records):
    if not records:
        raise ValueError("Empty scoring population")
    total = sum((len(r["instruction_id_list"]) for r in records))
    if not total:
        raise ValueError("No instructions")
    for record in records:
        flags = record["follow_instruction_list"]
        if (
            not flags
            or len(flags) != len(record["instruction_id_list"])
            or any((type(v) is not bool for v in flags))
        ):
            raise ValueError("Instruction result shape/type")
        if type(record["follow_all_instructions"]) is not bool or record[
            "follow_all_instructions"
        ] != all(flags):
            raise ValueError("Prompt result differs")
    pc = sum((r["follow_all_instructions"] for r in records))
    ic = sum((sum(r["follow_instruction_list"]) for r in records))
    return dict(
        prompt_correct=pc,
        prompt_total=len(records),
        prompt_accuracy=pc / len(records),
        instruction_correct=ic,
        instruction_total=total,
        instruction_accuracy=ic / total,
    )


def score(responses):
    rows = load_rows()
    aligned = align(rows, responses)
    output = {}
    with scoring_context() as lib:
        for mode in ("strict", "loose"):
            records = []
            fn = getattr(lib, "test_instruction_following_" + mode)
            for row in rows:
                inp = lib.InputExample(**copy.deepcopy(row))
                result = fn(inp, aligned)
                records.append(
                    dict(
                        key=row["key"],
                        instruction_id_list=result.instruction_id_list,
                        follow_all_instructions=bool(result.follow_all_instructions),
                        follow_instruction_list=[bool(v) for v in result.follow_instruction_list],
                    )
                )
            output[mode] = dict(summary=summarize(records), records=records)
    return output
