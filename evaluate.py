"""Evaluate native or PACE outputs without changing the benchmark denominator."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def load_responses(folder):
    """Read one-record JSON files. An explicitly empty answer remains empty."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Response directory does not exist: {folder}")
    result = {}
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text())
        if "key" not in record or not isinstance(record.get("response"), str):
            raise ValueError(f"Expected a key and string response in {path.name}")
        key = str(record["key"])
        if key in result:
            raise ValueError(f"Duplicate response key: {key}")
        result[key] = record["response"]
    return result


def summarize_flags(records):
    flags = [record["follow_instruction_list"] for record in records]
    if not flags or any(not row for row in flags):
        raise ValueError("A nonempty set of instruction results is required")
    if any(type(value) is not bool for row in flags for value in row):
        raise ValueError("Instruction results must be Boolean")
    prompt_correct = sum(all(row) for row in flags)
    instruction_correct = sum(sum(row) for row in flags)
    instruction_total = sum(map(len, flags))
    return dict(
        prompt_correct=prompt_correct,
        prompt_total=len(flags),
        prompt_accuracy_pct=100 * prompt_correct / len(flags),
        instruction_correct=instruction_correct,
        instruction_total=instruction_total,
        instruction_accuracy_pct=100 * instruction_correct / instruction_total,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tess", "llada", "llada2", "dream"), required=True)
    parser.add_argument("--benchmark", choices=("ifeval", "ifbench"), default="ifeval")
    parser.add_argument("--method", choices=("native", "pace"), default="pace")
    parser.add_argument("--split", choices=("heldout", "all"))
    parser.add_argument("--work-dir", default="outputs")
    parser.add_argument("--response-dir", type=Path, help="Score this directory instead of merging stage outputs")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args(argv)
    split = args.split or ("heldout" if args.benchmark == "ifeval" else "all")
    if args.benchmark == "ifbench" and split != "all":
        parser.error("IFBench is evaluated on the full benchmark; use --split all")

    base = Path(args.work_dir).resolve() / args.model / args.benchmark
    try:
        if args.response_dir:
            responses = load_responses(args.response_dir)
        else:
            responses = load_responses(base / "native")
            if args.method == "pace":
                responses.update(load_responses(base / "pace"))

        if args.benchmark == "ifeval":
            import ifeval_official_runtime as runtime
            rows = runtime.load_rows()
            scored = runtime.score([
                dict(key=row["key"], prompt=row["prompt"], response=responses.get(str(row["key"]), ""))
                for row in rows
            ])
            keep = set(str(row["key"]) for row in rows)
            if split == "heldout":
                keep = set((ROOT / "configs/ifeval_heldout_keys.txt").read_text().strip().split(","))
                assert len(keep) == 341
            metrics = {
                mode: summarize_flags([record for record in result["records"] if str(record["key"]) in keep])
                for mode, result in scored.items()
            }
        else:
            sys.path.insert(0, str(ROOT / "cah"))
            from score_hard_ifb import official_scores
            import ifb_official
            rows = [dict(row, key=str(row["key"])) for row in ifb_official.rows()]
            keep = set(row["key"] for row in rows)
            scored = official_scores(rows, responses)
            metrics = {
                mode: summarize_flags([
                    dict(follow_instruction_list=flags) for flags in result["records"].values()
                ]) for mode, result in scored.items()
            }

        output = dict(
            model=args.model, benchmark=args.benchmark, method=args.method, split=split,
            evaluated_prompts=len(keep),
            missing_response_records=sum(key not in responses for key in keep),
            empty_answers=sum(not responses.get(key, "").strip() for key in keep),
            metrics=metrics,
        )
        encoded = json.dumps(output, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Evaluation error: {error}\nRun python setup_assets.py before scoring.\n")


if __name__ == "__main__":
    main()

