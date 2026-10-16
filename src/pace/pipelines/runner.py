"""Run native generation, constraint compilation, or construct-then-certify inference."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from pace.paths import PROJECT_ROOT

MODELS = ("tess", "llada", "llada2", "dream")
BENCHMARKS = ("ifeval", "ifbench")
STAGES = ("native", "compile", "generate")


def stage_module(model, benchmark, stage):
    """Resolve a supported workflow without importing its model implementation."""
    if model not in MODELS or benchmark not in BENCHMARKS or stage not in STAGES:
        raise ValueError("Unsupported model, benchmark, or stage")
    if stage == "native":
        name = "native_ifbench" if model == "tess" and benchmark == "ifbench" else "native"
    elif stage == "compile":
        name = "compile"
    elif benchmark == "ifeval":
        name = "ifeval"
    elif model == "llada":
        name = "ifbench"
    else:
        return "pace.pipelines.ifbench"
    return f"pace.pipelines.{model}.{name}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="ifeval")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--checkpoint", help="public model ID or local checkpoint")
    parser.add_argument("--work-dir", default="outputs")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the command and paths without running it"
    )
    parser.add_argument(
        "extra", nargs=argparse.REMAINDER, help="arguments after -- go to the selected stage"
    )
    args = parser.parse_args(argv)
    base = Path(args.work_dir).expanduser().resolve() / args.model / args.benchmark
    native, graphs, output = base / "native", base / "graphs", base / "pace"
    config = dict(
        NATIVE_DIR=str(native),
        GRAPHS_DIR=str(graphs),
        GRAPH_OUT=str(graphs),
        GRAPHS_OUT=str(graphs),
        OUT_DIR=str(output),
        MAP_DIR="",
        NLTK_DATA=str(PROJECT_ROOT / "assets/nltk"),
        IFBL_CERT_WORKERS="0",
        IFB4_CERT_POOL="0",
        ROWS_FILE=str(
            PROJECT_ROOT
            / (
                "assets/ifeval/data/input_data.jsonl"
                if args.benchmark == "ifeval"
                else "assets/ifbench/data/IFBench_test.jsonl"
            )
        ),
        PREDICATES="pace.constraints.ifeval"
        if args.benchmark == "ifeval"
        else "pace.constraints.ifbench.acceptance",
    )
    if args.checkpoint:
        config[
            dict(tess="TESS_MODEL", llada="LLADA_SNAP", llada2="LLADA2_SNAP", dream="DREAM_SNAP")[
                args.model
            ]
        ] = args.checkpoint
    if args.benchmark == "ifbench" and args.stage == "generate":
        config["IFB4_MODEL"] = args.model
    flags = (
        ["--out", str(native if args.stage == "native" else output)]
        if args.stage != "compile"
        else []
    )
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    command = [
        sys.executable,
        "-m",
        stage_module(args.model, args.benchmark, args.stage),
        *flags,
        *extra,
    ]
    if args.dry_run:
        print(json.dumps(dict(command=command, configuration=config), indent=2))
        return 0
    for folder in (native, graphs, output):
        folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(config)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(command, cwd=PROJECT_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
