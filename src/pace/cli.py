"""Command-line interface for PACE research workflows."""

import argparse
import importlib
import sys

COMMANDS = {
    "run": ("pace.pipelines.runner", "Generate answers, compile constraints, or run PACE"),
    "evaluate": ("pace.evaluation.cli", "Evaluate official benchmark acceptance"),
    "setup-assets": ("pace.data.setup", "Fetch pinned public data and tokenizer resources"),
    "audit": ("pace.evaluation.audit", "Estimate fulfillment from supplied audit labels"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS, help="workflow to run")
    # Give each workflow its own parser and help text without importing GPU dependencies.
    if not argv or argv[0] not in COMMANDS:
        parser.parse_args(argv)
        return 0
    module = importlib.import_module(COMMANDS[argv[0]][0])
    return module.main(argv[1:])
