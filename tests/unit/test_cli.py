"""Validate workflow routing and package relocation without loading model weights."""

import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import re
import tempfile
import unittest

from pace.cli import main
from pace.paths import PROJECT_ROOT
from pace.pipelines.runner import BENCHMARKS, MODELS, STAGES, stage_module


class WorkflowTests(unittest.TestCase):
    def test_every_supported_workflow_has_an_entry_module(self):
        for model, benchmark, stage in itertools.product(MODELS, BENCHMARKS, STAGES):
            with self.subTest(model=model, benchmark=benchmark, stage=stage):
                self.assertIsNotNone(
                    importlib.util.find_spec(stage_module(model, benchmark, stage))
                )

    def test_dry_run_routes_ifbench_and_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "experiment"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                status = main(
                    [
                        "run",
                        "--model",
                        "dream",
                        "--benchmark",
                        "ifbench",
                        "--stage",
                        "generate",
                        "--work-dir",
                        str(work),
                        "--checkpoint",
                        "example/checkpoint",
                        "--dry-run",
                        "--",
                        "--ids",
                        "2,5",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertFalse(work.exists())
            result = json.loads(stream.getvalue())
            self.assertEqual(result["command"][1:3], ["-m", "pace.pipelines.ifbench"])
            self.assertEqual(result["command"][-2:], ["--ids", "2,5"])
            config = result["configuration"]
            self.assertEqual(config["DREAM_SNAP"], "example/checkpoint")
            self.assertEqual(config["IFB4_MODEL"], "dream")
            self.assertEqual(config["PREDICATES"], "pace.constraints.ifbench.acceptance")
            self.assertEqual(Path(config["GRAPHS_DIR"]), work / "dream/ifbench/graphs")
            self.assertTrue(Path(config["ROWS_FILE"]).is_relative_to(PROJECT_ROOT / "assets"))

    def test_invalid_workflow_is_rejected(self):
        with self.assertRaises(ValueError):
            stage_module("unknown", "ifeval", "generate")

    def test_capture_integrity_guard_matches_relocated_native_source(self):
        folder = PROJECT_ROOT / "src/pace/models/tess"
        capture = (folder / "capture.py").read_text()
        expected = re.search(r"NATIVE_SHA = [\'\"]([0-9a-f]{64})[\'\"]", capture).group(1)
        self.assertEqual(expected, hashlib.sha256((folder / "native.py").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
