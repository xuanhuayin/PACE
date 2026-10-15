import json
from pathlib import Path
import tempfile
import unittest

from evaluate import load_responses, summarize_flags
from scripts.audit_metrics import estimate


class EvaluationTests(unittest.TestCase):
    def test_explicit_empty_pace_output_overrides_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native, pace = root / "native", root / "pace"
            native.mkdir()
            pace.mkdir()
            (native / "one.json").write_text(json.dumps(dict(key=1, response="Native answer")))
            (native / "two.json").write_text(json.dumps(dict(key=2, response="Untouched answer")))
            (pace / "one.json").write_text(json.dumps(dict(key=1, response="")))
            merged = load_responses(native)
            merged.update(load_responses(pace))
            self.assertEqual(merged, {"1": "", "2": "Untouched answer"})

    def test_mixed_requirement_counts_keep_both_denominators(self):
        result = summarize_flags([
            dict(follow_instruction_list=[True, False]),
            dict(follow_instruction_list=[True]),
            dict(follow_instruction_list=[False, False, False]),
        ])
        self.assertEqual(result["prompt_total"], 3)
        self.assertEqual(result["instruction_total"], 6)
        self.assertAlmostEqual(result["prompt_accuracy_pct"], 100 / 3)
        self.assertAlmostEqual(result["instruction_accuracy_pct"], 100 / 3)

    def test_duplicate_response_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ["first", "second"]:
                (folder / f"{name}.json").write_text(json.dumps(dict(key=1, response="Answer")))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_responses(folder)


class AuditTests(unittest.TestCase):
    def test_audit_adjustment_can_reverse_official_ranking(self):
        data = dict(arms={
            "certification": dict(official_passes=90, official_total=100, draws=[
                dict(labels=["GENUINE", "LOOPHOLE"]), dict(labels=["LOOPHOLE", "LOOPHOLE"]),
            ]),
            "construction": dict(official_passes=60, official_total=100, draws=[
                dict(labels=["GENUINE", "GENUINE"]), dict(labels=["GENUINE", "GENUINE"]),
            ]),
        }, comparisons=[["construction", "certification"]])
        output = estimate(data, resamples=100)
        self.assertEqual(output["arms"]["certification"]["fulfillment_pct"], 22.5)
        self.assertEqual(output["arms"]["construction"]["fulfillment_pct"], 60)
        self.assertEqual(output["comparisons"]["construction - certification"]["difference_points"], 37.5)

    def test_retained_draws_are_not_deduplicated_by_shared_item_id(self):
        data = dict(arms={"method": dict(official_passes=4, official_total=8, draws=[
            dict(id="shared", labels=["GENUINE", "GENUINE"]),
            dict(id="shared", labels=["GENUINE", "GENUINE"]),
            dict(id="other", labels=["LOOPHOLE", "LOOPHOLE"]),
        ])})
        result = estimate(data, resamples=100)["arms"]["method"]
        self.assertEqual(result["audit_draws"], 3)
        self.assertAlmostEqual(result["fulfillment_pct"], 100 / 3)

    def test_invalid_official_counts_fail(self):
        with self.assertRaises(ValueError):
            estimate(dict(arms={"invalid": dict(official_passes=11, official_total=10, draws=[])}))


if __name__ == "__main__":
    unittest.main()

