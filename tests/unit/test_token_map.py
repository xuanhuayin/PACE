import copy
import itertools
from types import SimpleNamespace
import unittest
import numpy as np
from pace.automata.token_map import BranchedTokenMAP
from tests.fixtures.automata import fixture, paths


def exhaustive(event, values):
    lp = values - np.logaddexp.reduce(values, axis=1)[:, None]
    accepted = []
    for ids in itertools.product(range(values.shape[1]), repeat=len(values)):
        if paths(event, ids):
            score = float(lp[np.arange(len(values)), ids].sum())
            if np.isfinite(score):
                accepted.append((score, ids))
    return None if not accepted else sorted(accepted, key=lambda x: (-x[0], x[1]))[0]


class BranchedMAPTests(unittest.TestCase):
    def verify(self, event, values):
        before = values.copy()
        graph = event.destination.copy()
        solver = BranchedTokenMAP(event)
        oracle = exhaustive(event, values)
        if oracle is None:
            with self.assertRaisesRegex(ValueError, "zero support"):
                solver.solve(values)
            return
        actual = solver.solve(values)
        self.assertEqual(actual["token_ids"], list(oracle[1]))
        self.assertAlmostEqual(actual["optimum"], oracle[0], places=11)
        self.assertAlmostEqual(actual["log_probability"], oracle[0], places=11)
        self.assertTrue(paths(event, actual["token_ids"]))
        for i, token in enumerate(actual["token_ids"]):
            self.assertEqual(
                actual["state_ids"][i + 1],
                event.destination[
                    actual["state_ids"][i], event.token_group[token], actual["branches"][i]
                ],
            )
        self.assertTrue(event.accept[actual["state_ids"][-1]])
        self.assertEqual(actual, solver.solve(values))
        np.testing.assert_array_equal(values, before)
        np.testing.assert_array_equal(event.destination, graph)

    def test_unambiguous_and_ambiguous_full_enumeration(self):
        rng = np.random.default_rng(202609290700)
        for ambiguous in (False, True):
            for length in (1, 2, 3, 4):
                self.verify(fixture(ambiguous), rng.normal(size=(length, 4)))
                self.verify(fixture(ambiguous), np.zeros((length, 4), dtype=np.float64))

    def test_random_graphs_exhaustive_feasibility_and_optimum(self):
        rng = np.random.default_rng(202609290701)
        for _ in range(24):
            event = SimpleNamespace(
                destination=rng.integers(-1, 4, size=(4, 3, 2)),
                accept=rng.integers(0, 2, size=4).astype(bool),
                start=0,
                token_group=np.array([0, 0, 1, 2]),
                groups=((0, 1), (2,), (3,)),
            )
            self.verify(event, rng.normal(size=(3, 4)))
            self.verify(event, np.zeros((3, 4), dtype=np.float64))

    def test_equal_first_token_requires_suffix_order_not_branch_id(self):
        event = fixture(False)
        event.destination[1, 1, 0] = -1
        event.destination[1, 2, 0] = 3
        event.destination[2, 2, 0] = -1
        event.destination[2, 1, 0] = 3
        result = BranchedTokenMAP(event).solve(np.zeros((2, 4), dtype=np.float64))
        self.assertEqual(result["token_ids"], [0, 2])
        self.assertEqual(result["branches"][0], 1)
        self.verify(event, np.zeros((2, 4), dtype=np.float64))

    def test_duplicate_accepting_paths_do_not_reweight_strings(self):
        destination = np.full((2, 2, 2), -1, dtype=np.int64)
        destination[0, 0] = [1, 1]
        destination[0, 1, 0] = 1
        event = SimpleNamespace(
            destination=destination,
            accept=np.array([False, True]),
            start=0,
            token_group=np.array([0, 1]),
            groups=((0,), (1,)),
        )
        values = np.log(np.array([[0.4, 0.6]], dtype=np.float64))
        self.assertEqual(BranchedTokenMAP(event).solve(values)["token_ids"], [1])
        self.verify(event, values)

    def test_deterministic_count_bridge_and_eos_tail(self):
        from pace.automata.lexical.compiler import compile_event
        from pace.automata.counting import DecodedCountMAP

        rng = np.random.default_rng(202609290702)
        graph = compile_event("word", ("", " word", "x"), 0)
        for count in (0, 1, 2):
            old = DecodedCountMAP(graph, count)
            destination = np.full((*old.destination.shape, 2), -1, dtype=np.int64)
            destination[:, :, 0] = old.destination
            event = SimpleNamespace(
                destination=destination,
                accept=old.accept,
                start=old.start,
                token_group=old.token_group,
                groups=old.groups,
            )
            for values in (
                rng.normal(size=(4, 3)),
                np.zeros((4, 3)),
                np.array([[0.0, 10.0, -100.0], [20.0, 0.0, -100.0], [-100.0, 50.0, 0.0]]),
            ):
                try:
                    expected = old.solve(values)
                except ValueError:
                    with self.assertRaises(ValueError):
                        BranchedTokenMAP(event).solve(values)
                    continue
                actual = BranchedTokenMAP(event).solve(values)
                self.assertEqual(actual["token_ids"], expected["token_ids"])
                self.assertAlmostEqual(
                    actual["log_probability"], expected["log_probability"], places=11
                )
                if count == 1 and len(values) == 3:
                    self.assertEqual(actual["token_ids"], [1, 0, 1])

    def test_extreme_masked_and_shifted_logits(self):
        event = fixture()
        values = np.array([[0.0, -2000.0, -np.inf, 0.0], [0.0, 0.0, -2200.0, -2000.0]])
        result = BranchedTokenMAP(event).solve(values)
        shifted = BranchedTokenMAP(event).solve(values + 10000.0)
        self.assertEqual(result["token_ids"], shifted["token_ids"])
        self.assertAlmostEqual(result["log_probability"], shifted["log_probability"], places=10)
        self.verify(event, values)
        values[0, :2] = -np.inf
        with self.assertRaisesRegex(ValueError, "zero support"):
            BranchedTokenMAP(event).solve(values)

    def test_invalid_contracts_and_explicit_memory_rejection(self):
        event = fixture()
        solver = BranchedTokenMAP(event)
        good = np.zeros((2, 4), dtype=np.float64)
        for bad in (
            good.astype(np.float32),
            np.zeros((0, 4)),
            good[:, :3],
            np.full_like(good, np.inf),
            np.full_like(good, np.nan),
            np.full_like(good, -np.inf),
        ):
            with self.assertRaises(ValueError):
                solver.solve(bad)
        for mutate in (
            lambda e: setattr(e, "start", True),
            lambda e: setattr(e, "start", 4),
            lambda e: setattr(e, "groups", ((0, 1), (1, 2), (3,))),
            lambda e: setattr(e, "groups", ((0.0, 1), (2,), (3,))),
            lambda e: e.destination.__setitem__((0, 0, 0), 4),
        ):
            bad = copy.deepcopy(event)
            mutate(bad)
            with self.assertRaises(ValueError):
                BranchedTokenMAP(bad)
        with self.assertRaises(MemoryError):
            BranchedTokenMAP(event, max_trace_bytes=31).solve(good)
        self.assertEqual(BranchedTokenMAP(event, max_trace_bytes=32).solve(good)["trace_bytes"], 32)


if __name__ == "__main__":
    unittest.main()
