import unittest
from unittest.mock import patch
import numpy as np
from types import SimpleNamespace
import tests.unit.test_token_map as dense_tests
from pace.automata.token_map import BranchedTokenMAP
from pace.automata.sparse_token_map import SparseBranchedTokenMAP


class SparseExhaustiveTests(dense_tests.BranchedMAPTests):
    def setUp(self):
        self.patch = patch.object(dense_tests, "BranchedTokenMAP", SparseBranchedTokenMAP)
        self.patch.start()
        self.addCleanup(self.patch.stop)


class SparseDenseTests(unittest.TestCase):
    def test_all_edges_and_exact_results(self):
        rng = np.random.default_rng(202609291100)
        for states, groups, length in [(4, 3, 4), (17, 9, 23), (71, 23, 64)]:
            for case in range(8):
                destination = rng.integers(-1, states, size=(states, groups, 2))
                destination[rng.random(destination.shape) < 0.7] = -1
                if case == 0:
                    destination[:] = -1
                event = SimpleNamespace(
                    destination=destination,
                    accept=rng.integers(0, 2, size=states).astype(bool),
                    start=0,
                    groups=tuple(((2 * g, 2 * g + 1) for g in range(groups))),
                    token_group=np.repeat(np.arange(groups), 2),
                )
                sparse = SparseBranchedTokenMAP(event)
                src, grp, branch = np.nonzero(destination >= 0)
                for actual, expected in [
                    (sparse.source, src),
                    (sparse.group, grp),
                    (sparse.branch, branch),
                    (sparse.target, destination[src, grp, branch]),
                ]:
                    np.testing.assert_array_equal(actual, expected)
                for logits in [
                    rng.normal(size=(length, 2 * groups)),
                    np.zeros((length, 2 * groups)),
                ]:
                    before = logits.copy()
                    try:
                        expected = BranchedTokenMAP(event).solve(logits)
                    except ValueError as error:
                        self.assertIn("zero support", str(error))
                        with self.assertRaisesRegex(ValueError, "zero support"):
                            sparse.solve(logits)
                    else:
                        self.assertEqual(expected, sparse.solve(logits))
                        self.assertEqual(expected, sparse.solve(logits))
                    np.testing.assert_array_equal(logits, before)
                    np.testing.assert_array_equal(sparse.destination, destination)


if __name__ == "__main__":
    unittest.main()
