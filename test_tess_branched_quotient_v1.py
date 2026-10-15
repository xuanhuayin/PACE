import copy
import itertools
from types import SimpleNamespace
import unittest
import numpy as np
from test_automaton_conditional_sampling_v1 import fixture, paths
from tess_branched_numpy_reference import reference
from tess_branched_quotient_v1 import minimize, certify

class QuotientTests(unittest.TestCase):

    def duplicated(self):
        d = np.full((4, 2, 2), -1, dtype=np.int32)
        d[0, 0] = [1, 2]
        d[1, 1, 0] = 3
        d[2, 1, 0] = 3
        return SimpleNamespace(destination=d, accept=np.array([False, False, False, True]), start=0, token_group=np.array([0, 1]), groups=((0,), (1,)))

    def test_parallel_multiplicity_and_all_short_strings(self):
        for event in (self.duplicated(), fixture(False), fixture(True)):
            before = copy.deepcopy(event)
            q, sc, gc, _ = minimize(event)
            self.assertTrue(certify(event, q, sc, gc))
            for length in range(5):
                for ids in itertools.product(range(len(event.token_group)), repeat=length):
                    self.assertEqual(len(paths(event, ids)), len(paths(q, ids)))
            for name in ('destination', 'accept', 'token_group'):
                np.testing.assert_array_equal(getattr(event, name), getattr(before, name))
        event = self.duplicated()
        q, *_ = minimize(event)
        self.assertEqual(len(q.accept), 3)
        self.assertEqual(len(paths(q, [0, 1])), 2)

    def test_full_token_probability_and_gradient(self):
        rng = np.random.default_rng(20260928)
        for event in (self.duplicated(), fixture(False), fixture(True)):
            q, _, _, _ = minimize(event)
            for length in (2, 3, 7):
                logits = rng.normal(size=(length, len(event.token_group)))
                a, ga = reference(logits, event.destination[:, event.token_group], event.accept, event.start)
                b, gb = reference(logits, q.destination[:, q.token_group], q.accept, q.start)
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
                np.testing.assert_allclose(ga, gb, rtol=0, atol=1e-12)

    def test_idempotence_and_transition_group_merge(self):
        event = self.duplicated()
        event.destination = np.repeat(event.destination, 2, axis=1)
        event.groups = ((0,), (1,), (2,), (3,))
        event.token_group = np.arange(4)
        a, *_ = minimize(event)
        b, *_ = minimize(a)
        self.assertEqual(len(a.groups), 2)
        self.assertEqual(a.destination.shape, b.destination.shape)
        for ids in itertools.product(range(4), repeat=3):
            self.assertEqual(len(paths(event, ids)), len(paths(b, ids)))

    def test_certificate_rejects_mutations(self):
        event = self.duplicated()
        q, sc, gc, _ = minimize(event)
        for field in ('start', 'accept', 'destination', 'token_group'):
            bad = copy.deepcopy(q)
            if field == 'start':
                bad.start = (q.start + 1) % len(q.accept)
            elif field == 'accept':
                bad.accept[0] = not bad.accept[0]
            elif field == 'destination':
                bad.destination[q.start, 0, 0] = -1
            else:
                bad.token_group = bad.token_group[::-1].copy()
            with self.assertRaises(ValueError):
                certify(event, bad, sc, gc)

    def test_uniform_acceptance_and_dead_sentinels(self):
        for accepting in (False, True):
            event = SimpleNamespace(destination=np.array([[[1, -1]], [[1, -1]]], dtype=np.int32), accept=np.array([accepting, accepting]), start=0, groups=((0,),), token_group=np.array([0]))
            q, sc, gc, _ = minimize(event)
            self.assertEqual(len(q.accept), 1)
            self.assertTrue(certify(event, q, sc, gc))
            self.assertEqual(q.destination[0, 0].tolist(), [-1, 0])
if __name__ == '__main__':
    unittest.main()
