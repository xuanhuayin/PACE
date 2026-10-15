import numpy as np

class DecodedCountMAP:

    def __init__(self, compiled, k):
        if type(k) is not int or k < 0:
            raise ValueError('Nonnegative integer count required')
        groups = tuple((tuple(sorted(g)) for g in compiled.groups))
        destination = np.asarray(compiled.destination)
        increment = np.asarray(compiled.increment)
        final = np.asarray(compiled.final_increment)
        token_group = np.asarray(compiled.token_group)
        width = len(final)
        if not groups or not width or token_group.ndim != 1 or (destination.shape != (len(groups), width)) or (increment.shape != destination.shape) or (final.ndim != 1) or any((a.dtype.kind not in 'iu' for a in (destination, increment, final, token_group))) or (destination < 0).any() or (destination >= width).any() or (increment < 0).any() or (final < 0).any() or (not 0 <= compiled.start < width) or any((not g for g in groups)) or (sorted((t for g in groups for t in g)) != list(range(len(token_group)))) or any((token_group[t] != i for i, g in enumerate(groups) for t in g)):
            raise ValueError('Complete deterministic count graph required')
        self.groups = groups
        self.token_group = token_group.copy()
        self.start = int(compiled.start)
        self.width = width
        self.k = k
        self.destination = np.full(((k + 1) * width, len(groups)), -1, dtype=np.int64)
        self.accept = np.zeros((k + 1) * width, dtype=bool)
        for count in range(k + 1):
            for state in range(width):
                source = count * width + state
                self.accept[source] = count + int(final[state]) == k
                for group in range(len(groups)):
                    after = count + int(increment[group, state])
                    if after <= k:
                        self.destination[source, group] = after * width + destination[group, state]

    def solve(self, logits):
        values = np.asarray(logits)
        if values.ndim != 2 or not all(values.shape) or values.dtype != np.float64 or (values.shape[1] != len(self.token_group)) or np.isnan(values).any() or np.isposinf(values).any() or (not np.isfinite(values).any(axis=1).all()):
            raise ValueError('L,V FP64 logits with nonempty finite support required')
        maximum = values.max(axis=1, keepdims=True)
        log_probs = values - maximum
        log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=1, keepdims=True))
        length, vocabulary = values.shape
        group_scores = np.empty((length, len(self.groups)), dtype=np.float64)
        group_tokens = np.empty((length, len(self.groups)), dtype=np.int64)
        for group, ids in enumerate(self.groups):
            indices = np.asarray(ids)
            local = log_probs[:, indices]
            best = local.argmax(axis=1)
            group_scores[:, group] = local[np.arange(length), best]
            group_tokens[:, group] = indices[best]
        suffix = np.full((length + 1, len(self.accept)), -np.inf)
        suffix[-1, self.accept] = 0.0
        choices = np.full((length, len(self.accept)), -1, dtype=np.int64)
        valid = self.destination >= 0
        targets = self.destination.clip(min=0)
        for position in range(length - 1, -1, -1):
            edges = group_scores[position][None, :] + suffix[position + 1, targets]
            edges = np.where(valid, edges, -np.inf)
            best = edges.max(axis=1)
            suffix[position] = best
            token_candidates = np.where(np.isfinite(best[:, None]) & (edges == best[:, None]), group_tokens[position][None, :], vocabulary)
            selected = token_candidates.min(axis=1)
            choices[position] = np.where(selected < vocabulary, selected, -1)
        optimum = float(suffix[0, self.start])
        if not np.isfinite(optimum):
            raise ValueError('Condition has zero support; no fallback or repair')
        states = [self.start]
        tokens = []
        for position in range(length):
            token = int(choices[position, states[-1]])
            if token < 0:
                raise AssertionError('Missing optimal continuation')
            tokens.append(token)
            states.append(int(self.destination[states[-1], self.token_group[token]]))
        if not self.accept[states[-1]]:
            raise AssertionError('Optimal sequence did not accept')
        selected_score = float(log_probs[np.arange(length), tokens].sum())
        if not np.isclose(selected_score, optimum, rtol=1e-12, atol=1e-10):
            raise AssertionError('Path and dynamic-program scores disagree')
        return dict(token_ids=tokens, state_ids=states, log_probability=selected_score, optimum=optimum, unconstrained_log_probability=float(log_probs.max(axis=1).sum()))
