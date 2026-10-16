import numpy as np
from pace.automata.token_map import BranchedTokenMAP


class SparseBranchedTokenMAP(BranchedTokenMAP):
    def __init__(self, event, max_trace_bytes=4 * 1024**3):
        super().__init__(event, max_trace_bytes)
        self.source, self.group, self.branch = np.nonzero(self.destination >= 0)
        self.target = self.destination[self.source, self.group, self.branch]
        self.edge_code = self.group * 2 + self.branch
        self.active, self.offset = np.unique(self.source, return_index=True)

    def _reduce(self, values, operation, fill):
        result = np.full(len(self.accept), fill, dtype=values.dtype)
        if len(self.offset):
            result[self.active] = operation.reduceat(values, self.offset)
        return result

    def solve(self, logits):
        values = np.asarray(logits)
        if (
            values.ndim != 2
            or not all(values.shape)
            or values.dtype != np.float64
            or (values.shape[1] != len(self.token_group))
            or np.isnan(values).any()
            or np.isposinf(values).any()
            or (not np.isfinite(values).any(axis=1).all())
        ):
            raise ValueError(
                "L,V FP64 logits with nonempty finite support at every position required"
            )
        length, vocabulary = values.shape
        states, groups = (len(self.accept), len(self.groups))
        trace_bytes = length * states * np.dtype(np.int32).itemsize
        if trace_bytes > self.max_trace_bytes:
            raise MemoryError(
                "Full trace exceeds allocation ceiling; no graph or length truncation"
            )
        lp = values - values.max(axis=1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))
        group_scores = np.empty((length, groups), dtype=np.float64)
        group_tokens = np.empty((length, groups), dtype=np.int64)
        for group, ids in enumerate(self.groups):
            tokens = np.asarray(ids, dtype=np.int64)
            local = lp[:, tokens]
            best = local.argmax(axis=1)
            group_scores[:, group] = local[np.arange(length), best]
            group_tokens[:, group] = tokens[best]
        value = np.where(self.accept, 0.0, -np.inf)
        rank = np.where(self.accept, 0, states).astype(np.int64)
        choices = np.full((length, states), -1, dtype=np.int32)
        for position in range(length - 1, -1, -1):
            edges = group_scores[position, self.group] + value[self.target]
            best = self._reduce(edges, np.maximum, -np.inf)
            possible = np.isfinite(best)
            optimal = possible[self.source] & (edges == best[self.source])
            tokens = group_tokens[position, self.group]
            selected_token = self._reduce(
                np.where(optimal, tokens, vocabulary), np.minimum, vocabulary
            )
            optimal &= tokens == selected_token[self.source]
            successor_rank = rank[self.target]
            selected_rank = self._reduce(
                np.where(optimal, successor_rank, states), np.minimum, states
            )
            optimal &= successor_rank == selected_rank[self.source]
            choice = self._reduce(
                np.where(optimal, self.edge_code, groups * 2), np.minimum, groups * 2
            )
            choices[position] = np.where(possible, choice, -1)
            next_rank = np.full(states, states, dtype=np.int64)
            if possible.any():
                keys = np.stack((selected_token[possible], selected_rank[possible]), axis=1)
                _, inverse = np.unique(keys, axis=0, return_inverse=True)
                next_rank[possible] = inverse
            value, rank = (best, next_rank)
        optimum = float(value[self.start])
        if not np.isfinite(optimum):
            raise ValueError("Condition has zero support; no fallback or repair")
        tokens, state_ids, branches = ([], [self.start], [])
        for position in range(length):
            code = int(choices[position, state_ids[-1]])
            if code < 0:
                raise AssertionError("Missing optimal continuation")
            group, branch = divmod(code, 2)
            tokens.append(int(group_tokens[position, group]))
            branches.append(branch)
            target = int(self.destination[state_ids[-1], group, branch])
            if target < 0:
                raise AssertionError("Invalid selected branch")
            state_ids.append(target)
        if not self.accept[state_ids[-1]]:
            raise AssertionError("Optimal sequence did not accept")
        score = float(lp[np.arange(length), tokens].sum())
        if not np.isclose(score, optimum, rtol=1e-12, atol=1e-10):
            raise AssertionError("Path and dynamic-program scores disagree")
        return dict(
            token_ids=tokens,
            state_ids=state_ids,
            branches=branches,
            log_probability=score,
            optimum=optimum,
            unconstrained_log_probability=float(lp.max(axis=1).sum()),
            trace_bytes=trace_bytes,
        )
