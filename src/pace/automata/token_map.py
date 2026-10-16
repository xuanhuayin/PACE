import numpy as np


class BranchedTokenMAP:
    def __init__(self, event, max_trace_bytes=4 * 1024**3):
        if type(max_trace_bytes) is not int or max_trace_bytes < 1:
            raise ValueError("Positive explicit trace-allocation limit required")
        destination = np.asarray(event.destination)
        accept = np.asarray(event.accept)
        token_group = np.asarray(event.token_group)
        groups = tuple((tuple(g) for g in event.groups))
        if (
            accept.ndim != 1
            or accept.dtype != bool
            or (not len(accept))
            or (destination.shape != (len(accept), len(groups), 2))
            or (destination.dtype.kind not in "iu")
            or (destination < -1).any()
            or (destination >= len(accept)).any()
            or (type(event.start) is not int)
            or (not 0 <= event.start < len(accept))
            or (len(accept) >= np.iinfo(np.int32).max)
            or (2 * len(groups) >= np.iinfo(np.int32).max)
        ):
            raise ValueError("Valid complete S,G,2 graph required")
        if (
            token_group.ndim != 1
            or token_group.dtype.kind not in "iu"
            or (not len(token_group))
            or (not groups)
            or any((not g for g in groups))
            or any(
                (
                    not isinstance(t, (int, np.integer)) or isinstance(t, (bool, np.bool_))
                    for g in groups
                    for t in g
                )
            )
            or (sorted((int(t) for g in groups for t in g)) != list(range(len(token_group))))
            or any((token_group[t] != i for i, g in enumerate(groups) for t in g))
        ):
            raise ValueError("Whole-vocabulary partition required; no missing or duplicated tokens")
        self.destination = destination.astype(np.int64, copy=True)
        self.accept = accept.copy()
        self.token_group = token_group.copy()
        self.groups = tuple((tuple(sorted(map(int, g))) for g in groups))
        self.start = event.start
        self.max_trace_bytes = max_trace_bytes

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
        states = len(self.accept)
        groups = len(self.groups)
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
        valid = self.destination >= 0
        targets = self.destination.clip(min=0)
        value = np.where(self.accept, 0.0, -np.inf)
        rank = np.where(self.accept, 0, states).astype(np.int64)
        choices = np.full((length, states), -1, dtype=np.int32)
        edge_ids = np.arange(groups * 2, dtype=np.int64).reshape(1, groups, 2)
        for position in range(length - 1, -1, -1):
            edges = group_scores[position][None, :, None] + value[targets]
            edges = np.where(valid, edges, -np.inf)
            best = edges.max(axis=(1, 2))
            possible = np.isfinite(best)
            optimal = valid & possible[:, None, None] & (edges == best[:, None, None])
            tokens = group_tokens[position][None, :, None]
            selected_token = np.where(optimal, tokens, vocabulary).min(axis=(1, 2))
            optimal &= tokens == selected_token[:, None, None]
            successor_rank = rank[targets]
            selected_rank = np.where(optimal, successor_rank, states).min(axis=(1, 2))
            optimal &= successor_rank == selected_rank[:, None, None]
            choice = np.where(optimal, edge_ids, groups * 2).min(axis=(1, 2))
            choices[position] = np.where(possible, choice, -1)
            next_rank = np.full(states, states, dtype=np.int64)
            if possible.any():
                keys = np.stack((selected_token[possible], selected_rank[possible]), axis=1)
                _, inverse = np.unique(keys, axis=0, return_inverse=True)
                next_rank[possible] = inverse
            value = best
            rank = next_rank
        optimum = float(value[self.start])
        if not np.isfinite(optimum):
            raise ValueError("Condition has zero support; no fallback or repair")
        tokens = []
        state_ids = [self.start]
        branches = []
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
