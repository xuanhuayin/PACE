import numpy as np
from pace.automata.unicode import UnambiguousDecoder, BranchState, BranchedTokenEvent, VALID
from pace.automata.compiler import minimize_text_dfa


class ContextualDecoder(UnambiguousDecoder):
    def __init__(self, decoder, transition, accept, start):
        super().__init__(decoder, transition, accept, start)
        table, root = (self.base.table, self.base.root)
        texts = 2 * len(transition)
        mapping = np.full((texts, len(table)), -1, dtype=np.int32)
        representatives, signatures, rows = ([], {}, [])
        for text in range(texts):
            if not self.viable[text // 2]:
                continue
            for pending, source in enumerate(table):
                if pending == root:
                    continue
                children = source >= 0
                if np.any(source[children] >= pending) or np.any(source[children] == root):
                    raise ValueError("UTF-8 pending trie is not bottom-up acyclic")
                row = np.full(256, -1, dtype=np.int32)
                row[children] = mapping[text, source[children]]
                if np.any(row[children] < 0):
                    raise ValueError("Unmapped pending child")
                emitted = source < -1
                for char in np.unique(-source[emitted] - 2):
                    after = self.base.char_step(text, int(char))
                    row[source == -int(char) - 2] = -after - 2 if self.viable[after // 2] else -1
                signature = row.tobytes()
                if signature not in signatures:
                    signatures[signature] = len(representatives)
                    representatives.append((text, pending))
                    rows.append(row)
                mapping[text, pending] = signatures[signature]
        self.context_class = mapping
        self.context_representatives = np.asarray(representatives, dtype=np.int32).reshape(-1, 2)
        self.context_rows = np.asarray(rows, dtype=np.int32).reshape(-1, 256)
        for array in (self.context_class, self.context_representatives, self.context_rows):
            array.flags.writeable = False

    def canonical(self, state):
        if state.mode != VALID or state.utf8 == self.base.root:
            return state
        index = int(self.context_class[state.text, state.utf8])
        if index < 0:
            raise ValueError("Nonviable or unmapped pending state")
        text, pending = self.context_representatives[index]
        return BranchState(VALID, int(text), int(pending))

    def step(self, state, token):
        return tuple((self.canonical(after) for after in super().step(state, token)))


def certify_context(machine):
    """Check the full residual commuting relation, not sampled input paths.

    For every viable text state and pending UTF-8 state, enumerate all 256 byte
        labels from the original decoder's table and character-step equations. Pending states reject
    EOS and ordinary text tokens and retain skipped-special self loops. Thus the
    byte relation plus these mode rules covers the full token vocabulary. Induction
    preserves every completed path count and every weighted sequence probability.
    """
    old = UnambiguousDecoder(machine.decoder, machine.base.transition, machine.accept, 0)
    if not np.array_equal(old.base.table, machine.base.table) or old.base.root != machine.base.root:
        raise ValueError("Underlying UTF-8 transducer differs")
    mapping, reps = (machine.context_class, machine.context_representatives)
    expected_shape = (2 * len(machine.accept), len(old.base.table))
    if mapping.shape != expected_shape or machine.context_rows.shape != (len(reps), 256):
        raise ValueError("Certificate dimensions differ")
    active = old.viable[np.arange(expected_shape[0]) // 2]
    expected = np.broadcast_to(active[:, None], expected_shape).copy()
    expected[:, old.base.root] = False
    if not np.array_equal(mapping >= 0, expected):
        raise ValueError("Residual domain differs")
    if set(mapping[expected].tolist()) != set(range(len(reps))):
        raise ValueError("Residual classes not surjective")
    for index, (text, pending) in enumerate(reps):
        if (
            not (0 <= text < expected_shape[0] and 0 <= pending < expected_shape[1])
            or mapping[text, pending] != index
        ):
            raise ValueError("Invalid class representative")
    checked = 0
    for text in range(expected_shape[0]):
        if not active[text]:
            continue
        for pending, raw in enumerate(old.base.table):
            if pending == old.base.root:
                continue
            left = np.full(256, -1, dtype=np.int32)
            for byte, after in enumerate(raw):
                after = int(after)
                if after >= 0:
                    left[byte] = mapping[text, after]
                elif after < -1:
                    target = old.base.char_step(text, -after - 2)
                    if old.viable[target // 2]:
                        left[byte] = -target - 2
            index = int(mapping[text, pending])
            rt, ru = map(int, reps[index])
            right = np.full(256, -1, dtype=np.int32)
            for byte, after in enumerate(old.base.table[ru]):
                after = int(after)
                if after >= 0:
                    right[byte] = mapping[rt, after]
                elif after < -1:
                    target = old.base.char_step(rt, -after - 2)
                    if old.viable[target // 2]:
                        right[byte] = -target - 2
            if not np.array_equal(left, right) or not np.array_equal(
                left, machine.context_rows[index]
            ):
                raise ValueError("Residual continuation/weight relation differs")
            if old.terminal(BranchState(VALID, text, pending)) or old.flush_allowed(
                BranchState(VALID, text, pending)
            ):
                raise ValueError("Merged residual is flushable or terminal")
            checked += 1
    return dict(
        raw_pending_states=checked,
        shared_pending_states=len(reps),
        byte_relations=256 * checked,
        unchanged_modes=["NORMAL", "VALID_root", "FALLBACK", "STOPPED"],
    )


def compile_contextual_event(decoder, dfa, max_states=100000):
    """Same full vocabulary/branch semantics and state ceiling; share before BFS.

    Returns the event, its contextual decoder (finite certificate), and BFS states.
    No generated text, reference answers or approximate score enters compilation.
    """
    if type(max_states) is not int or max_states < 1:
        raise ValueError("Positive state limit required")
    transition, accept, start = minimize_text_dfa(dfa)
    machine = ContextualDecoder(decoder, transition, accept, start)
    groups, group_ids, token_group, maps = ([], {}, [], {})
    for token, chars in enumerate(machine.base.pieces):
        if token == decoder.eos:
            signature = ("eos",)
        elif token in decoder.special:
            signature = ("skip",)
        elif decoder.bytes[token] is not None:
            signature = ("byte", decoder.bytes[token])
        else:
            if chars not in maps:
                states = np.arange(2 * len(transition), dtype=np.int32)
                for char in chars:
                    previous = states
                    states = 2 * transition[previous // 2, char]
                    if char == ord(" "):
                        states = np.where(previous % 2 == 1, 2 * (previous // 2), states)
                maps[chars] = states.tobytes()
            signature = ("normal", maps[chars])
        if signature not in group_ids:
            group_ids[signature] = len(groups)
            groups.append([])
        group = group_ids[signature]
        groups[group].append(token)
        token_group.append(group)
    representatives = [g[0] for g in groups]
    states, index, rows = ([machine.start], {machine.start: 0}, [])
    cursor = 0
    while cursor < len(states):
        row = np.full((len(groups), 2), -1, dtype=np.int32)
        for group, token in enumerate(representatives):
            following = machine.step(states[cursor], token)
            if len(following) > 2:
                raise ValueError("Unexpected branch multiplicity")
            for branch, after in enumerate(following):
                if after not in index:
                    if len(states) >= max_states:
                        raise ValueError("Contextual state limit; no truncation")
                    index[after] = len(states)
                    states.append(after)
                row[group, branch] = index[after]
        rows.append(row)
        cursor += 1
    destination = np.sort(np.stack(rows), axis=-1)
    signatures, merged, columns, mapping = ({}, [], [], [])
    for group, tokens in enumerate(groups):
        signature = destination[:, group, :].tobytes()
        if signature not in signatures:
            signatures[signature] = len(merged)
            merged.append([])
            columns.append(group)
        new = signatures[signature]
        merged[new].extend(tokens)
        mapping.append(new)
    event = BranchedTokenEvent(
        destination[:, columns, :],
        np.asarray([machine.terminal(s) for s in states], dtype=bool),
        np.asarray(mapping, dtype=np.int32)[np.asarray(token_group)],
        tuple((tuple(sorted(g)) for g in merged)),
        0,
        len(transition),
        len(machine.base.table),
    )
    return (event, machine, tuple(states))
