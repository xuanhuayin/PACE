from dataclasses import dataclass
import numpy as np
from pace.automata.decoder_product import FiniteDecoderProduct


def minimize_text_dfa(dfa):
    partition = dfa.accept.astype(np.int32)
    while True:
        signatures, updated = ({}, [])
        for state in range(len(partition)):
            signature = (bool(dfa.accept[state]), tuple(partition[dfa.transition[state]].tolist()))
            updated.append(signatures.setdefault(signature, len(signatures)))
        updated = np.asarray(updated, dtype=np.int32)
        if np.array_equal(updated, partition):
            break
        partition = updated
    chosen = [
        int(np.flatnonzero(partition == group)[0]) for group in range(int(partition.max()) + 1)
    ]
    return (partition[dfa.transition[chosen]], dfa.accept[chosen], int(partition[dfa.start]))


@dataclass
class TokenEvent:
    destination: np.ndarray
    accept: np.ndarray
    token_group: np.ndarray
    groups: tuple
    start: int
    text_states: int
    utf8_states: int

    def accepts_ids(self, ids):
        state = self.start
        for token in ids:
            if type(token) is not int or not 0 <= token < len(self.token_group):
                raise ValueError("Invalid token ID")
            state = int(self.destination[state, self.token_group[token]])
        return bool(self.accept[state])


def compile_token_event(decoder, dfa, max_states=100000):
    if type(max_states) is not int or max_states < 1:
        raise ValueError("Positive integer state limit required")
    transition, accept, start = minimize_text_dfa(dfa)
    machine = FiniteDecoderProduct(decoder, transition, start)
    width = 2 * len(transition)
    signatures, groups, token_group = ({}, [], [])
    normal_maps = {}
    for token, chars in enumerate(machine.pieces):
        if token == decoder.eos:
            signature = ("eos",)
        elif token in decoder.special:
            signature = ("skip",)
        elif decoder.bytes[token] is not None:
            signature = ("byte", decoder.bytes[token])
        else:
            if chars not in normal_maps:
                states = np.arange(width, dtype=np.int32)
                for char in chars:
                    previous = states
                    states = 2 * transition[previous // 2, char]
                    if char == ord(" "):
                        states = np.where(previous % 2 == 1, 2 * (previous // 2), states)
                normal_maps[chars] = states.tobytes()
            signature = ("normal", normal_maps[chars])
        if signature not in signatures:
            signatures[signature] = len(groups)
            groups.append([])
        group = signatures[signature]
        groups[group].append(token)
        token_group.append(group)
    representatives = [group[0] for group in groups]
    states, indices, rows = ([machine.start], {machine.start: 0}, [])
    stopped = {}
    cursor = 0
    while cursor < len(states):
        row = []
        for token in representatives:
            after = machine.step(states[cursor], token)
            if after.stopped:
                flag = bool(accept[machine.final_state(after)])
                after = stopped.setdefault(flag, after)
            if after not in indices:
                if len(states) >= max_states:
                    raise ValueError("Reachable state safety limit; no truncation")
                indices[after] = len(states)
                states.append(after)
            row.append(indices[after])
        rows.append(row)
        cursor += 1
    destination = np.asarray(rows, dtype=np.int32)
    return TokenEvent(
        destination,
        np.asarray([accept[machine.final_state(s)] for s in states]),
        np.asarray(token_group, dtype=np.int32),
        tuple((tuple(g) for g in groups)),
        0,
        len(transition),
        len(machine.table),
    )
