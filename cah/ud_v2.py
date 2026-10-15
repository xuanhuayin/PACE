from dataclasses import dataclass
import numpy as np
from fdp_v2 import FiniteDecoderProduct, specialize_utf8
from tess_ifeval_joint_lexical import coaccessible
from tess_token_event_compiler import minimize_text_dfa
from alphabet_v2 import INVALID, N_CLASSES
NORMAL, VALID, FALLBACK, STOPPED = range(4)

@dataclass(frozen=True)
class BranchState:
    mode: int
    text: int
    utf8: int = -1

class UnambiguousDecoder:

    def __init__(self, decoder, transition, accept, start):
        self.base = FiniteDecoderProduct(decoder, transition, start)
        self.decoder = decoder
        self.accept = np.asarray(accept, dtype=bool)
        self.viable = coaccessible(transition, self.accept)
        self.validation, self.validation_root = specialize_utf8(np.zeros((1, N_CLASSES), dtype=np.int32))
        self.start = BranchState(NORMAL, self.base.start.candidate)
        self.stop = BranchState(STOPPED, 0)

    def flush_allowed(self, state):
        return state.mode in (NORMAL, STOPPED) or (state.mode == VALID and state.utf8 == self.base.root) or (state.mode == FALLBACK and state.utf8 != self.validation_root)

    def terminal(self, state):
        if state.mode == STOPPED:
            return True
        return self.flush_allowed(state) and bool(self.accept[state.text // 2])

    def keep(self, state):
        return (state,) if self.viable[state.text // 2] else ()

    def valid_byte(self, text, utf8, byte):
        after = int(self.base.table[utf8, byte])
        if after == INVALID:
            return ()
        if after < INVALID:
            text = self.base.char_step(text, -after - 2)
            after = self.base.root
        return self.keep(BranchState(VALID, text, after))

    def fallback_byte(self, text, utf8, byte):
        after = INVALID if utf8 == INVALID else int(self.validation[utf8, byte])
        if after < INVALID:
            after = self.validation_root
        text = self.base.char_step(text, self.base.replacement)
        return self.keep(BranchState(FALLBACK, text, after))

    def step(self, state, token):
        if type(token) is not int or not 0 <= token < len(self.base.pieces):
            raise ValueError('Invalid token ID')
        if state.mode == STOPPED:
            return (state,)
        if token == self.decoder.eos:
            return (self.stop,) if self.terminal(state) else ()
        if token in self.decoder.special:
            return (state,)
        byte = self.decoder.bytes[token]
        if byte is None:
            if not self.flush_allowed(state):
                return ()
            current = state.text
            for char in self.base.pieces[token]:
                current = self.base.char_step(current, char)
            return self.keep(BranchState(NORMAL, current))
        if state.mode == NORMAL:
            return self.valid_byte(state.text, self.base.root, byte) + self.fallback_byte(state.text, self.validation_root, byte)
        if state.mode == VALID:
            return self.valid_byte(state.text, state.utf8, byte)
        if state.mode == FALLBACK:
            return self.fallback_byte(state.text, state.utf8, byte)
        raise ValueError('Unknown branch state')

    def accepting_paths(self, ids):
        states = {self.start: 1}
        for token in ids:
            if type(token) is not int or not 0 <= token < len(self.base.pieces):
                raise ValueError('Invalid token ID')
            following = {}
            for state, count in states.items():
                for after in self.step(state, token):
                    following[after] = following.get(after, 0) + count
            states = following
        return sum((count for state, count in states.items() if self.terminal(state)))

@dataclass
class BranchedTokenEvent:
    destination: np.ndarray
    accept: np.ndarray
    token_group: np.ndarray
    groups: tuple
    start: int
    text_states: int
    utf8_states: int

    def accepting_paths(self, ids):
        states = {self.start: 1}
        for token in ids:
            if type(token) is not int or not 0 <= token < len(self.token_group):
                raise ValueError('Invalid token ID')
            group = self.token_group[token]
            following = {}
            for state, count in states.items():
                for after in self.destination[state, group]:
                    if after >= 0:
                        after = int(after)
                        following[after] = following.get(after, 0) + count
            states = following
        return sum((count for state, count in states.items() if self.accept[state]))

    def accepts_ids(self, ids):
        count = self.accepting_paths(ids)
        if count not in (0, 1):
            raise ValueError('Ambiguous completed decoder path')
        return count == 1

def compile_unambiguous_event(decoder, dfa, max_states=100000):
    if type(max_states) is not int or max_states < 1:
        raise ValueError('Positive state limit required')
    transition, accept, start = minimize_text_dfa(dfa)
    machine = UnambiguousDecoder(decoder, transition, accept, start)
    group_ids, groups, token_group, maps = ({}, [], [], {})
    for token, chars in enumerate(machine.base.pieces):
        if token == decoder.eos:
            signature = ('eos',)
        elif token in decoder.special:
            signature = ('skip',)
        elif decoder.bytes[token] is not None:
            signature = ('byte', decoder.bytes[token])
        else:
            if chars not in maps:
                states = np.arange(2 * len(transition), dtype=np.int32)
                for char in chars:
                    previous = states
                    states = 2 * transition[previous // 2, char]
                    if char == ord(' '):
                        states = np.where(previous % 2 == 1, 2 * (previous // 2), states)
                maps[chars] = states.tobytes()
            signature = ('normal', maps[chars])
        if signature not in group_ids:
            group_ids[signature] = len(groups)
            groups.append([])
        group = group_ids[signature]
        groups[group].append(token)
        token_group.append(group)
    representatives = [group[0] for group in groups]
    states, indices, rows = ([machine.start], {machine.start: 0}, [])
    cursor = 0
    while cursor < len(states):
        row = np.full((len(groups), 2), -1, dtype=np.int32)
        for group, token in enumerate(representatives):
            following = machine.step(states[cursor], token)
            if len(following) > 2:
                raise ValueError('Unexpected branch multiplicity')
            for branch, after in enumerate(following):
                if after not in indices:
                    if len(states) >= max_states:
                        raise ValueError('Branched state limit; no truncation')
                    indices[after] = len(states)
                    states.append(after)
                row[group, branch] = indices[after]
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
        index = signatures[signature]
        merged[index].extend(tokens)
        mapping.append(index)
    return BranchedTokenEvent(destination[:, columns, :], np.asarray([machine.terminal(state) for state in states], dtype=bool), np.asarray(mapping, dtype=np.int32)[np.asarray(token_group)], tuple((tuple(sorted(group)) for group in merged)), 0, len(transition), len(machine.base.table))
