from dataclasses import dataclass
from tess_utf8_class_machine import character_class, utf8_table, INVALID, N_CLASSES

def specialize_utf8(transition):
    """Merge only characters with identical DFA effects and strip behavior.

    Bottom-up minimization of the acyclic UTF-8 trie preserves every byte;
    this is not byte-token filtering or a restricted vocabulary approximation.
    """
    import numpy as np
    table, root = utf8_table()
    representatives, labels = ({}, [])
    for char in range(N_CLASSES):
        signature = (transition[:, char].tobytes(), char == ord(' '))
        labels.append(representatives.setdefault(signature, char))
    rows, ids, mapping = ([], {}, {})
    for index, row in enumerate(table):
        values = []
        for after in row:
            after = int(after)
            values.append(mapping[after] if after >= 0 else INVALID if after == INVALID else -labels[-after - 2] - 2)
        signature = tuple(values)
        if signature not in ids:
            ids[signature] = len(rows)
            rows.append(values)
        mapping[index] = ids[signature]
    result = np.asarray(rows, dtype=np.int32)
    result.flags.writeable = False
    return (result, mapping[root])

@dataclass(frozen=True)
class DecoderState:
    candidate: int
    fallback: int = -1
    utf8: int = -1
    in_bytes: bool = False
    stopped: bool = False

class FiniteDecoderProduct:

    def __init__(self, decoder, transition, start=0):
        """transition[state, class] defines a finite text DFA (131 classes).

        Leading-space removal is handled here by doubling DFA states with a
        flag that means no output character has yet been processed. Callers
        classify final states themselves; no particular count event is assumed.
        """
        import numpy as np
        if not isinstance(transition, np.ndarray) or transition.ndim != 2 or transition.shape[1] != N_CLASSES or (transition.dtype.kind not in 'iu') or (transition.shape[0] < 1) or (transition < 0).any() or (transition >= len(transition)).any():
            raise ValueError('Finite integer S x 131 DFA required')
        if type(start) is not int or not 0 <= start < len(transition):
            raise ValueError('Invalid start state')
        self.decoder = decoder
        self.transition = transition.copy()
        self.transition.flags.writeable = False
        self.table, self.root = specialize_utf8(self.transition)
        self.start = DecoderState(2 * start + 1)
        self.pieces = tuple((tuple((character_class(c) for c in s)) for s in decoder.pieces))
        self.replacement = character_class('�')

    def char_step(self, state, char):
        base, first = divmod(state, 2)
        if first and char == ord(' '):
            return 2 * base
        return 2 * int(self.transition[base, char])

    def flush(self, state):
        if not state.in_bytes:
            return state.candidate
        return state.candidate if state.utf8 == self.root else state.fallback

    def step(self, state, token):
        if type(token) is not int or not 0 <= token < len(self.pieces):
            raise ValueError('Invalid tokenizer ID')
        if state.stopped:
            return state
        if token == self.decoder.eos:
            return DecoderState(self.flush(state), stopped=True)
        if token in self.decoder.special:
            return state
        byte = self.decoder.bytes[token]
        if byte is None:
            current = self.flush(state)
            for char in self.pieces[token]:
                current = self.char_step(current, char)
            return DecoderState(current)
        if state.in_bytes:
            candidate, fallback, utf8 = (state.candidate, state.fallback, state.utf8)
        else:
            candidate = fallback = state.candidate
            utf8 = self.root
        fallback = self.char_step(fallback, self.replacement)
        if utf8 == INVALID:
            return DecoderState(-1, fallback, INVALID, True)
        after = int(self.table[utf8, byte])
        if after == INVALID:
            return DecoderState(-1, fallback, INVALID, True)
        if after < INVALID:
            candidate = self.char_step(candidate, -after - 2)
            after = self.root
        return DecoderState(candidate, fallback, after, True)

    def final_state(self, state):
        return self.flush(state) // 2

    def run(self, ids):
        state = self.start
        for token in ids:
            state = self.step(state, token)
        return self.final_state(state)
