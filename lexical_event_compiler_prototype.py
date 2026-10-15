from dataclasses import dataclass
import numpy as np
from lexical_event_stream_prototype import LETTER
from lexical_event_boundary_stream import EVENT_CLEANUP, LexicalBoundaryStream

@dataclass
class CompiledLexicalEvent:
    word: str
    groups: tuple
    token_group: np.ndarray
    destination: np.ndarray
    increment: np.ndarray
    final_increment: np.ndarray
    start: int
    eos_state: int
    character_states: int
    alphabet_size: int

    def count_ids(self, ids):
        state = self.start
        count = 0
        for token in ids:
            if type(token) is not int or not 0 <= token < len(self.token_group):
                raise ValueError('Token identity')
            group = self.token_group[token]
            count += int(self.increment[group, state])
            state = int(self.destination[group, state])
        return count + int(self.final_increment[state])

def compile_event(word, surfaces, eos_id=1, max_states=10000):
    if not surfaces or any((type(v) is not str for v in surfaces)) or type(eos_id) is not int or (not 0 <= eos_id < len(surfaces)):
        raise ValueError('Surface vocabulary')
    machine = LexicalBoundaryStream(word)
    literal = set(''.join((old + new for old, new in EVENT_CLEANUP)))
    chars = sorted(set(''.join(surfaces)) | literal)
    representatives = []
    signature_ids = {}
    character_ids = {}
    for char in chars:
        signature = (char if char in literal else None, bool(LETTER.fullmatch(char)), tuple((bool(letter.fullmatch(char)) for letter in machine.letters)))
        if signature not in signature_ids:
            signature_ids[signature] = len(representatives)
            representatives.append(char)
        character_ids[char] = signature_ids[signature]
    states = [machine.start]
    state_ids = {machine.start: 0}
    destinations = []
    increments = []
    cursor = 0
    while cursor < len(states):
        row = []
        counts = []
        for char in representatives:
            after, count = machine.step(states[cursor], char)
            if after not in state_ids:
                if len(states) >= max_states:
                    raise ValueError('Character graph safety limit; no truncation')
                state_ids[after] = len(states)
                states.append(after)
            row.append(state_ids[after])
            counts.append(count)
        destinations.append(row)
        increments.append(counts)
        cursor += 1
    dest = np.asarray(destinations, dtype=np.int32)
    inc = np.asarray(increments, dtype=np.int32)
    final = np.asarray([machine.finish(state) for state in states], dtype=np.int32)
    partition = np.zeros(len(states), dtype=np.int32)
    while True:
        signatures = {}
        updated = []
        for i in range(len(states)):
            sig = (int(final[i]), tuple(zip(inc[i].tolist(), partition[dest[i]].tolist())))
            updated.append(signatures.setdefault(sig, len(signatures)))
        updated = np.asarray(updated, dtype=np.int32)
        if np.array_equal(updated, partition):
            break
        partition = updated
    chosen = [int(np.flatnonzero(partition == i)[0]) for i in range(int(partition.max()) + 1)]
    char_dest = partition[dest[chosen]]
    char_inc = inc[chosen]
    final = final[chosen]
    width = len(chosen)
    indices = np.arange(width, dtype=np.int32)
    groups = []
    group_ids = {}
    token_group = []
    group_dest = []
    group_inc = []
    surfaces_seen = {}
    for token, surface in enumerate(surfaces):
        code = tuple((character_ids[char] for char in surface))
        if token == eos_id:
            target = np.full(width + 1, width, dtype=np.int32)
            counts = np.r_[final, 0].astype(np.int32)
        else:
            if code not in surfaces_seen:
                target = indices.copy()
                counts = np.zeros(width, dtype=np.int32)
                for char in code:
                    counts += char_inc[target, char]
                    target = char_dest[target, char]
                surfaces_seen[code] = (np.r_[target, width].astype(np.int32), np.r_[counts, 0].astype(np.int32))
            target, counts = surfaces_seen[code]
        signature = (target.tobytes(), counts.tobytes())
        if signature not in group_ids:
            group_ids[signature] = len(groups)
            groups.append([])
            group_dest.append(target.copy())
            group_inc.append(counts.copy())
        group = group_ids[signature]
        groups[group].append(token)
        token_group.append(group)
    return CompiledLexicalEvent(word=word, groups=tuple((tuple(v) for v in groups)), token_group=np.asarray(token_group, dtype=np.int32), destination=np.stack(group_dest), increment=np.stack(group_inc), final_increment=np.r_[final, 0].astype(np.int32), start=int(partition[0]), eos_state=width, character_states=len(states), alphabet_size=len(representatives))
