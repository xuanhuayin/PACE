from dataclasses import dataclass
import re
import numpy as np
from pace.automata.utf8 import N_CLASSES, character_class, is_space_class, is_word_class


@dataclass
class LexicalDFA:
    transition: np.ndarray
    accept: np.ndarray
    start: int
    states: tuple

    def accepts_text(self, text):
        state = self.start
        for char in text:
            state = int(self.transition[state, character_class(char)])
        return bool(self.accept[state])


def literal_count_dfa(word, threshold, relation="exact", whole_word=False, reject_empty=True):
    if type(word) is not str or re.fullmatch("[A-Za-z0-9_]+", word) is None:
        raise ValueError("Explicit ASCII word literal required; arbitrary regex unsupported")
    if (
        type(threshold) is not int
        or threshold < 0
        or relation not in ("exact", "at least", "less than")
    ):
        raise ValueError("Nonnegative integer threshold and supported relation required")
    if type(whole_word) is not bool or type(reject_empty) is not bool:
        raise ValueError("Boolean boundary/empty policy required")
    letters = tuple(map(ord, word.lower()))
    cap = threshold + 1 if relation == "exact" else max(threshold, 1)
    start = (0, 0, False, False)
    states, indices, rows = ([start], {start: 0}, [])
    cursor = 0
    while cursor < len(states):
        prefix, count, previous_word, content = states[cursor]
        row = []
        for char in range(N_CLASSES):
            after_count = count
            word_char = is_word_class(char)
            if whole_word:
                if prefix == len(letters) and (not word_char):
                    after_count = min(cap, count + 1)
                if prefix < len(letters) and prefix > 0 and (char == letters[prefix]):
                    after_prefix = prefix + 1
                elif not previous_word and char == letters[0]:
                    after_prefix = 1
                else:
                    after_prefix = 0
            else:
                sequence = letters[:prefix] + (char,)
                if len(sequence) == len(letters) and sequence == letters:
                    after_count = min(cap, count + 1)
                    after_prefix = 0
                else:
                    after_prefix = max(
                        (n for n in range(1, len(letters)) if sequence[-n:] == letters[:n]),
                        default=0,
                    )
            state = (
                after_prefix,
                after_count,
                word_char if whole_word else False,
                content or not is_space_class(char),
            )
            if state not in indices:
                indices[state] = len(states)
                states.append(state)
            row.append(indices[state])
        rows.append(row)
        cursor += 1
    accepted = []
    for prefix, count, previous_word, content in states:
        count = min(cap, count + int(whole_word and prefix == len(letters)))
        flag = (
            count == threshold
            if relation == "exact"
            else count >= threshold
            if relation == "at least"
            else count < threshold
        )
        accepted.append(flag and (content or not reject_empty))
    return LexicalDFA(
        np.asarray(rows, dtype=np.int32), np.asarray(accepted, dtype=bool), 0, tuple(states)
    )
