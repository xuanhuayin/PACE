from dataclasses import dataclass
import re
import numpy as np
from pace.automata.lexical.dfa import LexicalDFA, literal_count_dfa
from pace.automata.compiler import minimize_text_dfa
from pace.automata.utf8 import N_CLASSES

KINDS = ("keywords:existence", "keywords:frequency", "keywords:forbidden_words")


@dataclass(frozen=True)
class AtomicConstraint:
    instruction_index: int
    instruction_id: str
    term_index: int
    word: str
    threshold: int
    relation: str
    whole_word: bool


@dataclass(frozen=True)
class LexicalPromptContract:
    key: int
    prompt: str
    atoms: tuple
    lexical_indices: tuple
    other_indices: tuple


def prompt_contract(row):
    if type(row.get("key")) is not int or type(row.get("prompt")) is not str:
        raise ValueError("Original prompt key and text required")
    names, args = (row["instruction_id_list"], row["kwargs"])
    if not names or len(names) != len(args):
        raise ValueError("Instruction/argument alignment")
    atoms, lexical, other = ([], [], [])
    for index, (name, kwargs) in enumerate(zip(names, args)):
        if name not in KINDS:
            other.append(index)
            continue
        lexical.append(index)
        if name == "keywords:frequency":
            word, count, relation = (kwargs["keyword"], kwargs["frequency"], kwargs["relation"])
            if (
                type(word) is not str
                or not word
                or type(count) is not int
                or (count < 0)
                or (relation not in ("at least", "less than"))
            ):
                raise ValueError(
                    "Original explicit frequency arguments required; no random fallback"
                )
            terms = [word.strip()]
            boundary = False
        else:
            field = "keywords" if name == "keywords:existence" else "forbidden_words"
            terms = kwargs[field]
            if (
                not isinstance(terms, (list, tuple))
                or not terms
                or any((type(w) is not str for w in terms))
            ):
                raise ValueError("Explicit nonempty keyword list required; no random fallback")
            terms = sorted(terms if name == "keywords:existence" else set(terms))
            count, relation, boundary = (
                (1, "at least", False) if name == "keywords:existence" else (1, "less than", True)
            )
        for term_index, word in enumerate(terms):
            if re.fullmatch("[A-Za-z0-9_]+", word) is None:
                raise ValueError(
                    "Only explicit ASCII word literals are supported; do not escape a regex silently"
                )
            atoms.append(AtomicConstraint(index, name, term_index, word, count, relation, boundary))
    return LexicalPromptContract(
        row["key"], row["prompt"], tuple(atoms), tuple(lexical), tuple(other)
    )


def coaccessible(transition, accept):
    """States with some accepting continuation; all other histories may merge."""
    incoming = [set() for _ in accept]
    for state, row in enumerate(transition):
        for target in set(row.tolist()):
            incoming[target].add(state)
    viable = np.asarray(accept, dtype=bool).copy()
    pending = list(np.flatnonzero(viable))
    while pending:
        for source in incoming[pending.pop()]:
            if not viable[source]:
                viable[source] = True
                pending.append(source)
    return viable


def joint_text_dfa(contract, max_states=100000):
    if not isinstance(contract, LexicalPromptContract) or not contract.atoms:
        raise ValueError(
            "At least one lexical constraint required; bypass nonlexical rows explicitly"
        )
    if type(max_states) is not int or max_states < 1:
        raise ValueError("Positive state limit required")
    components, signatures = ([], set())
    for atom in contract.atoms:
        signature = (atom.word.lower(), atom.threshold, atom.relation, atom.whole_word)
        if signature in signatures:
            continue
        signatures.add(signature)
        dfa = literal_count_dfa(
            atom.word, atom.threshold, atom.relation, atom.whole_word, reject_empty=True
        )
        transition, accept, start = minimize_text_dfa(dfa)
        components.append((transition, accept, start, coaccessible(transition, accept)))
    labels, representatives, signature_ids = ([], [], {})
    for char in range(N_CLASSES):
        signature = tuple((table[:, char].tobytes() for table, _, _, _ in components))
        if signature not in signature_ids:
            signature_ids[signature] = len(representatives)
            representatives.append(char)
        labels.append(signature_ids[signature])
    start = tuple((c[2] for c in components))
    if not all((c[3][s] for c, s in zip(components, start))):
        start = None
    states, indices, rows, accepted = ([start], {start: 0}, [], [])
    cursor = 0
    while cursor < len(states):
        state = states[cursor]
        row = []
        accepted.append(state is not None and all((c[1][s] for c, s in zip(components, state))))
        for char in representatives:
            after = (
                None
                if state is None
                else tuple((int(c[0][s, char]) for c, s in zip(components, state)))
            )
            if after is not None and (not all((c[3][s] for c, s in zip(components, after)))):
                after = None
            if after not in indices:
                if len(states) >= max_states:
                    raise ValueError("Joint text state limit; no constraint or state truncation")
                indices[after] = len(states)
                states.append(after)
            row.append(indices[after])
        rows.append([row[label] for label in labels])
        cursor += 1
    return LexicalDFA(
        np.asarray(rows, dtype=np.int32), np.asarray(accepted, dtype=bool), 0, tuple(states)
    )
