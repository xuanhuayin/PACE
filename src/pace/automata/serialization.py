from types import SimpleNamespace
from pace.automata.validation import validate_arrays


def to_event(arrays):
    validate_arrays(arrays)
    tokens, offsets = (arrays["group_tokens"], arrays["group_offsets"])
    return SimpleNamespace(
        destination=arrays["destination"],
        accept=arrays["accept"],
        start=int(arrays["start"]),
        token_group=arrays["token_group"],
        groups=tuple(
            (tuple(map(int, tokens[offsets[g] : offsets[g + 1]])) for g in range(len(offsets) - 1))
        ),
    )
