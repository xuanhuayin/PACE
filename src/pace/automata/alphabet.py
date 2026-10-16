from functools import lru_cache
import re
import numpy as np

INVALID = -1
SPECIAL = {"İ": 128, "ı": 129, "ſ": 130, "K": 131}
SIGNATURES = [
    (False, False, False, False, False, False),
    (False, True, False, False, False, False),
    (False, True, True, False, False, False),
    (False, True, False, True, False, False),
    (False, True, False, False, True, False),
    (False, False, False, True, False, False),
    (False, False, True, False, False, False),
    (True, False, False, False, False, False),
    (False, True, False, False, False, True),
    (False, False, False, False, False, True),
]
SIG_BASE = 132
N_CLASSES = SIG_BASE + len(SIGNATURES)
OTHER_NONWORD, OTHER_SPACE = (132, 139)
_WORD_RE = re.compile("\\w")


def character_class(char):
    if type(char) is not str or len(char) != 1:
        raise ValueError("One Unicode character required")
    value = ord(char)
    if value < 128:
        return value
    if char in SPECIAL:
        return SPECIAL[char]
    lo, up = (char.islower(), char.isupper())
    sig = (
        char.isspace(),
        _WORD_RE.fullmatch(char) is not None,
        lo,
        up,
        char.isdecimal(),
        bool(char.istitle() and (not lo) and (not up)),
    )
    try:
        return SIG_BASE + SIGNATURES.index(sig)
    except ValueError:
        raise ValueError(f"Unmapped non-ASCII signature {sig} for U+{value:04X}")


_ASCII_SPACE = frozenset((c for c in range(128) if chr(c).isspace()))


def is_space_class(c):
    return c in _ASCII_SPACE or c == OTHER_SPACE


def is_word_class(c):
    if c < 128:
        return c == 95 or 48 <= c <= 57 or 65 <= c <= 90 or (97 <= c <= 122)
    return c in (128, 129, 130, 131, 133, 134, 135, 136, 140)


def is_lower_class(c):
    return 97 <= c <= 122 or c in (129, 130, 134, 138)


def is_upper_class(c):
    return 65 <= c <= 90 or c in (128, 131, 135, 137)


def is_title_class(c):
    return c in (140, 141)


def is_cased_class(c):
    return is_lower_class(c) or is_upper_class(c) or is_title_class(c)


def is_decimal_class(c):
    return 48 <= c <= 57 or c == 136


def fold_rei(c):
    """Python re.IGNORECASE equivalence onto ASCII lowercase (İ, ı -> i; ſ -> s; K -> k)."""
    if 65 <= c <= 90:
        return c + 32
    return {128: 105, 129: 105, 130: 115, 131: 107}.get(c, c)


def fold_lower(c):
    """str.lower() onto ASCII lowercase where it lands on one ASCII char (K -> k; İ -> i + U+0307,
    approximated as i for counting; ı and ſ stay themselves)."""
    if 65 <= c <= 90:
        return c + 32
    return {128: 105, 131: 107}.get(c, c)


@lru_cache(maxsize=1)
def utf8_table():
    root = {}
    for point in range(1114112):
        if 55296 <= point <= 57343:
            continue
        char = chr(point)
        encoded = char.encode("utf-8")
        node = root
        for byte in encoded[:-1]:
            node = node.setdefault(byte, {})
        node[encoded[-1]] = -character_class(char) - 2
    interned, rows = ({}, [])

    def intern(node):
        signature = tuple(
            (
                (byte, intern(value) if isinstance(value, dict) else value)
                for byte, value in sorted(node.items())
            )
        )
        if signature not in interned:
            index = len(rows)
            interned[signature] = index
            row = np.full(256, INVALID, dtype=np.int32)
            for byte, value in signature:
                row[byte] = value
            rows.append(row)
        return interned[signature]

    start = intern(root)
    table = np.stack(rows)
    table.flags.writeable = False
    return (table, start)


def classify_bytes(values):
    table, root = utf8_table()
    state, result = (root, [])
    for byte in values:
        after = int(table[state, byte])
        if after == INVALID:
            return None
        if after < INVALID:
            result.append(-after - 2)
            state = root
        else:
            state = after
    return result if state == root else None
