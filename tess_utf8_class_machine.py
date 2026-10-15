from functools import lru_cache
import numpy as np
OTHER_WORD = 128
OTHER_NONWORD = 129
OTHER_SPACE = 130
N_CLASSES = 131
INVALID = -1

def character_class(char):
    if type(char) is not str or len(char) != 1:
        raise ValueError('One Unicode character required')
    value = ord(char)
    if value < 128:
        return value + 32 if 65 <= value <= 90 else value
    if char in ('İ', 'ı'):
        return ord('i')
    if char == 'ſ':
        return ord('s')
    if char == 'K':
        return ord('k')
    if char.isspace():
        return OTHER_SPACE
    return OTHER_WORD if char.isalnum() else OTHER_NONWORD

def is_word_class(value):
    return value == OTHER_WORD or value == 95 or 48 <= value <= 57 or (97 <= value <= 122)

def is_space_class(value):
    return value == OTHER_SPACE or (value < 128 and chr(value).isspace())

@lru_cache(maxsize=1)
def utf8_table():
    """Minimized acyclic trie for ALL Unicode scalar encodings.

    Nonnegative entries are pending states. -1 is invalid. Other negative
    entries emit class ``-entry-2`` and return to root. Rejects overlong forms,
    surrogates and values above U+10FFFF by construction, not replacement.
    """
    root = {}
    for point in range(1114112):
        if 55296 <= point <= 57343:
            continue
        char = chr(point)
        encoded = char.encode('utf-8')
        node = root
        for byte in encoded[:-1]:
            node = node.setdefault(byte, {})
        node[encoded[-1]] = -character_class(char) - 2
    interned, rows = ({}, [])

    def intern(node):
        signature = tuple(((byte, intern(value) if isinstance(value, dict) else value) for byte, value in sorted(node.items())))
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
    """Testing helper: None for invalid/incomplete UTF-8, else emitted classes."""
    table, root = utf8_table()
    state, result = (root, [])
    for byte in values:
        if type(byte) is not int or not 0 <= byte <= 255:
            raise ValueError('Byte outside [0, 255]')
        after = int(table[state, byte])
        if after == INVALID:
            return None
        if after < INVALID:
            result.append(-after - 2)
            state = root
        else:
            state = after
    return result if state == root else None
