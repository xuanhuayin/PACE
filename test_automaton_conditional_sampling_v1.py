import numpy as np
from types import SimpleNamespace

def fixture(ambiguous=False):
    d = np.full((4, 3, 2), -1, dtype=np.int64)
    d[0, 0] = [1, 2]
    d[1, 1, 0] = 3
    d[2, 1 if ambiguous else 2, 0] = 3
    if ambiguous:
        d[0, 2, 0] = 3
        d[3, 2, 0] = 3
    return SimpleNamespace(destination=d, accept=np.array([False, False, False, True]), start=0, token_group=np.array([0, 0, 1, 2]), groups=((0, 1), (2,), (3,)))

def paths(event, ids):
    active = [(event.start, ())]
    for token in ids:
        active = [(int(dst), trace + ((state, int(event.token_group[token]), branch, int(dst)),)) for state, trace in active for branch, dst in enumerate(event.destination[state, event.token_group[token]]) if dst >= 0]
    return [trace for state, trace in active if event.accept[state]]
