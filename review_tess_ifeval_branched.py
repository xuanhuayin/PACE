import numpy as np

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_arrays(a, vocabulary=32001):
    require(set(a) == {'destination', 'accept', 'token_group', 'group_tokens', 'group_offsets', 'start'}, 'Array names')
    d, accept, tg = (a['destination'], a['accept'], a['token_group'])
    require(d.dtype == np.int32 and d.ndim == 3 and (d.shape[2] == 2), 'Branch array shape/type')
    s, g, _ = d.shape
    require(0 < s <= 100000 and g > 0, 'State/group limits')
    require(accept.dtype == np.bool_ and accept.shape == (s,), 'Acceptance array')
    require(a['start'].shape == () and a['start'].dtype == np.int32 and (0 <= int(a['start']) < s), 'Start state')
    require(np.all((d >= -1) & (d < s)), 'Invalid destination')
    require(tg.dtype == np.int32 and tg.shape == (vocabulary,) and np.all((tg >= 0) & (tg < g)), 'Token labels')
    tokens, offsets = (a['group_tokens'], a['group_offsets'])
    require(tokens.dtype == np.int32 and tokens.shape == (vocabulary,), 'Group tokens')
    require(offsets.dtype == np.int32 and offsets.shape == (g + 1,), 'Group offsets')
    require(offsets[0] == 0 and offsets[-1] == vocabulary and np.all(np.diff(offsets) > 0), 'Empty/incomplete group')
    require(np.array_equal(np.sort(tokens), np.arange(vocabulary)), 'Vocabulary partition')
    for group in range(g):
        require(np.all(tg[tokens[offsets[group]:offsets[group + 1]]] == group), 'Group mapping mismatch')
