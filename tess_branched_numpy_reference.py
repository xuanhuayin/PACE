import numpy as np

def reference(logits, destination, accept, start):
    x = np.asarray(logits, dtype=np.float64)
    d = np.asarray(destination)
    a = np.asarray(accept, dtype=bool)
    if x.ndim != 2 or not all(x.shape) or (not np.isfinite(x).all()):
        raise ValueError('Finite L,G logits required')
    if d.shape != (len(a), x.shape[1], 2) or d.dtype.kind not in 'iu':
        raise ValueError('S,G,2 integer branches required')
    if (d < -1).any() or (d >= len(a)).any() or (not 0 <= start < len(a)):
        raise ValueError('Invalid state')
    source, category, branch = np.nonzero(d >= 0)
    target = d[source, category, branch]
    shifted = x - x.max(-1, keepdims=True)
    lp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    length, groups = x.shape
    forward = np.full((length + 1, len(a)), -np.inf)
    forward[0, start] = 0.0
    for position in range(length):
        np.logaddexp.at(forward[position + 1], target, forward[position, source] + lp[position, category])
    value = np.logaddexp.reduce(forward[-1, a], initial=-np.inf)
    if not np.isfinite(value):
        return (value, np.zeros_like(x))
    suffix = np.where(a, 0.0, -np.inf)
    marginal = np.zeros_like(x)
    for position in range(length - 1, -1, -1):
        edges = lp[position, category] + suffix[target]
        posterior = np.exp(forward[position, source] + edges - value)
        marginal[position] = np.bincount(category, weights=posterior, minlength=groups)
        previous = np.full(len(a), -np.inf)
        np.logaddexp.at(previous, source, edges)
        suffix = previous
    if not np.isclose(value, suffix[start], atol=1e-09, rtol=1e-11):
        raise ValueError('Independent forward/backward differ')
    gradient = marginal - np.exp(lp) * marginal.sum(-1, keepdims=True)
    return (float(value), gradient)
