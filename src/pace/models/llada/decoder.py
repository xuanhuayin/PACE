from pace.paths import PROJECT_ROOT
import sys
from pathlib import Path
import numpy as np

ROOT = PROJECT_ROOT


from pace.automata.alphabet import character_class, INVALID, N_CLASSES
from pace.automata.finite_product import specialize_utf8
from pace.automata.compiler import minimize_text_dfa

MASK_ID = 126336


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class LLaDADecoderContract:
    """Byte-level-BPE decoder contract (LLaDA, LLaMA-3, GPT-2 style). `eos` is the id at which decoding
    stops (for LLaMA-3-Instruct pass eos=<|eot_id|>); extra ids are treated as skipped specials."""

    def __init__(self, tokenizer, eos=None, extra_special=(MASK_ID,)):
        inv = {v: k for k, v in bytes_to_unicode().items()}
        self.vocab = len(tokenizer)
        self.eos = tokenizer.eos_token_id if eos is None else int(eos)
        self.special = frozenset(set(tokenizer.all_special_ids) | set(extra_special))
        self.bytes = []
        for i in range(self.vocab):
            if i in self.special:
                self.bytes.append(None)
                continue
            s = tokenizer.convert_ids_to_tokens(i)
            try:
                self.bytes.append(bytes((inv[ch] for ch in s)))
            except KeyError:
                self.bytes.append(None)
        self.special = frozenset((i for i in range(self.vocab) if self.bytes[i] is None))

    def decode(self, ids):
        out = bytearray()
        for t in ids:
            if t == self.eos:
                break
            b = self.bytes[t]
            if b is not None:
                out += b
        return out.decode("utf-8", errors="replace")


class ProductMachine:
    """Deterministic (text, pending) machine with vectorized byte steps."""

    def __init__(self, transition, accept, start):
        self.T = np.asarray(transition, dtype=np.int32)
        self.accept = np.asarray(accept, dtype=bool)
        self.table, self.root = specialize_utf8(self.T)
        self.P = len(self.table)
        self.start_text = int(start)
        self.repl = character_class("�")
        reps, labels = ({}, [])
        for c in range(N_CLASSES):
            sig = (self.T[:, c].tobytes(), c == 32)
            labels.append(reps.setdefault(sig, c))
        self.labels = np.asarray(labels, dtype=np.int32)

    def step_bytes(self, text, pend, byte):
        """Vectorized: text, pend int arrays; byte int. Returns new arrays (lossy-UTF-8 approximation)."""
        for _ in range(2):
            after = self.table[pend, byte]
            emit = after < INVALID
            inval = after == INVALID
            cls = np.where(emit, -after - 2, 0)
            text = np.where(emit, self.T[text, cls], text)
            pend = np.where(emit, self.root, np.where(after >= 0, after, pend))
            if not inval.any():
                return (text, pend)
            retry = inval & (pend != self.root)
            text = np.where(inval, self.T[text, self.repl], text)
            pend = np.where(inval, self.root, pend)
            if not retry.any():
                return (text, pend)
            sel = retry
            t2, p2 = self.step_bytes_single(text[sel], pend[sel], byte)
            text = text.copy()
            pend = pend.copy()
            text[sel] = t2
            pend[sel] = p2
            return (text, pend)
        return (text, pend)

    def step_bytes_single(self, text, pend, byte):
        after = self.table[pend, byte]
        emit = after < INVALID
        inval = after == INVALID
        cls = np.where(emit, -after - 2, 0)
        text2 = np.where(emit, self.T[text, cls], text)
        text2 = np.where(inval, self.T[text, self.repl], text2)
        pend2 = np.where(emit | inval, self.root, np.where(after >= 0, after, pend))
        return (text2, pend2)

    def finish(self, text, pend):
        """At EOS/end: an incomplete pending sequence decodes to one replacement char."""
        t = np.where(pend != self.root, self.T[text, self.repl], text)
        return t


def compile_llada_event(decoder, dfa, max_states=150000):
    transition, accept, start = minimize_text_dfa(dfa)
    M = ProductMachine(transition, accept, start)
    T, P, nT = (M.T, M.P, len(M.T))
    groups, gid, token_group, seqs = ([], {}, np.zeros(decoder.vocab, dtype=np.int32), [])
    cache = {}
    base_text = np.arange(nT, dtype=np.int32)
    prod_text = np.repeat(np.arange(nT, dtype=np.int32), P)
    prod_pend = np.tile(np.arange(P, dtype=np.int32), nT)
    for t in range(decoder.vocab):
        b = decoder.bytes[t]
        if t == decoder.eos:
            key = ("eos",)
        elif b is None:
            key = ("skip",)
        else:
            try:
                seq = tuple((int(M.labels[character_class(ch)]) for ch in b.decode("utf-8")))
                if seq not in cache:
                    arr = base_text
                    for cls in seq:
                        arr = T[arr, cls]
                    cache[seq] = ("text", arr.tobytes())
                key = cache[seq]
            except UnicodeDecodeError:
                tt, pp = (prod_text, prod_pend)
                for byte in b:
                    tt, pp = M.step_bytes(tt, pp, int(byte))
                key = ("raw", (tt.astype(np.int64) * P + pp).tobytes())
        if key not in gid:
            gid[key] = len(groups)
            groups.append([])
            seqs.append(b)
        token_group[t] = gid[key]
        groups[gid[key]].append(t)
    G = len(groups)
    maxlen = max((len(b) for b in seqs if b is not None), default=1)
    byte_mat = np.full((G, maxlen), -1, dtype=np.int32)
    for g, b in enumerate(seqs):
        if b is not None:
            byte_mat[g, : len(b)] = np.frombuffer(b, dtype=np.uint8)
    eos_g = int(token_group[decoder.eos])
    skip_mask = np.array([b is None and g != eos_g for g, b in enumerate(seqs)])
    start_state = M.start_text * P + M.root
    index = {start_state: 0}
    states = [start_state]
    rows = []
    STOP = -2
    cursor = 0
    while cursor < len(states):
        s = states[cursor]
        cursor += 1
        text0, pend0 = (s // P, s % P)
        text = np.full(G, text0, dtype=np.int32)
        pend = np.full(G, pend0, dtype=np.int32)
        for j in range(maxlen):
            col = byte_mat[:, j]
            idx = np.flatnonzero(col >= 0)
            if not len(idx):
                break
            for bval in np.unique(col[idx]):
                sel = idx[col[idx] == bval]
                tt, pp = M.step_bytes(text[sel], pend[sel], int(bval))
                text[sel] = tt
                pend[sel] = pp
        row = text.astype(np.int64) * P + pend
        row[skip_mask] = s
        ft = int(M.finish(np.array([text0]), np.array([pend0]))[0])
        row[eos_g] = STOP if M.accept[ft] else -1
        for v in np.unique(row):
            v = int(v)
            if v >= 0 and v not in index:
                if len(index) >= max_states:
                    raise ValueError("LLaDA product state limit")
                index[v] = len(states)
                states.append(v)
        rows.append(row)
    S = len(states) + 1
    stop_id = S - 1
    lut = np.full(nT * P + 1, -1, dtype=np.int32)
    for v, i in index.items():
        lut[v] = i
    dest = np.full((S, G, 2), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        d = np.where(row >= 0, lut[np.clip(row, 0, nT * P)], -1).astype(np.int32)
        d[row == STOP] = stop_id
        dest[i, :, 0] = d
    dest[stop_id, :, 0] = stop_id
    sig, merged, cols, mapping = ({}, [], [], np.zeros(G, dtype=np.int32))
    for g in range(G):
        k = dest[:, g, :].tobytes()
        if k not in sig:
            sig[k] = len(merged)
            merged.append([])
            cols.append(g)
        merged[sig[k]].extend(groups[g])
        mapping[g] = sig[k]
    acc = np.zeros(S, dtype=bool)
    acc[stop_id] = True
    from types import SimpleNamespace

    return SimpleNamespace(
        destination=dest[:, cols, :].copy(),
        accept=acc,
        start=0,
        token_group=mapping[token_group],
        groups=tuple((tuple(sorted(g)) for g in merged)),
        text_states=nT,
        product_states=S,
    )


def accepts(event, ids):
    states = {int(event.start)}
    for t in ids:
        g = int(event.token_group[t])
        nxt = set()
        for s in states:
            for d in event.destination[s, g]:
                if d >= 0:
                    nxt.add(int(d))
        states = nxt
        if not states:
            return False
    return any((bool(event.accept[s]) for s in states))


def save_graph(path, q):
    tokens = np.asarray([t for g in q.groups for t in g], dtype=np.int32)
    offsets = np.r_[0, np.cumsum([len(g) for g in q.groups])].astype(np.int32)
    np.savez_compressed(
        path,
        destination=q.destination,
        accept=q.accept,
        start=np.array(q.start, dtype=np.int32),
        token_group=q.token_group,
        group_tokens=tokens,
        group_offsets=offsets,
    )
