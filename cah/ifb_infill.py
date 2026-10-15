from pathlib import Path
import json
import math
import os
import re
import string
import subprocess
import sys
import time
import unicodedata
from collections import Counter
ROOT = str(Path(__file__).resolve().parents[1])
for _p in (ROOT, ROOT + '/cah'):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('NLTK_DATA', str(Path(__file__).resolve().parents[1] / 'assets/nltk'))
IFB_PY = os.environ.get('IFBENCH_PYTHON', sys.executable)
SERVER = ROOT + '/cah/ifb_oracle_server.py'
PUNCT = string.punctuation
PUNCT_SPACE = PUNCT + ' '
LAST = {}
ATTEMPTS = 3
SCALES = (1.0, 1.5, 1.25)
TEMPS = (0.0, 0.0, 0.7)
BLOCK = 16
MAX_STEPS = 96
PENALTY = (2.5, 7.5, 0.5, 3.0)
REFINES = 2
EOS_TAIL = 8
NAMES = ['Emma', 'Liam', 'Sophia', 'Jackson', 'Olivia', 'Noah', 'Ava', 'Lucas', 'Isabella', 'Mason', 'Mia', 'Ethan', 'Charlotte', 'Alexander', 'Amelia', 'Benjamin', 'Harper', 'Leo', 'Zoe', 'Daniel', 'Chloe', 'Samuel', 'Lily', 'Matthew', 'Grace', 'Owen', 'Abigail', 'Gabriel', 'Ella', 'Jacob', 'Scarlett', 'Nathan', 'Victoria', 'Elijah', 'Layla', 'Nicholas', 'Audrey', 'David', 'Hannah', 'Christopher', 'Penelope', 'Thomas', 'Nora', 'Andrew', 'Aria', 'Joseph', 'Claire', 'Ryan', 'Stella', 'Jonathan']
PALINDROMES = ['level', 'civic', 'radar', 'refer', 'kayak', 'madam', 'rotor', 'stats', 'tenet', 'minim', 'solos', 'sagas', 'deified', 'reviver', 'racecar', 'redder', 'repaper', 'rotator', 'hannah', 'malayalam']
PRONOUNS = {'i', 'me', 'we', 'us', 'you', 'he', 'him', 'she', 'her', 'it', 'they', 'them', 'my', 'mine', 'our', 'ours', 'your', 'yours', 'his', 'hers', 'its', 'their', 'theirs', 'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what', 'whoever', 'whomever', 'whatever', 'whichever', 'anybody', 'anyone', 'anything', 'everybody', 'everyone', 'everything', 'nobody', 'nothing', 'somebody', 'someone', 'something', 'each', 'either', 'neither', 'both', 'all', 'some', 'any', 'none'}
STOP = {'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these', 'those', 'which', 'who', 'whom', 'what', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'we', 'our', 'you', 'your', 'i', 'my', 'me', 'us', 'him', 'not', 'no', 'so', 'if', 'then', 'than', 'also', 'can', 'could', 'would', 'should', 'will', 'may', 'might', 'must', 'do', 'does', 'did', 'have', 'has', 'had', 'there', 'here', 'when', 'where', 'while', 'how', 'why', 'all', 'any', 'some', 'each', 'more', 'most', 'such', 'into', 'about', 'over', 'under', 'between', 'through', 'during', 'out', 'up', 'down', 'only', 'just', 'very', 'even', 'both', 'other', 'one', 'own', 'same', 'too', 'yet', 'nor'}
COMMON = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'but', 'or', 'so', 'as', 'if', 'it', 'its', 'we', 'you', 'they', 'he', 'she', 'i', 'my', 'our', 'your', 'their', 'his', 'her', 'there', 'here', 'when', 'while', 'with', 'by', 'from', 'one', 'each', 'every', 'some', 'many', 'most', 'all', 'what', 'how', 'why', 'who', 'which', 'is', 'are', 'was', 'were', 'be', 'do', 'not', 'no', 'yes', 'first', 'finally', 'overall', 'however', 'also', 'then', 'now', 'after', 'before', 'once', 'today'}

def T(s):
    return {'k': 'T', 's': s}

def L(s, after=None):
    return {'k': 'L', 's': s, 'after': after}

def H(n, **opts):
    d = {'k': 'H', 'n': max(1, int(n))}
    d.update(opts)
    return d

def _sc(n, a):
    return max(1, int(round(n * SCALES[a])))

class Oracle:
    """Official IFBench checkers + deterministic transforms: in process when the official package imports (ifbench-env),
    otherwise through a persistent ifb_oracle_server.py child process in ifbench-env (JSON lines)."""

    def __init__(self, mode='auto'):
        self.O = self.TR = self.proc = None
        if mode in ('auto', 'inproc'):
            try:
                import ifb_official as O
                import ifb_transforms as TR
                self.O, self.TR = (O, TR)
            except Exception:
                if mode == 'inproc':
                    raise
                self.O = self.TR = None
        if self.O is None:
            env = dict(os.environ, NLTK_DATA=str(Path(__file__).resolve().parents[1] / 'assets/nltk'))
            env.pop('PYTHONPATH', None)
            self.proc = subprocess.Popen([IFB_PY, '-u', SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
            assert self._call(op='ping')['ok']
        self._names = set(self.TR.TRANSFORMS) if self.TR else set(self._call(op='list')['transforms'])

    def _call(self, **req):
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('oracle server died')
        out = json.loads(line)
        if 'error' in out:
            raise RuntimeError('oracle: ' + out['error'])
        return out

    def check(self, iid, kw, text, prompt=None):
        if self.O:
            return self.O.official(iid, self.O.clean_kwargs(kw), text, prompt)
        return bool(self._call(op='check', iid=iid, kw=kw, text=text, prompt=prompt)['ok'])

    def checks(self, pairs, text, prompt=None):
        pairs = [[i, kw] for i, kw in pairs]
        if self.O:
            return {i: self.O.official(i, self.O.clean_kwargs(kw), text, prompt) for i, kw in pairs}
        return self._call(op='checks', pairs=pairs, text=text, prompt=prompt)['ok']

    def has_transform(self, iid):
        return iid in self._names

    def transform(self, iid, text, kw, prompt=None):
        if self.TR:
            out = self.TR.TRANSFORMS[iid](text, self.O.clean_kwargs(kw), prompt)
            return (out, dict(self.TR.LAST))
        r = self._call(op='transform', iid=iid, kw=kw, text=text, prompt=prompt)
        return (r['text'], r['last'])

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
_ORACLE = None

def get_oracle():
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = Oracle()
    return _ORACLE

def clean_kw(kw):
    return {k: v for k, v in (kw or {}).items() if v is not None}

def _int(kw, key, default=0):
    try:
        return int(float(kw.get(key, default)))
    except (TypeError, ValueError):
        return default

def _latin_ok(ch):
    """Latin script + Latin-1, general punctuation / symbols / arrows / math (U+2000-2BFF), emoji; other scripts and
    their punctuation or combining marks (CJK, Cyrillic, Devanagari, ...) are not allowed in holes."""
    o = ord(ch)
    return o < 592 or 7680 <= o <= 7935 or 8192 <= o <= 11263 or (o >= 126976) or (o in (65039, 8205))

class Adapter:
    """Engine-specific facts: tokenizer, mask/eos ids, forward pass, prompt encoding, decoded vocabulary + token classes."""

    def __init__(self, kind, tok, engine=None, device='cuda'):
        import numpy as np
        self.np = np
        self.kind, self.tok, self.e, self.device = (kind, tok, engine, device)
        if kind == 'llada':
            self.mask, self.eos, self.V, self.shift, self.conf = (126336, 126081, 126349, False, 'maxprob')
            self.extra_ban = ()
        elif kind == 'dream':
            from dream_common import MASK_ID, EOS_ID, VOCAB, IM_END, SKIP_IDS
            self.mask, self.eos, self.V, self.shift, self.conf = (MASK_ID, EOS_ID, VOCAB, True, 'entropy')
            self.extra_ban = (IM_END,) + tuple(SKIP_IDS)
        elif kind == 'tess':
            self.eos = tok.eos_token_id
            self.V = engine.model.config.vocab_size if engine is not None else len(tok)
            pad = tok.pad_token_id
            self.mask = pad if pad is not None and pad < self.V else self.eos
            self.shift, self.conf, self.extra_ban = (False, None, ())
        else:
            raise ValueError(kind)
        self.sent = self._raw('\n')
        self._sent_txt = self.tok.decode(self.sent, clean_up_tokenization_spaces=False)
        self._strs = None
        self._cls = {}
        self._vec = {}
        self._words = {}

    def _raw(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def encode_piece(self, s):
        """Token ids of `s` as it would be tokenized in running text (sentinel trick: no dummy prefix / BOS)."""
        if not s:
            return []
        full = self._raw('\n' + s)
        if full[:len(self.sent)] == self.sent:
            return full[len(self.sent):]
        return self._raw(s)

    def decode_piece(self, ids):
        if not ids:
            return ''
        d = self.tok.decode(list(self.sent) + list(ids), clean_up_tokenization_spaces=False)
        if d.startswith(self._sent_txt):
            return d[len(self._sent_txt):]
        return self.tok.decode(list(ids), clean_up_tokenization_spaces=False)

    def prompt_ids(self, prompt):
        if self.kind == 'tess':
            from tess_ifeval_prompt_contract import format_prompt
            return self.tok.encode(format_prompt(prompt, self.tok.bos_token), add_special_tokens=False)
        if self.e is not None:
            return list(self.e.prompt_ids(prompt))
        text = self.tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
        if self.kind == 'dream':
            return self.tok(text, add_special_tokens=False)['input_ids']
        return self.tok(text)['input_ids']

    def strs(self):
        if self._strs is None:
            V, nt = (self.V, len(self.tok))
            pairs = [list(self.sent) + [i] for i in range(min(V, nt))]
            dec = self.tok.batch_decode(pairs, clean_up_tokenization_spaces=False)
            b = self._sent_txt
            self._strs = [d[len(b):] if d.startswith(b) else d for d in dec] + [''] * max(0, V - nt)
        return self._strs

    def specials(self):
        sp = set(self.tok.all_special_ids) | set(getattr(self.tok, 'added_tokens_decoder', {}).keys())
        sp |= set(self.extra_ban) | {self.mask}
        sp |= set(range(len(self.tok), self.V))
        return {i for i in sp if i is not None and 0 <= i < self.V}

    def cls(self, name):
        """Boolean numpy vector (V,) for a token class."""
        if name in self._cls:
            return self._cls[name]
        np, S, V = (self.np, self.strs(), self.V)

        def f(pred):
            return np.fromiter((pred(s) for s in S), dtype=bool, count=V)
        if name == 'special':
            v = np.zeros(V, bool)
            v[list(self.specials())] = True
        elif name == 'eos':
            v = np.zeros(V, bool)
            v[self.eos] = True
        elif name == 'nl':
            v = f(lambda s: '\n' in s or '\r' in s)
        elif name == 'spaces':
            v = f(lambda s: bool(s) and (not s.strip()) and ('\n' not in s))
        elif name == 'endpunct':
            v = f(lambda s: bool(s.strip()) and (s.rstrip()[-1] in PUNCT or not s.strip(PUNCT_SPACE)) or not s.strip())
        elif name == 'quotes':
            v = f(lambda s: '"' in s or "'" in s)
        elif name == 'brackets':
            v = f(lambda s: any((c in '()[]{}' for c in s)))
        elif name == 'digits':
            v = f(lambda s: any((c.isdecimal() for c in s)))
        elif name == 'nonlatin':
            v = f(lambda s: '�' in s or not all((_latin_ok(c) for c in s)))
        elif name == 'wordstart':
            v = f(lambda s: bool(s) and (s[0].isalnum() or s[0] == '_'))
        elif name == 'spacestart':
            v = f(lambda s: bool(s) and s[0].isspace())
        elif name == 'notspacepunct':
            v = f(lambda s: bool(s) and (not s[0].isspace()) and any((c.isalnum() for c in s)))
        elif name == 'digit1':
            v = f(lambda s: bool(s) and s.isdecimal() and s.isascii())
        elif name == 'digit_or_boundary':
            v = f(lambda s: bool(s) and (s.isdecimal() and s.isascii() or (not any((c.isdecimal() for c in s)) and (not (s[0].isalnum() or s[0] == '_')))))
        elif name.startswith('kw:') or name.startswith('kwf:'):
            frag = name.startswith('kwf:')
            words = [w.lower() for w in name.split(':', 1)[1].split('\x1f') if w]
            v = f(lambda s: s.strip().lower() in words)
            if frag:
                for i, s in enumerate(S):
                    st = s.strip().lower()
                    if st.isalpha() and any((len(st) >= 3 and s[:1].isspace() and w.startswith(st) or (len(st) >= 4 and (w.startswith(st) or w.endswith(st))) for w in words)):
                        v[i] = True
                for w in words:
                    for var in (' ' + w, w, ' ' + w.capitalize(), w.capitalize(), ' ' + w.upper()):
                        ids = self.encode_piece(var)
                        if ids:
                            v[ids[0]] = True
        elif name.startswith('names:'):
            want = {' ' + n for n in name[6:].split('\x1f') if n}
            v = f(lambda s: s in want)
        elif name == 'palin':
            ok = set(PALINDROMES)
            v = f(lambda s: len(s) > 1 and s[0] == ' ' and (s[1:].lower() in ok))
        elif name.startswith('meta:'):
            mw = {w for w in name[5:].split('\x1f') if w}
            v = f(lambda s: s.strip().lower() in mw)
        elif name == 'numwords':
            nw = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion'}
            v = f(lambda s: s.strip().lower() in nw)
        elif name == 'pronoun':
            v = f(lambda s: s.strip().lower() in PRONOUNS and (s[:1].isspace() or s.strip() == s))
        else:
            raise KeyError(name)
        if name not in ('special', 'eos'):
            v[list(self.specials())] = False
        self._cls[name] = v
        return v

    def class_words(self, name):
        """Distinct lower-cased words available as single tokens in a slot class."""
        S = self.strs()
        return sorted({S[i].strip().lower() for i in self.np.nonzero(self.cls(name))[0]})

    def single_token(self, piece):
        ids = self.encode_piece(piece)
        return len(ids) == 1

    def vector(self, key):
        """(allowed bool (V,), bias float32 (V,) or None) for a hole-position key (bans, only, eos_ok, boost)."""
        if key in self._vec:
            return self._vec[key]
        np = self.np
        bans, only, eos_ok, boost = key
        banned = self.cls('special').copy()
        for b in bans:
            banned |= self.cls(b)
        allowed = ~banned
        if only:
            allowed &= self.cls(only)
        allowed[self.eos] = bool(eos_ok)
        if not allowed.any():
            raise ValueError('empty allowed set for %r' % (key,))
        bias = None
        if boost:
            cname, strength = boost.rsplit('@', 1)
            bias = self.cls(cname).astype(np.float32) * float(strength)
        self._vec[key] = (allowed, bias)
        return self._vec[key]

    def forward(self, x):
        return self.e.model(x).logits[0]
_ADAPTERS = {}

def adapter(engine):
    a = _ADAPTERS.get(id(engine))
    if a is None:
        name = type(engine).__name__
        kind = {'LLaDAEngine': 'llada', 'DreamEngine': 'dream', 'Tess': 'tess'}[name]
        a = _ADAPTERS[id(engine)] = Adapter(kind, engine.tok, engine)
    return a
_AFTER = {'word': 'wordstart', 'space': 'notspacepunct', 'nospace': 'spacestart'}

def build_canvas(ad, segs, ctx_bans=()):
    """-> dict(ids (answer ids, holes = placeholder), pos (hole positions in the answer), keys (per position),
    holes [(seg index, start, n)], hole_of (per position), groups {group: [position index]})."""
    ids, pos, keys, holes, hole_of, groups = ([], [], [], [], [], {})
    pending = None
    for si, s in enumerate(segs):
        if s['k'] in ('T', 'L'):
            piece = ad.encode_piece(s['s'])
            ids += piece
            if piece:
                pending = s.get('after') if s['k'] == 'L' else None
            continue
        start = len(ids)
        for j in range(s['n']):
            cb = set(ctx_bans)
            if (s.get('only') or '').startswith('digit') or s.get('free_digits'):
                cb.discard('digits')
            bans = set(s.get('ban', ())) | cb
            if j == 0 and pending:
                bans.add(_AFTER[pending])
            if s.get('end_ban') and j == s['n'] - 1:
                bans.add('endpunct')
            eos_ok = bool(s.get('eos')) and j >= s.get('eos_min', 0)
            keys.append((tuple(sorted(bans)), s.get('only'), eos_ok, s.get('boost')))
            if s.get('distinct'):
                groups.setdefault(s['distinct'], []).append(len(pos))
            pos.append(start + j)
            hole_of.append(len(holes))
        ids += [ad.mask] * s['n']
        holes.append((si, start, s['n']))
        pending = s.get('after')
    return dict(ids=ids, pos=pos, keys=keys, holes=holes, hole_of=hole_of, groups=groups)

def assemble(ad, segs, cv, toks):
    """Output text + log view.  T segments verbatim, literals verbatim, holes decoded (cut at EOS)."""
    p2t = dict(zip(cv['pos'], toks))
    hole_by_seg = {si: (start, n) for si, start, n in cv['holes']}
    out, view, regions, cur = ([], [], [], [])
    n_items = sum((1 for s in segs if s['k'] == 'L' and any((c.isalnum() for c in s['s'])) or (s['k'] == 'H' and s.get('only'))))
    for si, s in enumerate(segs):
        if s['k'] == 'T':
            out.append(s['s'])
            if cur:
                regions.append(''.join(cur))
                cur = []
            if s['s']:
                view.append('…')
            continue
        if s['k'] == 'L':
            out.append(s['s'])
            cur.append(s['s'])
            view.append(s['s'])
            continue
        start, n = hole_by_seg[si]
        t = [p2t[start + j] for j in range(n)]
        if ad.eos in t:
            t = t[:t.index(ad.eos)]
        piece = ad.decode_piece(t)
        out.append(piece)
        cur.append(piece)
        view.append('⟦' + piece + '⟧')
    if cur:
        regions.append(''.join(cur))
    v = ''.join(view)
    v = re.sub('(…)+', '…', v)
    return (''.join(out), dict(regions=regions, view=v, n_items=n_items, context=''.join((s['s'] for s in segs if s['k'] == 'T'))))

def _mask_predict(ad, pids, cv, seed, temperature):
    """LLaDA / Dream: masked positions filled one (or a few) at a time, most confident first, inside a left-to-right
    window over the first BLOCK masked positions of the canvas (as LLaDA's semi-AR blocks); fixed tokens stay;
    bans/only/boost and a presence penalty on already-written content tokens are applied at every step."""
    import torch
    dev = ad.device
    Lp = len(pids)
    x = torch.tensor([list(pids) + cv['ids'] + [ad.eos] * EOS_TAIL], device=dev)
    n = len(cv['pos'])
    P = torch.tensor([Lp + p for p in cv['pos']], device=dev)
    rows = P - 1 if ad.shift else P
    uk = sorted(set(cv['keys']), key=str)
    kidx = {k: i for i, k in enumerate(uk)}
    vecs = [ad.vector(k) for k in uk]
    A = torch.from_numpy(ad.np.stack([v[0] for v in vecs])).to(dev)
    sel_k = torch.tensor([kidx[k] for k in cv['keys']], device=dev)
    allowed = A[sel_k].clone()
    bias = None
    if any((v[1] is not None for v in vecs)):
        B = torch.from_numpy(ad.np.stack([v[1] if v[1] is not None else ad.np.zeros(ad.V, ad.np.float32) for v in vecs])).to(dev)
        bias = B[sel_k]
    group_of, word_ids = ({}, {})
    S = ad.strs() if cv['groups'] else None
    for g, members in cv['groups'].items():
        for m in members:
            group_of[m] = g
        only = cv['keys'][members[0]][1]
        wi = {}
        if only:
            for i in ad.np.nonzero(ad.cls(only))[0].tolist():
                wi.setdefault(S[i].strip().lower(), []).append(i)
        word_ids[g] = wi
    masked = [True] * n
    hole_members = {}
    for j, h in enumerate(cv['hole_of']):
        hole_members.setdefault(h, []).append(j)
    out = [None] * n
    per_step = max(1, math.ceil(n / MAX_STEPS))
    S_all = ad.strs()
    pen = torch.zeros(ad.V, device=dev)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    with torch.no_grad():
        while any(masked):
            lg = ad.forward(x)[rows][:, :ad.V].float() - pen
            if bias is not None:
                lg = lg + bias
            lg = lg.masked_fill(~allowed, float('-inf'))
            if temperature > 0:
                u = torch.rand(lg.shape, generator=gen, device=dev).clamp_(1e-10, 1 - 1e-10)
                tok = (lg / temperature - torch.log(-torch.log(u))).argmax(-1)
            else:
                tok = lg.argmax(-1)
            if ad.conf == 'entropy':
                q = torch.softmax(lg / 0.1, -1)
                conf = (q * torch.log(q.clamp_min(1e-30))).sum(-1)
            else:
                conf = torch.softmax(lg, -1).gather(-1, tok[:, None])[:, 0]
            cand = torch.zeros(n, dtype=torch.bool, device=dev)
            cand[[j for j in range(n) if masked[j]][:BLOCK]] = True
            conf = conf.masked_fill(~cand, float('-inf'))
            k = min(per_step, int(cand.sum()))
            for j in torch.topk(conf, k).indices.tolist():
                t = int(tok[j])
                out[j] = t
                masked[j] = False
                x[0, P[j]] = t
                st = S_all[t].strip()
                if st.isalnum():
                    content = st.isdigit() or (len(st) >= 3 and st.lower() not in STOP)
                    step_, cap = (PENALTY[0], PENALTY[1]) if content else (PENALTY[2], PENALTY[3])
                    pen[t] = min(cap, float(pen[t]) + step_)
                g = group_of.get(j)
                if g is not None:
                    same = torch.tensor(word_ids[g].get(S[t].strip().lower(), [t]), device=dev)
                    for m in cv['groups'][g]:
                        if masked[m]:
                            row = allowed[m].clone()
                            row[same] = False
                            if row.any():
                                allowed[m] = row
                if t == ad.eos:
                    for m in hole_members[cv['hole_of'][j]]:
                        if m > j and masked[m]:
                            out[m] = ad.eos
                            masked[m] = False
                            x[0, P[m]] = ad.eos
    return out

def _tess_fill(ad, pids, cv, seed, temperature):
    """TESS-2 simplex diffusion over the holes (span_mask = holes, clean tokens elsewhere), 100 steps from pure noise.
    Mirrors Tess.harmonize_batch (bidirectional adapter, top-p 0.99 projection, self-conditioning) + per-step bans."""
    import torch
    from sdlm.schedulers import TokenWiseSimplexDDPMScheduler
    from sdlm.inference.inference_utils import logits_projection
    from sdlm.utils import scale, self_condition_preds, convert_to_simplex
    from tess_bidirectional_attention_adapter import bidirectional_attention
    model = ad.e.model
    P0 = len(pids)
    ids = torch.tensor([list(pids) + cv['ids'] + [ad.eos] * EOS_TAIL], device='cuda')
    Ltot, V = (ids.shape[1], model.config.vocab_size)
    rows = torch.tensor([P0 + p for p in cv['pos']], device='cuda')
    span = torch.zeros((1, Ltot), dtype=torch.bool, device='cuda')
    span[0, rows] = True
    uk = sorted(set(cv['keys']), key=str)
    kidx = {k: i for i, k in enumerate(uk)}
    vecs = [ad.vector(k) for k in uk]
    sel = [kidx[k] for k in cv['keys']]
    allowed = torch.from_numpy(ad.np.stack([v[0] for v in vecs])[sel]).to('cuda')
    bias = None
    if any((v[1] is not None for v in vecs)):
        bias = torch.from_numpy(ad.np.stack([v[1] if v[1] is not None else ad.np.zeros(V, ad.np.float32) for v in vecs])[sel]).to('cuda')
    sched = TokenWiseSimplexDDPMScheduler(device='cuda', simplex_value=5.0, num_train_timesteps=100, beta_schedule='squaredcos_improved_ddpm')
    gen = torch.Generator(device='cuda').manual_seed(int(seed))
    top_p = 0.99 if temperature == 0 else 0.9
    proj = lambda z: logits_projection(z, 'top_p', top_p, 5.0, 1.0)
    tstar = 99
    with torch.no_grad(), bidirectional_attention(model), torch.autocast('cuda', dtype=torch.bfloat16):
        clean = convert_to_simplex(ids, 5.0, V).float()
        simplex = sched.add_noise(clean, 5.0 * torch.randn(clean.shape, generator=gen, device='cuda'), torch.tensor([tstar], device='cuda'))
        previous = torch.zeros((1, Ltot, V), device='cuda')
        hidden = None
        last = None
        for t in range(tstar, -1, -1):
            tt = torch.full((1, Ltot), t, device='cuda')
            out = model(input_ids=ids, span_mask=span, simplex=simplex, timesteps=scale(tt, 100), previous_pred=previous, classifier_free_guidance=False, reduce_loss='none', max_timestep=100, previous_hidden=hidden)
            logits = out.logits.float()
            hidden = out.hidden_states
            sub = logits[0, rows]
            if bias is not None:
                sub = sub + bias
            logits[0, rows] = sub.masked_fill(~allowed, -10000.0)
            last = logits[0, rows]
            logits = logits.to(torch.bfloat16)
            previous = self_condition_preds(model.config.self_condition, logits, proj)
            simplex = sched.step(proj(logits), tt, tt - 1, 5.0 * torch.randn(clean.shape, generator=gen, device='cuda'), generator=gen).prev_sample
    toks = last.argmax(-1).tolist()
    hole_members = {}
    for j, h in enumerate(cv['hole_of']):
        hole_members.setdefault(h, []).append(j)
    for mem in hole_members.values():
        hit = False
        for j in mem:
            hit = hit or toks[j] == ad.eos
            if hit:
                toks[j] = ad.eos
    return toks

def fill(engine, prompt, segs, *, seed=0, temperature=0.0, ctx_bans=(), ad=None):
    """Fill the holes of `segs` with the DLM; returns (text, info)."""
    ad = ad or adapter(engine)
    cv = build_canvas(ad, segs, ctx_bans)
    t0 = time.perf_counter()
    if not cv['pos']:
        toks = []
    else:
        pids = ad.prompt_ids(prompt)
        toks = _tess_fill(ad, pids, cv, seed, temperature) if ad.kind == 'tess' else _mask_predict(ad, pids, cv, seed, temperature)
    text, info = assemble(ad, segs, cv, toks)
    info.update(n_masked=len(cv['pos']), seconds=time.perf_counter() - t0)
    return (text, info)

def _kw_count(text, k):
    return len(re.findall('\\b{}\\b'.format(re.escape(k)), text, flags=re.IGNORECASE))

def _names_present(text):
    return [nm for nm in NAMES if re.search('\\b{}\\b'.format(re.escape(nm)), text)]

def _palindromes(text):
    words = text.translate(str.maketrans('', '', PUNCT)).lower().split()
    return [w for w in words if w == w[::-1] and len(w) >= 5]

def _unique_words(text):
    return {w.strip(PUNCT_SPACE) for w in text.lower().split()}
_NUM_UNIT = re.compile('\\d(?:[%s]*\\d)*' % re.escape(PUNCT))

def _number_units(text):
    """Spans that the checker counts as one number each (punctuation is deleted first, then \\d+)."""
    return [(m.start(), m.end()) for m in _NUM_UNIT.finditer(text)]

def _pronoun_count(text):
    try:
        import nltk
        words = nltk.word_tokenize(text.replace('/', ' ').lower())
    except Exception:
        words = re.findall('\\w+', text.lower())
    return sum((1 for w in words if w in PRONOUNS))

def _punct_missing(text):
    """(needs_interrobang, missing marks) under the checker's semantics."""
    inter = '!?' in text or '?!' in text or '‽' in text
    nv = text.replace('?!', '', 1)
    if len(nv) == len(text):
        nv = text.replace('!?', '', 1)
    missing = [c for c in '.,;:?!' if c not in nv]
    return (not inter, missing)

def _sent_spans(text):
    try:
        import nltk
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = [s for s in re.split('(?<=[.!?])\\s+', text) if s]
    out, pos = ([], 0)
    for s in sents:
        i = text.find(s, pos)
        if i < 0:
            i = pos
        out.append((i, i + len(s)))
        pos = i + len(s)
    return out

def _round_robin(deficits):
    """[(item, k)] -> interleaved list with no item twice in a row when avoidable."""
    left = {it: k for it, k in deficits if k > 0}
    out = []
    while left:
        for it in sorted(left, key=lambda z: -left[z]):
            if out and out[-1] == it and (len(left) > 1):
                continue
            out.append(it)
            left[it] -= 1
            if not left[it]:
                del left[it]
            break
    return out

def _chunks(xs, g):
    return [xs[i:i + g] for i in range(0, len(xs), g)]

def _rot(seq, key):
    """Deterministic per-prompt rotation (so not every response gets the same first names)."""
    if not seq:
        return seq
    h = sum((ord(c) for c in str(key))) % len(seq)
    return seq[h:] + seq[:h]
META = {'count:keywords_multiple': 'keyword keywords', 'count:person_names': 'names', 'words:palindrome': 'palindrome palindromes palindromic', 'count:pronouns': 'pronoun pronouns', 'count:numbers': 'number numbers numeral numerals digit digits', 'count:punctuation': 'punctuation semicolon semicolons colon colons interrobang interrobangs exclamation question interrogative comma commas period mark marks', 'format:parentheses': 'parentheses parenthesis parenthetical brackets bracket braces brace nested nest nesting levels', 'format:quotes': 'quote quotes quotation quotations nested nesting levels', 'words:paragraph_last_first': 'paragraph paragraphs'}

def meta(iid):
    return 'meta:' + '\x1f'.join(META.get(iid, '').split())
_LISTLINE = re.compile('^\\s*(?:[-*•]|\\d{1,3}[.)])\\s+')
_TERMINAL = '.!?:;)"\'*`'

def _par_ends(text):
    """Offsets right after the last sentence of each prose paragraph (blocks separated by blank lines whose last line
    ends with . ! ?, is not a list item and has >= 6 words) plus the end of the text if it ends a sentence."""
    body = text.rstrip()
    ends = []
    for m in re.finditer('\\n[ \\t]*\\n', body):
        seg = body[:m.start()].rstrip()
        last = seg[seg.rfind('\n') + 1:].strip()
        if last and last[-1] in '.!?' and (not _LISTLINE.match(last)) and (len(last.split()) >= 6):
            ends.append(len(seg))
    last = body[body.rfind('\n') + 1:].strip()
    if last and last[-1] in '.!?' and (not _LISTLINE.match(last)):
        ends.append(len(body))
    return sorted(set(ends))

def _append_prefix(text, a):
    """Segments that open an appended block: a truncated last sentence (native cut at the length limit) is first
    completed by the model (hole + '.'), then a new paragraph starts."""
    body = text.rstrip()
    segs = [T(body)]
    if body and body[-1] not in _TERMINAL:
        segs += [H(_sc(10, a), ban={'nl'}, end_ban=True), L('.')]
    if body:
        segs.append(L('\n\n', after='nospace'))
    return segs

def _sentence(items, a, lead, mid, tail, bans=(), boost=None):
    """One sentence canvas: hole, item, hole, item, ..., hole, '.'; items = segment lists (literals or slots)."""
    b = set(bans) | {'nl'}
    segs = [H(_sc(lead, a), ban=b, boost=boost)]
    for i, it in enumerate(items):
        segs += it
        last = i == len(items) - 1
        segs.append(H(_sc(tail if last else mid, a), ban=b, end_ban=last, boost=boost))
    segs.append(L('.'))
    return segs

def _compose(text, edits):
    """edits: [(start, end, segs)] (start == end: insertion; same offset keeps list order) -> segment list."""
    segs, prev = ([], 0)
    for s, e, ins in sorted(edits, key=lambda z: (z[0], z[1])):
        if s > prev:
            segs.append(T(text[prev:s]))
        segs += ins
        prev = max(prev, e)
    if prev < len(text):
        segs.append(T(text[prev:]))
    return segs

def _place(text, sentences, mode, a, edits=()):
    """Insert sentence canvases: mode 'spread' = one at the end of each of several prose paragraphs (evenly spaced, in
    text order; the sentence continues that paragraph), else (or < 2 paragraphs) appended as a new final paragraph."""
    edits = list(edits)
    ends = _par_ends(text) if mode == 'spread' else []
    if len(ends) >= 2:
        n, P = (len(sentences), len(ends))
        idx = [round(i * (P - 1) / (n - 1)) for i in range(n)] if 1 < n <= P else [P - 1] if n == 1 else [i * P // n for i in range(n)]
        edits += [(ends[i], ends[i], s) for s, i in zip(sentences, idx)]
        return _compose(text, edits)
    body = text.rstrip()
    pre = _append_prefix(text, a)[1:]
    block = pre + [x for s in sentences for x in s]
    return _compose(text, edits + [(len(body), len(text), block + [T(text[len(body):])])])

def _mode(a):
    return 'spread' if a < 2 else 'append'

def _clause_points(text, k):
    """k offsets (the final punctuation mark of distinct prose sentences, evenly spread over the text) where a clause can
    be inserted: '... of the human condition⟦, a⟧ paradox⟦ that shapes us⟧.'"""
    cands = []
    for s0, e0 in _sent_spans(text):
        sent = text[s0:e0].rstrip()
        if len(sent.split()) < 6 or sent[-1] not in '.!?' or '\n' in sent.strip() or _LISTLINE.match(sent):
            continue
        cands.append(s0 + len(sent) - 1)
    if not cands or k <= 0:
        return []
    C = len(cands)
    idx = [round(i * (C - 1) / (k - 1)) for i in range(k)] if 1 < k <= C else [C - 1] if k == 1 else [i * C // k for i in range(k)]
    return [cands[i] for i in idx]

def _clauses(text, items, a, bans=(), boost=None, edits=()):
    """Insert each item (segment list) as a model-written clause before the final punctuation of a different sentence.
    Returns None when the text has no suitable sentence."""
    pts = _clause_points(text, len(items))
    if not pts:
        return None
    b = set(bans) | {'nl'}
    ins = [(p, p, [H(_sc(3, a), ban=b, boost=boost)] + it + [H(_sc(4, a), ban=b, end_ban=True, boost=boost)]) for p, it in zip(pts, items)]
    return _compose(text, list(edits) + ins)

def b_keywords(text, kw, row, a, ctx):
    pairs = [(str(kw.get('keyword%d' % i) or '').strip(), n) for i, n in zip(range(1, 6), (1, 2, 3, 5, 7))]
    pairs = [(k, n) for k, n in pairs if k]
    kwban = 'kw:' + '\x1f'.join((k for k, _ in pairs))
    kwfrag = 'kwf:' + '\x1f'.join((k for k, _ in pairs))
    edits = []
    for k, n in pairs:
        for m in list(re.finditer('\\b{}\\b'.format(re.escape(k)), text, flags=re.IGNORECASE))[n:]:
            s = m.start() - 1 if m.start() > 0 and text[m.start() - 1] == ' ' else m.start()
            edits.append((s, m.end(), [H(2, ban={kwfrag, 'nl'})]))
    adds = _round_robin([(k, n - min(n, _kw_count(text, k))) for k, n in pairs])
    if not adds and (not edits):
        return None
    if not adds:
        return _compose(text, edits)
    if a == 0:
        segs = _clauses(text, [[L(' ' + k, after='word')] for k in adds], a, bans={kwban}, edits=edits)
        if segs:
            return segs
    g = (2, 1, 2)[a]
    sents = [_sentence([[L(' ' + k, after='word')] for k in grp], a, lead=7, mid=6, tail=5, bans={kwban}) for grp in _chunks(adds, g)]
    return _place(text, sents, _mode(a), a, edits)

def b_person_names(text, kw, row, a, ctx):
    ad = ctx['ad']
    present = set(_names_present(text))
    need = _int(kw, 'N', 1) - len(present)
    if need <= 0:
        return None
    missing = _rot([nm for nm in NAMES if nm not in present], ctx.get('key', ''))
    single = [nm for nm in missing if ad.single_token(' ' + nm)]
    n_slots = min(need, len(single))
    items = []
    if n_slots:
        cname = 'names:' + '\x1f'.join(single)
        items += [[H(1, only=cname, distinct='names', after='word')] for _ in range(n_slots)]
    rest = [nm for nm in missing if nm not in single][:need - n_slots]
    items += [[L(' ' + nm, after='word')] for nm in rest]
    allnames = 'names:' + '\x1f'.join(NAMES)
    if a == 0:
        segs = _clauses(text, items, a, bans={allnames})
        if segs:
            return segs
    sents = [_sentence(grp, a, lead=6, mid=2, tail=5, bans={allnames}) for grp in _chunks(items, (3, 2, 3)[a])]
    return _place(text, sents, _mode(a), a)

def b_palindrome(text, kw, row, a, ctx):
    ad = ctx['ad']
    need = 10 - len(_palindromes(text))
    if need <= 0:
        return None
    words = ad.class_words('palin')
    n_slots = min(need, len(words))
    items = [[H(1, only='palin', distinct='palin', after='space')] for _ in range(n_slots)]
    have = set(_palindromes(text)) | set(words)
    rest = [w for w in PALINDROMES if w not in have][:need - n_slots]
    if len(rest) < need - n_slots:
        items += [[H(1, only='palin', after='space')] for _ in range(need - n_slots - len(rest))]
    items += [[L(' ' + w, after='space')] for w in rest]
    sents = [_sentence(grp, a, lead=6, mid=5, tail=5, bans={'palin'}) for grp in _chunks(items, (2, 1, 2)[a])]
    return _place(text, sents, _mode(a), a)

def b_unique_words(text, kw, row, a, ctx):
    d = _int(kw, 'N', 1) - len(_unique_words(text))
    if d <= 0:
        return None
    n = min(320, max(40, int(round((2.5 * d + 32) * SCALES[a]))))
    return _append_prefix(text, a) + [H(n, eos=True, eos_min=min(n - 4, max(16, int(1.5 * d))))]

def b_pronouns(text, kw, row, a, ctx):
    d = _int(kw, 'N', 1) - _pronoun_count(text)
    if d <= 0:
        return None
    n = min(320, max(40, int(round((5.0 * d + 32) * SCALES[a]))))
    return _append_prefix(text, a) + [H(n, eos=True, eos_min=min(n - 4, 4 * d), boost='pronoun@%s' % (1.5, 2.0, 1.75)[a])]

def _quote_stack(text):
    levels = []
    for ch in text:
        if levels and ch == levels[-1]:
            levels.pop()
        elif ch in '"\'':
            levels.append(ch)
    return levels

def b_quotes(text, kw, row, a, ctx):
    pre = _append_prefix(text, a)
    for x in pre:
        if x['k'] == 'H':
            x['ban'] = set(x.get('ban', ())) | {'quotes'}
    lv = _quote_stack(''.join((s['s'] for s in pre if s['k'] != 'H')))
    q1 = "'" if lv and lv[-1] == '"' else '"'
    q2 = '"' if q1 == "'" else "'"
    qb = {'quotes', 'nl'}
    return pre + [H(_sc(7, a), ban=qb), L(' ' + q1, after='nospace'), H(_sc(6, a), ban=qb), L(' ' + q2, after='nospace'), H(_sc(5, a), ban=qb), L(' ' + q1, after='nospace'), H(_sc(3, a), ban=qb, end_ban=True), L(q1), H(_sc(3, a), ban=qb, end_ban=True), L(q2), H(_sc(3, a), ban=qb, end_ban=True), L(q1), H(_sc(4, a), ban=qb, end_ban=True), L('.')]

def b_parentheses(text, kw, row, a, ctx):
    bb = {'brackets', 'nl'}
    segs = _append_prefix(text, a)
    for x in segs:
        if x['k'] == 'H':
            x['ban'] = set(x.get('ban', ())) | {'brackets'}
    segs.append(H(_sc(6, a), ban=bb))
    for o, n in zip('([{([', (4, 4, 3, 3, 3)):
        segs += [L(' ' + o, after='nospace'), H(_sc(n, a), ban=bb, end_ban=True)]
    for c in '])}])':
        segs += [L(c), H(_sc(2, a), ban=bb, end_ban=True)]
    segs[-1] = H(_sc(4, a), ban=bb, end_ban=True)
    return segs + [L('.')]

def _plf_first(line):
    """(normalised line, first word) as in ifb_transforms.t_paragraph_last_first: trailing punctuation glued to the
    first word can never be matched by a last word, so it is dropped from the first word."""
    core = line.strip().strip(PUNCT_SPACE)
    words = core.split()
    if not words:
        return (line, None)
    first = words[0]
    cf = first.rstrip(PUNCT)
    if cf and cf != first:
        k = line.find(first)
        line = line[:k] + cf + line[k + len(first):]
        first = cf
    return (line, first or None)

def b_paragraph_last_first(text, kw, row, a, ctx):
    ad = ctx['ad']
    segs, changed = ([], False)
    for li, line in enumerate(text.split('\n')):
        if li:
            segs.append(T('\n'))
        p = line.strip().lower()
        words = p.strip(PUNCT_SPACE).split()
        if not words or words[0] == words[-1]:
            segs.append(T(line))
            continue
        line2, first = _plf_first(line)
        if not first:
            segs.append(T(line))
            continue
        end_word = first.lower() if first.lower() in COMMON else first
        body = line2.rstrip()
        trail = line2[len(body):]
        spans = _sent_spans(body)
        ws = list(re.finditer('\\S+', body))
        if len(spans) >= 2:
            cut = spans[-1][0]
        elif len(ws) >= 6:
            cut = ws[int(math.ceil(0.5 * len(ws))) - 1].end()
        else:
            cut = len(body)
        head, tail_txt = (body[:cut].rstrip(), body[cut:])
        n = min(40, max(4, int(round(len(ad.encode_piece(tail_txt)) * SCALES[a]))))
        segs += [T(head), H(n, ban={'nl'}), L(' ' + end_word, after='word'), L('.'), T(trail)]
        changed = True
    return segs if changed else None

def b_punctuation(text, kw, row, a, ctx):
    need_inter, missing = _punct_missing(text)
    if not need_inter and (not missing):
        return None
    segs = _append_prefix(text, a)
    nb = {'nl'}
    mid = [c for c in ',;:' if c in missing]
    if mid or '.' in missing:
        segs.append(H(_sc(6, a), ban=nb))
        for c in mid:
            segs += [L(c), H(_sc(6, a), ban=nb, end_ban=True)]
        segs[-1]['end_ban'] = True
        segs.append(L('.'))
    if '?' in missing:
        segs += [H(_sc(8, a), ban=nb, end_ban=True), L('?')]
    if need_inter:
        segs += [H(_sc(8, a), ban=nb, end_ban=True), L('?!')]
    if '!' in missing:
        segs += [H(_sc(7, a), ban=nb, end_ban=True), L('!')]
    return segs

def b_numbers(text, kw, row, a, ctx):
    N = _int(kw, 'N', 0)
    units = _number_units(text)
    c = len(units)
    if c == N:
        return None
    if c > N:
        edits = []
        for s, e in units[N:]:
            line_start = text.rfind('\n', 0, s) + 1
            if text[line_start:s].strip() == '' and e < len(text) and (text[e] in '.)'):
                e += 1
            if s > 0 and text[s - 1] == ' ':
                s -= 1
            edits.append((s, e, [H(2, ban={'digits', 'nl'})]))
        return _compose(text, edits)
    slots = [[L(' '), H(1, only='digit1'), H(1, only='digit_or_boundary', after='word')] for _ in range(N - c)]
    if a == 0:
        segs = _clauses(text, slots, a, bans={'digits'}, boost='numwords@-3')
        if segs:
            return segs
    if a == 1:
        d = N - c
        n = min(320, max(40, 14 * d + 24))
        return _append_prefix(text, a) + [H(n, eos=True, eos_min=min(n - 4, 8 * d), boost='digit1@1.5', free_digits=True)]
    sents = [_sentence(grp, a, lead=7, mid=6, tail=5, bans={'digits'}, boost='numwords@-3') for grp in _chunks(slots, (2, 1, 2)[a])]
    return _place(text, sents, _mode(a), a)

def _nonlatin_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum((not _latin_ok(c) for c in letters)) / max(1, len(letters))

def ctx_bans(row, iid, text=''):
    """Bans for every hole of this row: whitespace-only tokens; other scripts (unless the response itself is mostly in
    another script or a language instruction is present); digits if count:numbers is a constraint; the keywords of
    count:keywords_multiple in holes written for other constraints."""
    ids = row.get('instruction_id_list', [])
    bans = {'nonlatin', 'spaces'}
    if any(('japanese' in i or 'language' in i for i in ids)) or _nonlatin_ratio(text) > 0.2:
        bans = {'spaces'}
    if 'count:numbers' in ids:
        bans.add('digits')
    if META.get(iid):
        bans.add(meta(iid))
    if 'count:keywords_multiple' in ids and iid != 'count:keywords_multiple':
        kw = clean_kw(row['kwargs'][ids.index('count:keywords_multiple')])
        ks = [str(kw.get('keyword%d' % i) or '').strip() for i in range(1, 6)]
        bans.add('kw:' + '\x1f'.join((k for k in ks if k)))
    return bans

def _sane(info, copy_n=12, bad_words=()):
    """Reject degenerate fills: replacement chars, >= 3 repeats of a content 4-gram, a word 3x in a row, a 12-word span copied
    from the visible context, required items making up > 35% of the added words (a dump), or (for longer additions)
    less than 30% letters."""
    added = ' '.join(info.get('regions', []))
    if '�' in added:
        return False
    words = re.findall("[A-Za-z']+", added.lower())
    ctxw = re.findall("[A-Za-z']+", info.get('context', '').lower())
    if info.get('n_items', 0) >= 3 and info['n_items'] > 0.35 * max(1, len(words)):
        return False
    if len(words) >= copy_n and len(ctxw) >= copy_n:
        seen = {tuple(ctxw[i:i + copy_n]) for i in range(len(ctxw) - copy_n + 1)}
        if any((tuple(words[i:i + copy_n]) in seen for i in range(len(words) - copy_n + 1))):
            return False
    lines = [ln for r in info.get('regions', []) for ln in r.split('\n') if ln.strip()]
    if len(lines) >= 4 and sum((len(ln.split()) for ln in lines)) / len(lines) <= 3.5:
        return False
    toks = added.split()
    if len(toks) >= 6 and len(_number_units(added)) > 0.2 * len(toks):
        return False
    if bad_words and any((w.strip("'") in bad_words for w in words)):
        return False
    if len(words) >= 8:
        all_g = Counter((tuple(words[i:i + 4]) for i in range(len(words) - 3)))
        grams = Counter({g: c for g, c in all_g.items() if sum((w not in STOP for w in g)) >= 2})
        if grams and grams.most_common(1)[0][1] >= 3 or all_g.most_common(1)[0][1] >= 4:
            return False
        run = 1
        for i in range(1, len(words)):
            run = run + 1 if words[i] == words[i - 1] else 1
            if run >= 3:
                return False
    letters = sum((ch.isalpha() for ch in added))
    return letters >= 0.3 * max(1, len(added.strip())) or len(added.strip()) < 12

def _make(iid, builder, copy_n=12):
    bad = set(META.get(iid, '').split())

    def fn(engine, row, text, kw, prompt, seed=0, oracle=None, ad=None):
        LAST.clear()
        LAST.update(iid=iid, path=None, tries=[])
        kw = clean_kw(kw)
        oracle = oracle or get_oracle()
        ad = ad or adapter(engine)
        if oracle.check(iid, kw, text, prompt):
            LAST['path'] = 'asis'
            return text
        ctx = dict(ad=ad, key=row.get('key', ''))
        bans = ctx_bans(row, iid, text)
        for a in range(ATTEMPTS):
            segs = builder(text, kw, row, a, ctx)
            if not segs:
                break
            out, info = fill(engine, prompt, segs, seed=seed + a, temperature=TEMPS[a], ctx_bans=bans, ad=ad)
            ok = oracle.check(iid, kw, out, prompt)
            sane = _sane(info, copy_n, bad)
            for r in range(REFINES):
                if ok or not sane:
                    break
                segs2 = builder(out, kw, row, a, ctx)
                if not segs2:
                    break
                out2, info2 = fill(engine, prompt, segs2, seed=seed + 100 + a, temperature=TEMPS[a], ctx_bans=bans, ad=ad)
                if not _sane(info2, copy_n, bad):
                    break
                info = dict(info2, regions=info['regions'] + info2['regions'], view=info['view'] + ' ⟫refine⟫ ' + info2['view'], n_masked=info['n_masked'] + info2['n_masked'], seconds=info['seconds'] + info2['seconds'])
                out, ok = (out2, oracle.check(iid, kw, out2, prompt))
            LAST['tries'].append(dict(attempt=a, ok=ok, sane=sane, view=info['view'], n_masked=info['n_masked'], seconds=round(info['seconds'], 2)))
            if ok and sane:
                LAST.update(path='infill' if a == 0 else 'infill_retry%d' % a, added=info['regions'], view=info['view'])
                return out
        if oracle.has_transform(iid):
            out, last = oracle.transform(iid, text, kw, prompt)
            LAST.update(path='template', template=last)
            return out
        LAST['path'] = 'none'
        return text
    fn.__name__ = 'infill_' + iid.replace(':', '_')
    return fn
BUILDERS = {'count:keywords_multiple': b_keywords, 'count:person_names': b_person_names, 'words:palindrome': b_palindrome, 'count:unique_word_count': b_unique_words, 'count:pronouns': b_pronouns, 'format:quotes': b_quotes, 'format:parentheses': b_parentheses, 'words:paragraph_last_first': b_paragraph_last_first, 'count:punctuation': b_punctuation, 'count:numbers': b_numbers}
INFILL = {iid: _make(iid, b, copy_n=16 if iid == 'words:paragraph_last_first' else 12) for iid, b in BUILDERS.items()}
