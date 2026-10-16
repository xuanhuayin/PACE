from dataclasses import dataclass
import numpy as np
from pace.automata.alphabet import (
    N_CLASSES,
    character_class,
    is_space_class,
    is_word_class,
    is_lower_class,
    is_upper_class,
    is_title_class,
    is_decimal_class,
    fold_rei,
    fold_lower,
)

DEAD = ("DEAD",)
PUNCT_FW = frozenset(map(ord, ".,?!'\""))
CONSTRAINED_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")


class Unsupported(ValueError):
    pass


@dataclass
class DFA:
    transition: np.ndarray
    accept: np.ndarray
    start: int
    states: tuple

    def run(self, text):
        state = self.start
        for char in text:
            state = int(self.transition[state, character_class(char)])
        return state

    def accepts_text(self, text):
        return bool(self.accept[self.run(text)])


def build(start, step, accept):
    states, index, rows = ([start], {start: 0}, [])
    cursor = 0
    while cursor < len(states):
        s = states[cursor]
        row = []
        for c in range(N_CLASSES):
            t = DEAD if s is DEAD else step(s, c)
            if t is None:
                t = DEAD
            if t not in index:
                index[t] = len(states)
                states.append(t)
            row.append(index[t])
        rows.append(row)
        cursor += 1
    acc = [False if s is DEAD else bool(accept(s)) for s in states]
    return DFA(np.asarray(rows, dtype=np.int32), np.asarray(acc, dtype=bool), 0, tuple(states))


def lower_strict(c):
    """str.lower() single-char image; İ lowers to two code points, so it never matches ASCII."""
    return -1 if c == 128 else fold_lower(c)


def kmp_table(pat):
    f, k = ([0] * len(pat), 0)
    for i in range(1, len(pat)):
        while k and pat[i] != pat[k]:
            k = f[k - 1]
        if pat[i] == pat[k]:
            k += 1
        f[i] = k
    return f


def kmp_advance(pat, f, k, c):
    while k and (k == len(pat) or pat[k] != c):
        k = f[k - 1]
    if k < len(pat) and pat[k] == c:
        k += 1
    return k


def nonempty():
    return build(0, lambda s, c: 1 if s == 1 or not is_space_class(c) else 0, lambda s: s == 1)


def literal_count(word, threshold, relation, whole_word):
    """Port of tess_lexical_dfa.literal_count_dfa (re.I, nonoverlapping, optional \\b) without the
    content flag."""
    letters = tuple((ord(ch) for ch in word.lower()))
    L = len(letters)
    cap = threshold + 1 if relation == "exact" else max(threshold, 1)

    def step(s, c):
        prefix, count, prev_word = s
        fc, wc = (fold_rei(c), is_word_class(c))
        after = count
        if whole_word:
            if prefix == L and (not wc):
                after = min(cap, count + 1)
            if 0 < prefix < L and fc == letters[prefix]:
                ap = prefix + 1
            elif not prev_word and fc == letters[0]:
                ap = 1
            else:
                ap = 0
            return (ap, after, wc)
        seq = letters[:prefix] + (fc,)
        if len(seq) == L and seq == letters:
            return (0, min(cap, count + 1), False)
        ap = max((n for n in range(1, L) if seq[-n:] == letters[:n]), default=0)
        return (ap, after, False)

    def accept(s):
        prefix, count, _ = s
        count = min(cap, count + int(whole_word and prefix == L))
        if relation == "exact":
            return count == threshold
        return count >= threshold if relation == "at least" else count < threshold

    return build((0, 0, False), step, accept)


def no_comma():
    return build(0, lambda s, c: DEAD if c == 44 else 0, lambda s: True)


def case_all(upper):
    """islower()/isupper() restricted to ASCII letters: the official checker also requires
    langdetect=='en', so non-ASCII letters/symbols (incl. uncased CJK) are forbidden here."""

    def step(s, c):
        if is_title_class(c) or (c >= 128 and (not is_space_class(c))):
            return DEAD
        if upper:
            if is_lower_class(c):
                return DEAD
            return 1 if s == 1 or is_upper_class(c) else 0
        if is_upper_class(c):
            return DEAD
        return 1 if s == 1 or is_lower_class(c) else 0

    return build(0, step, lambda s: s == 1)


def letter_count(letter, threshold, relation):
    if len(letter) != 1 or not "a" <= letter.lower() <= "z":
        raise Unsupported("letter_frequency with non-letter uses a random official fallback")
    target = ord(letter.lower())
    cap = max(threshold, 1)

    def step(s, c):
        return min(cap, s + 1) if fold_lower(c) == target else s

    return build(0, step, lambda s: s >= threshold if relation == "at least" else s < threshold)


def quotation():

    def step(s, c):
        phase, lastq, n = s
        if is_space_class(c):
            return s
        if phase == 0:
            return (1, True, 1) if c == 34 else DEAD
        return (1, c == 34, min(2, n + 1))

    return build((0, False, 0), step, lambda s: s[0] == 1 and s[1] and (s[2] >= 2))


def ends_with(phrase):
    pat = tuple((ord(ch) for ch in phrase.strip().lower()))
    if not pat or any((x > 127 for x in pat)):
        raise Unsupported("end phrase must be nonempty ASCII")
    f, L = (kmp_table(pat), len(pat))

    def step(s, c):
        k, flag, inws = s
        k2 = kmp_advance(pat, f, k, lower_strict(c))
        if is_space_class(c):
            return (k2, flag, True)
        if c == 34:
            if inws:
                flag = k == L
            return (k2, flag, False)
        return (k2, k2 == L, False)

    return build((0, False, False), step, lambda s: s[1])


def _substring_pattern(tokens):
    """tokens: list of (set_of_classes or None for optional whitespace)."""
    n = len(tokens)

    def closure(S):
        S = set(S)
        changed = True
        while changed:
            changed = False
            for i in list(S):
                if i < n and tokens[i] is None and (i + 1 not in S):
                    S.add(i + 1)
                    changed = True
        return S

    start = frozenset(closure({0}))
    if n in start:
        raise ValueError("Empty pattern")

    def step(S, c):
        if S == "ACC":
            return "ACC"
        nxt = {0}
        for i in S:
            if i >= n:
                continue
            tok = tokens[i]
            if tok is None:
                if is_space_class(c):
                    nxt.add(i + 1)
            elif c in tok:
                nxt.add(i + 1)
        nxt = closure(nxt)
        return "ACC" if n in nxt else frozenset(nxt)

    return build(start, step, lambda s: s == "ACC")


def _lower_set(ch):
    """Classes whose str.lower() image equals the ASCII char ch (for lowered-text patterns)."""
    o = ord(ch)
    return frozenset((c for c in range(N_CLASSES) if lower_strict(c) == o))


def postscript(marker):
    if marker == "P.S.":
        toks = [_lower_set("p"), _lower_set("."), None, _lower_set("s"), _lower_set(".")]
    elif marker == "P.P.S":
        toks = [
            _lower_set("p"),
            _lower_set("."),
            None,
            _lower_set("p"),
            _lower_set("."),
            None,
            _lower_set("s"),
        ]
    else:
        raise Unsupported("postscript marker " + repr(marker))
    return _substring_pattern(toks)


def title():
    """Sufficient condition: a line containing << ... g ... >> with g not in {<, >, whitespace}."""

    def step(s, c):
        if s == "ACC":
            return "ACC"
        if c == 10:
            return 0
        if s == 0:
            return 1 if c == 60 else 0
        if s == 1:
            return 2 if c == 60 else 0
        good = not (c == 60 or c == 62 or is_space_class(c))
        if s == 2:
            return 3 if good else 2
        if s == 3:
            return 4 if c == 62 else 3
        if s == 4:
            return "ACC" if c == 62 else 3
        raise AssertionError

    return build(0, step, lambda s: s == "ACC")


def highlights(n):

    def step(s, c):
        total, sg, dd = s
        nl, star, ws = (c == 10, c == 42, is_space_class(c))
        if sg is None:
            sg2 = (False,) if star else None
        elif star:
            if sg[0]:
                total = min(n, total + 1)
            sg2 = None
        elif nl:
            sg2 = None
        else:
            sg2 = (sg[0] or not ws,)
        if dd == 0:
            dd2 = 1 if star else 0
        elif dd == 1:
            dd2 = (2, True, False) if star else 0
        else:
            phase, empty, good = dd
            if phase == 2:
                if star:
                    dd2 = (3, empty, good)
                elif nl:
                    dd2 = 0
                else:
                    dd2 = (2, False, good or not ws)
            elif star:
                if good:
                    total = min(n, total + 1)
                dd2 = 0
            elif nl:
                dd2 = 0
            elif empty:
                dd2 = (2, False, not ws)
            else:
                dd2 = 0
        return (total, sg2, dd2)

    return build((0, None, 0), step, lambda s: s[0] >= n)


def placeholders(n):

    def step(s, c):
        count, inside = s
        if inside:
            if c == 93:
                return (min(n, count + 1), False)
            if c == 10:
                return (count, False)
            return s
        return (count, True) if c == 91 else s

    return build((0, False), step, lambda s: s[0] >= n)


def bullets(n):

    def step(s, c):
        total, a, b = s
        nl, ws, star, dash = (c == 10, is_space_class(c), c == 42, c == 45)
        if a in ("LS", "WS"):
            a2 = "WS" if ws else "STAR" if star else "FAIL"
        elif a == "STAR":
            if star:
                a2 = "FAIL"
            else:
                total = min(n + 1, total + 1)
                a2 = "IN"
        elif a == "IN":
            a2 = "LS" if nl else "IN"
        else:
            a2 = "LS" if nl else "FAIL"
        if b in ("LS", "WS"):
            if ws:
                b2 = "WS"
            elif dash:
                total = min(n + 1, total + 1)
                b2 = "IN"
            else:
                b2 = "FAIL"
        elif b == "IN":
            b2 = "LS" if nl else "IN"
        else:
            b2 = "LS" if nl else "FAIL"
        return (total, a2, b2)

    return build((0, "LS", "LS"), step, lambda s: s[0] == n)


def paragraphs(n):

    def step(s, c):
        closed, first_empty, content, stars = s
        if c == 42:
            if stars == 2:
                if not content:
                    if closed == 0:
                        first_empty = True
                    else:
                        return DEAD
                return (min(n + 2, closed + 1), first_empty, False, 0)
            return (closed, first_empty, content, stars + 1)
        if stars > 0 or not is_space_class(c):
            content = True
        return (closed, first_empty, content, 0)

    def accept(s):
        closed, first_empty, content, stars = s
        content = content or stars > 0
        return closed + 1 - int(first_empty) - int(not content) == n

    return build((0, False, False, 0), step, accept)


def nth_paragraph_first_word(n, nth, word):
    W = tuple((ord(ch) for ch in word.lower()))
    L = len(W)
    if not W or any((x > 127 for x in W)) or nth < 1 or (nth > n):
        raise Unsupported("nth_paragraph_first_word arguments")
    target = nth - 1

    def fw_char(fw, c):
        k = fw[1]
        return ("m", k + 1) if k < L and lower_strict(c) == W[k] else "fail"

    def fw_step(fw, c):
        if fw in ("-", "ok", "fail"):
            return fw
        ws = is_space_class(c)
        if fw == "ws":
            if ws:
                return "ws"
            if c == 39:
                return "apos"
            if c == 34:
                return "quot"
            return fw_char(("m", 0), c)
        if fw == "apos":
            if c == 39:
                return "apos"
            if c == 34:
                return "quot"
            if ws or c in PUNCT_FW:
                return "fail"
            return fw_char(("m", 0), c)
        if fw == "quot":
            if c == 34:
                return "quot"
            if ws or c in PUNCT_FW:
                return "fail"
            return fw_char(("m", 0), c)
        if ws or c in PUNCT_FW:
            return "ok" if fw[1] == L else "fail"
        return fw_char(fw, c)

    def finalize(fw):
        if fw == "ok":
            return True
        if isinstance(fw, tuple):
            return fw[1] == L
        return False

    def step(s, c):
        raw, cnt, cur, nl, fw, res = s
        if c == 10:
            if nl:
                if raw == target:
                    res = finalize(fw)
                if cur:
                    cnt = min(n + 1, cnt + 1)
                raw2 = min(nth, raw + 1)
                return (raw2, cnt, False, False, "ws" if raw2 == target else "-", res)
            return (raw, cnt, cur, True, fw_step(fw, c), res)
        return (raw, cnt, cur or not is_space_class(c), False, fw_step(fw, c), res)

    def accept(s):
        raw, cnt, cur, nl, fw, res = s
        total = min(n + 1, cnt + int(cur))
        ok = res if raw > target else finalize(fw) if raw == target else False
        return bool(ok) and total == n

    start = (0, 0, False, False, "ws" if target == 0 else "-", None)
    return build(start, step, accept)


def sections(splitter, n):
    pat = tuple((ord(ch) for ch in splitter))
    if (
        not pat
        or any((x > 127 for x in pat))
        or any((is_decimal_class(x) or is_space_class(x) for x in pat))
    ):
        raise Unsupported("section splitter")
    f, L = (kmp_table(pat), len(pat))

    def step(s, c):
        k, ph, cnt = s
        k2 = kmp_advance(pat, f, k, c)
        dig, ws = (is_decimal_class(c), is_space_class(c))
        if ph == "afterS":
            if dig:
                ph2, cnt = ("digits", min(n, cnt + 1))
            elif ws:
                ph2 = "afterS_ws"
            else:
                ph2 = "none"
        elif ph == "afterS_ws":
            if dig:
                ph2, cnt = ("digits", min(n, cnt + 1))
            else:
                ph2 = "none"
        elif ph == "digits":
            ph2 = "digits" if dig else "none"
        else:
            ph2 = "none"
        if k2 == L and ph2 != "digits":
            ph2 = "afterS"
        return (k2, ph2, cnt)

    return build((0, "none", 0), step, lambda s: s[2] >= n)


def constrained_response():
    pats = [tuple((ord(ch) for ch in o)) for o in CONSTRAINED_OPTIONS]
    fs = [kmp_table(p) for p in pats]

    def step(s, c):
        if s == "ACC":
            return "ACC"
        ks = tuple((kmp_advance(p, f, k, c) for p, f, k in zip(pats, fs, s)))
        return "ACC" if any((k == len(p) for k, p in zip(ks, pats))) else ks

    return build((0, 0, 0), step, lambda s: s == "ACC")


def repeat_prompt(prompt):
    pat = tuple((ord(ch) for ch in prompt.strip().lower()))
    if not pat or any((x > 127 for x in pat)):
        raise Unsupported("prompt_to_repeat must be nonempty ASCII")
    L = len(pat)

    def step(s, c):
        if s == "done":
            return "done"
        if s == "lead":
            if is_space_class(c):
                return "lead"
            s = 0
        if lower_strict(c) == pat[s]:
            return "done" if s + 1 == L else s + 1
        return DEAD

    return build("lead", step, lambda s: s == "done")


def word_count(n, relation):
    cap = n

    def step(s, c):
        count, inw = s
        if is_word_class(c):
            return (count if inw else min(cap, count + 1), True)
        return (count, False)

    return build((0, False), step, lambda s: s[0] >= n if relation == "at least" else s[0] < n)


def predicate(instruction_id, kw):
    kw = {k: v for k, v in (kw or {}).items() if v is not None}
    i = instruction_id
    if i == "keywords:existence":
        return [literal_count(w, 1, "at least", False) for w in sorted(kw["keywords"])]
    if i == "keywords:forbidden_words":
        return [literal_count(w, 1, "less than", True) for w in sorted(set(kw["forbidden_words"]))]
    if i == "keywords:frequency":
        return [literal_count(kw["keyword"].strip(), kw["frequency"], kw["relation"], False)]
    if i == "keywords:letter_frequency":
        return [letter_count(kw["letter"].strip(), kw["let_frequency"], kw["let_relation"])]
    if i == "punctuation:no_comma":
        return [no_comma()]
    if i == "change_case:english_lowercase":
        return [case_all(False)]
    if i == "change_case:english_capital":
        return [case_all(True)]
    if i == "startend:quotation":
        return [quotation()]
    if i == "startend:end_checker":
        return [ends_with(kw["end_phrase"])]
    if i == "detectable_content:postscript":
        return [postscript(kw["postscript_marker"].strip())]
    if i == "detectable_format:title":
        return [title()]
    if i == "detectable_format:number_highlighted_sections":
        return [highlights(kw["num_highlights"])]
    if i == "detectable_content:number_placeholders":
        return [placeholders(kw["num_placeholders"])]
    if i == "detectable_format:number_bullet_lists":
        return [bullets(kw["num_bullets"])]
    if i == "length_constraints:number_paragraphs":
        return [paragraphs(kw["num_paragraphs"])]
    if i == "length_constraints:nth_paragraph_first_word":
        return [
            nth_paragraph_first_word(kw["num_paragraphs"], kw["nth_paragraph"], kw["first_word"])
        ]
    if i == "detectable_format:multiple_sections":
        return [sections(kw["section_spliter"].strip(), kw["num_sections"])]
    if i == "detectable_format:constrained_response":
        return [constrained_response()]
    if i == "combination:repeat_prompt":
        return [repeat_prompt(kw["prompt_to_repeat"])]
    if i == "length_constraints:number_words":
        return [word_count(kw["num_words"], kw["relation"])]
    raise Unsupported(i)


UNSUPPORTED_IDS = (
    "combination:two_responses",
    "detectable_format:json_format",
    "language:response_language",
    "length_constraints:number_sentences",
    "change_case:capital_word_frequency",
)
PROGRAM_IDS = ("length_constraints:number_words",)


def components_for(row):
    comps, unsupported = ([], [])
    for idx, (iid, kw) in enumerate(zip(row["instruction_id_list"], row["kwargs"])):
        if iid in PROGRAM_IDS:
            unsupported.append((idx, iid, "program-handled"))
            continue
        try:
            for d in predicate(iid, kw):
                comps.append((idx, iid, d))
        except Unsupported as e:
            unsupported.append((idx, iid, str(e)))
    return (comps, unsupported)


def minimize(dfa):
    from pace.automata.compiler import minimize_text_dfa

    return minimize_text_dfa(dfa)


def product(dfas, max_states=100000):
    from pace.automata.inference import coaccessible

    components = []
    for d in dfas:
        transition, accept, start = minimize(d)
        components.append((transition, accept, start, coaccessible(transition, accept)))
    labels, representatives, sig_ids = ([], [], {})
    for c in range(N_CLASSES):
        sig = tuple((t[:, c].tobytes() for t, _, _, _ in components))
        if sig not in sig_ids:
            sig_ids[sig] = len(representatives)
            representatives.append(c)
        labels.append(sig_ids[sig])
    start = tuple((comp[2] for comp in components))
    if not all((comp[3][s] for comp, s in zip(components, start))):
        start = None
    states, index, rows, accepted = ([start], {start: 0}, [], [])
    cursor = 0
    while cursor < len(states):
        state = states[cursor]
        accepted.append(
            state is not None and all((comp[1][s] for comp, s in zip(components, state)))
        )
        row = []
        for c in representatives:
            after = (
                None
                if state is None
                else tuple((int(comp[0][s, c]) for comp, s in zip(components, state)))
            )
            if after is not None and (not all((comp[3][s] for comp, s in zip(components, after)))):
                after = None
            if after not in index:
                if len(states) >= max_states:
                    raise ValueError("Joint text state limit")
                index[after] = len(states)
                states.append(after)
            row.append(index[after])
        rows.append([row[l] for l in labels])
        cursor += 1
    return DFA(np.asarray(rows, dtype=np.int32), np.asarray(accepted, dtype=bool), 0, tuple(states))
